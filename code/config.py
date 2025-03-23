import os
import sys
import yaml
import shutil
import logging
from pathlib import Path


class ParseConfig(object):
    """
    ParseConfig Class: Handles configuration loading, auto-restart, and initialization for the Human Pose Forecasting project.

    This class is responsible for:
    1. Loading the configuration from `configuration.yml`.
    2. Handling auto-restart functionality for interrupted runs.
    3. Initializing dataset-specific settings, model architecture, and experiment parameters.
    4. Creating directories for saving model outputs and copying the codebase for reproducibility.
    5. Configuring the number of workers based on the computing environment.

    Notes:
    ------
    - The class assumes the presence of a `configuration.yml` file in the working directory.
    - Auto-restart functionality is designed for GPU cluster environments and uses `interrupt.txt` 
      to track interrupted runs.
    - The save path includes a copy of the current codebase for reproducibility.
    """
    def __init__(self) -> None:

        # 1. Load the current configuration file ------------------------------------------------------------------------------
        try:
            f = open('configuration.yml', 'r')
            conf_yml = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        except FileNotFoundError:
            logging.warning('Could not find configuration.yml')
            sys.exit(0)


        # 2. Handling auto-restart -------------------------------------------------------------------------------------
        # Please ignore, this is for internal use to work with our GPU cluster.
        # We handle auto-restart by loading the configuration.yml for an interrupted run.
        # We then set the configuration.yml load_path to its own location to load latest model.
        priority = conf_yml['auto_restart']['priority']
        priority_path = conf_yml['auto_restart']['priority_path']

        if priority == 0:
            logging.info('Resuming interrupted executions (if any).')
           
            # Read from interrupt.txt
            try:
                # If can't open, file doesn't exist. This means no interruptions to resume.
                with open('interrupt.txt', 'r') as f:
                    locations = list(set(f.read().splitlines()))
                    
            except FileNotFoundError:
                logging.info('Could not find interrupt.txt and no interruptions to resume.\nExiting.')
                sys.exit(0)

            if priority_path is not None:
                try:
                    locations.remove(priority_path)
                    load_path = priority_path

                # Priority path is either not interrupted or already resumed by another process
                except ValueError:
                    logging.warning('Could not find priority path in interrupt.txt')
                    load_path = locations.pop(0)
            
            else:
                logging.info('No priority path specified. Resuming the first interrupted run.')
                load_path = locations.pop(0)

            # Read from the location of the first (or priority) interrupted run
            with open(os.path.join(load_path, 'code', 'configuration.yml'), 'r') as f:
                conf_yml = yaml.load(f, Loader=yaml.FullLoader)
                
                # We want to load latest model which is stored in that directory
                conf_yml['load_path'] = load_path
                
                # Write the remaining locations back to interrupt.txt
                if len(locations) == 0:
                    logging.info('All interrupted locations have been processed. Removing interrupt.txt.')
                    os.remove('interrupt.txt')

                else:
                    # Overwrite the file with the remaining locations
                    with open('interrupt.txt', 'w') as f:
                        for loc in locations:
                            f.write(loc + '\n')

        else:
            assert priority == 1, 'Priority should be either 0 or 1.'
            logging.info('Starting a new run.\n')


        # 2. Initializing ParseConfig object --------------------------------------------------------------------------
        self.dataset = conf_yml['dataset']       # Human36M or AMASS
        self.load_path = conf_yml['load_path']   # Previously saved model location
        
        # The three stages from Fig. 2 in the paper
        self.autoencoder = conf_yml['autoencoder']
        self.motionmap = conf_yml['motionmap']
        self.multimodal = conf_yml['multimodal']

        # t-SNE since PCA is linear. But this choice exists for visualization purposes only.
        self.reduce_algorithm = conf_yml['reduce_algorithm'] # PCA or t-SNE

        # Epochs, LR, batch size, etc.
        self.experiment_settings = conf_yml['experiment_settings']

        # 3. Set configurations manually -------------------------------------------------------------------------------
        # Refers to projection size used in E_x, E_y, D
        self.encoder = {
            "dimensions": 128,
            "projection": 128
        }
        
        # This is the same as BeLFusion and not changed hence defining it here.
        self.architecture = {
            "encoder": {
                "nh_mlp": [300, 200, 200],
                "nh_rnn": -1,  # Specified in dimensions
                "recurrent": True,
                "residual": True,
                "rnn_type": "gru",
                "use_drnn_mlp": True,
                "dropout": 0.,
                "n_landmarks": -1, #16, 21 - H36M, AMASS
                "n_features": 3,
                "obs_length": -1,  #100, 120 - H36M, AMASS
                "pred_length": -1  #100, 120 - H36M, AMASS
            },
            "pose": {
                "n_landmarks": -1,  #16, 21 - H36M, AMASS
                "n_features": 3,
                "obs_length": -1,   #100, 120 - H36M, AMASS
                "pred_length": -1,  #100, 120 - H36M, AMASS
                "hidden_dim": -1    # Same as autoencoder dimensions
            }
        }
        
        self.architecture['encoder']['nh_rnn'] = self.encoder['dimensions']
        self.architecture['pose']['hidden_dim']  = self.encoder['dimensions']
        self.architecture['pose']['encoder_arch'] = self.architecture['encoder']
        self.architecture['pose']['projection'] = self.encoder['projection']

        # Dataset settings used to determine how to construct the ground truth and augmentation related stuff
        global NUM_FRAMES, PATH
        NUM_FRAMES = 3  # Follows our new definition of the multimodal ground truth
        self.num_frames = NUM_FRAMES
        PATH = conf_yml['dataset_path']

        # mmgt_stride: multimodal ground truth will be atleast this many frames apart
        # stride: __getitem__ indexing will be i * stride
        # augmentation: +/- this many frames will be added to i * stride
        self.dataset_settings = {
            "Human36M": {"mmgt_stride": 5, "stride": 10, "augmentation": 5},
            "AMASS": {"mmgt_stride": 30, "stride": 60, "augmentation": 30}
            }

        assert self.dataset in ['Human36M', 'AMASS'], "Dataset not in {Human36M, AMASS}"
        self.architecture['encoder']['pred_length'] = 100 if self.dataset == 'Human36M' else 120
        self.architecture['encoder']['obs_length'] = 100 if self.dataset == 'Human36M' else 120
        self.architecture['encoder']['n_landmarks'] = 16 if self.dataset == 'Human36M' else 21

        self.architecture['pose']['pred_length'] = 100 if self.dataset == 'Human36M' else 120
        self.architecture['pose']['obs_length'] = 100 if self.dataset == 'Human36M' else 120
        self.architecture['pose']['n_landmarks'] = 16 if self.dataset == 'Human36M' else 21
        self.architecture['pose']['context_length'] = 25 if self.dataset == 'Human36M' else 30
        
        self.context_length = self.architecture['pose']['context_length']

        # 3. Create directory for model save path ----------------------------------------------------------------------
        self.experiment_name = conf_yml['experiment_name']
            
        i = 1
        # FYI: os.path.join does max(pathA, pathB) if both paths start from root.
        model_save_path = os.path.join(conf_yml['save_path'], self.experiment_name + '_' + str(i))
            
        while os.path.exists(model_save_path):
            i += 1
            model_save_path = os.path.join(conf_yml['save_path'], self.experiment_name + '_' + str(i))

        logging.info('Saving the model at: ' + model_save_path)
        self.save_path = model_save_path

        # Copy the configuration file into the model dump path
        code_directory = Path(os.path.abspath(__file__)).parent
        shutil.copytree(src=str(code_directory),
                        dst=os.path.join(model_save_path, code_directory.parts[-1]))


        # 4. Define number of workers depending on computing node ---------------------------------------------------
        if conf_yml['gpu_cluster']:
            # Workstation
            self.num_workers = 32
        else:
            # Central Computing
            self.num_workers = 2

        print('Configuration loaded successfully.')