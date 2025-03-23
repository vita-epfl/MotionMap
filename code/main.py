import os
import sys
import signal
import logging
import torch
import numpy as np


# Our code is based on BeLFusion, hence we borrow data preprocessing from there
# Barquero et al., BeLFusion, ICCV 2023 https://barquerogerman.github.io/BeLFusion/ 
sys.path.append(os.path.join(os.getcwd(), 'belfusion'))


# Set the cache directory for numba kernels
# Numba is used by OpenTSNE
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the parent directory
numba_cache_dir = os.path.join(parent_dir, "numba_kernel_cache")  # Path to the cache directory
os.makedirs(numba_cache_dir, exist_ok=True)  # Create the directory if it doesn't exist
os.environ["NUMBA_CACHE_DIR"] = numba_cache_dir  # Set the environment variabl


import autoencoder
import motionmap
import multimodal
from config import ParseConfig


from dataset import MultimodalDataset, MotionMapDataset, FineTuneDataset
from utilities import fit_reduction, interrupt_handler
from utilities import create_training_package, load_training_package, save_training_package


logging.getLogger().setLevel(logging.INFO)


def main():
    """
    Main function for the Human Pose Forecasting project.

    This function orchestrates the entire pipeline for training, evaluating, and visualizing 
    the autoencoder model, MotionMap model, and multimodal inference. It also handles dataset 
    preprocessing, dimensionality reduction, and saving/loading model checkpoints.

    Workflow:
    1. Load configuration settings.
    2. Define, load, and save models.
    3. Train, evaluate, and visualize the autoencoder model.
       3.1. Obtain embeddings and reduce dimensionality if required.
       3.2. Mapping between embeddings and pose sequences for visualization if autoencoder->reduce is True.
    4. Train, evaluate, and visualize the MotionMap model.
    5. Perform multimodal finetuning, visualization and testing.

    Notes:
    ------
    - The function exits early if no further models need to be trained, visualized, or evaluated.
    - The model saves both autoencoder and motionmap model after every step

    Example Usage:
    --------------
    Run the script directly to execute the pipeline:
        $ python main.py
    """
    
    conf = ParseConfig()

    # 1. Define, load and save the models ---------------------------------------------------------------
    autoencoder_dict = create_training_package(conf, 'autoencoder')
    motionmap_dict = create_training_package(conf, 'motionmap')

    try: autoencoder_dict = load_training_package(autoencoder_dict, conf.load_path, type='autoencoder')
    except FileNotFoundError: logging.info('Could not find autoencoder to load in: {}'.format(conf.load_path))

    try: motionmap_dict = load_training_package(motionmap_dict, conf.load_path, type='motionmap')
    except FileNotFoundError: logging.info('Could not find the MotionMap model to load in: {}'.format(conf.load_path))

    save_training_package(autoencoder_dict, conf.save_path, type='autoencoder')
    save_training_package(motionmap_dict, conf.save_path, type='motionmap')
        
    # (Internal usage - GPU cluster): Setup interrupt signals for auto-restart
    # Watch for interrupt signal sent by kubernetes (SIGTERM) or user (SIGINT)
    signal.signal(signal.SIGTERM, interrupt_handler(conf.save_path))
    signal.signal(signal.SIGINT, interrupt_handler(conf.save_path))

    print('-- Finished setting up interrupt signals.')

    # Multimodal dataset is used for both training the autoencoder model as well as evaluation
    multimodal_dataset = MultimodalDataset(conf)
    
    
    #1. Autoencoder Model ----------------------------------------------------------------------------
    print('\n #### [MAIN]: Autoencoder Model ####\n')

    if conf.autoencoder['train']:

        print('\n[MAIN]: Training autoencoder.')
        multimodal_dataset.preprocess('train', conf.num_frames)
        autoencoder_dict = autoencoder.train(autoencoder_dict, multimodal_dataset, conf)
        print('[MAIN]: Completed training autoencoder.\n')

    # Visualizes the quality of autoencoding
    if conf.autoencoder['visualize']:
        print('\n[MAIN]: Visualizing autoencoder (train split).')
        multimodal_dataset.preprocess('train', conf.num_frames)
        autoencoder.visualize(autoencoder_dict, multimodal_dataset, conf, split='train')
        print('[MAIN]: Completed visualizing autoencoder (train split).\n')

        print('\n[MAIN]: Visualizing autoencoder (test split).')
        multimodal_dataset.preprocess('test', conf.num_frames)
        autoencoder.visualize(autoencoder_dict, multimodal_dataset, conf, split='test')
        print('[MAIN]: Completed visualizing autoencoder (test split).\n')

    if conf.autoencoder['evaluate']:
        # Results may differ sometimes due to randomness in returning mm_y

        print('\n[MAIN]: Evaluating autoencoder with training split.')
        multimodal_dataset.preprocess('train', conf.num_frames)
        autoencoder.evaluate(autoencoder_dict, multimodal_dataset, conf, split='train')
        print('[MAIN]: Completed evaluating autoencoder with training split.\n')

        print('\n[MAIN]: Evaluating autoencoder with testing split.')
        multimodal_dataset.preprocess('test', conf.num_frames)
        autoencoder.evaluate(autoencoder_dict, multimodal_dataset, conf, split='test')
        print('[MAIN]: Completed evaluating autoencoder with testing split.\n')

    # Get train and test embeddings using E_y
    # The projection method ignores the stride
    # Action_labels is used to show the figure on controllability
    print('\n[MAIN]: Obtaining embeddings with training split.')
    multimodal_dataset.preprocess('train', conf.num_frames)
    train_embeddings, actions_train = autoencoder.projection(autoencoder_dict, multimodal_dataset, conf)
    print('[MAIN]: Completed obtaining embeddings with training split.\n')

    print('\n[MAIN]: Obtaining embeddings with testing split.')
    multimodal_dataset.preprocess('test', conf.num_frames)
    test_embeddings, actions_test = autoencoder.projection(autoencoder_dict, multimodal_dataset, conf)
    print('[MAIN]: Completed obtaining embeddings with testing split.\n')

    # Get two dimensional fitting if not already low dimension
    stride = conf.dataset_settings[conf.dataset]['mmgt_stride']

    if (train_embeddings.shape[-1] not in [2, 3]) and (autoencoder_dict['reduction'] is None):
        print('\n[MAIN]: Fitting dimensionality reduction algorithm.')
        
        # We reduce with stride since reducing over all sequences is computationally expensive
        # Moreover, i and i+1 frames are very similar any which ways
        autoencoder_dict['reduction'] = fit_reduction(
            train_embeddings[::stride], dimensions=2, algorithm=conf.reduce_algorithm)
        print('[MAIN]: Completed fitting dimensionality reduction algorithm.\n')
    
    # Save the model with fitted reduction
    save_training_package(autoencoder_dict, conf.save_path, type='autoencoder')
        
    # Transform into two dimensions
    if train_embeddings.shape[-1] not in [2, 3]:
        if conf.reduce_algorithm == 'tsne':
            len_train = len(train_embeddings)
            # It is faster to reduce all of them together (fitting is already done on train)
            all_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)

            all_dr = autoencoder_dict['reduction'].transform(all_embeddings, perplexity=20)
            train_dr, test_dr = all_dr[:len_train], all_dr[len_train:]
            del all_dr

        else:
            train_dr = autoencoder_dict['reduction'].transform(train_embeddings)
            test_dr = autoencoder_dict['reduction'].transform(test_embeddings)

    else:
        train_dr = train_embeddings
        test_dr = test_embeddings
    
    # Convert to tensor if not tensor
    train_embeddings = torch.as_tensor(train_embeddings).float()
    test_embeddings = torch.as_tensor(test_embeddings).float()
    train_dr = torch.as_tensor(train_dr).float()
    test_dr = torch.as_tensor(test_dr).float()

    # This is meant to visualize the 2-D embedding and associated pose sequences interactively.
    # The demo shows uses this to show the controllability of the model.
    if conf.autoencoder['reduce']:
        print('\n[MAIN]: Obtaining reduced dimension with training split.')
        multimodal_dataset.preprocess('train', conf.num_frames)
        autoencoder.projection(
            autoencoder_dict, multimodal_dataset, conf, viz=True)
        print('[MAIN]: Completed obtaining reduced dimension with training split.\n')

    # Exit code condition- if no further models to train, visualize or evaluate
    if not (conf.motionmap['train'] or conf.motionmap['evaluate']  or \
            conf.multimodal['finetune'] or conf.multimodal['visualize'] or conf.multimodal['evaluate']):
        print('\n[MAIN]: Exiting code. No further models to train, visualize or evaluate.\n')
        exit()


    #2. MotionMap Model -----------------------------------------------------------------------------
    save_training_package(autoencoder_dict, conf.save_path, type='autoencoder')

    print('\n #### [MAIN]: MotionMap Model ####\n')

    # Arrange embeddings for both train and test splits
    z_reduced = dict()
    z_reduced['train'] = (train_embeddings, train_dr)
    z_reduced['test'] = (test_embeddings, test_dr)

    print('[MAIN]: Initializing MotionMap dataset.')
    actions = [actions_train, actions_test]
    motionmap_dataset = MotionMapDataset(z_reduced, actions, conf)

    # Get and save template sequence
    # The template sequence is used to normalize all pose sequences w.r.t this
    # Having a template sequence makes MotionMap invariant to the skeletal size
    motionmap_dataset.preprocess('train', conf.num_frames)
    motionmap_dataset.augmentation = False
    x_template, _ = motionmap_dataset.__getitem__(0)
    x_template = torch.from_numpy(x_template).cuda()
    motionmap_dict['x_template'] = x_template
    print('[MAIN]: Completed initializing MotionMap dataset.')

    if conf.motionmap['train']:
        print('\n[MAIN]: Training MotionMap model.')
        motionmap_dataset.preprocess('train', conf.num_frames)
        motionmap_dict = motionmap.train(
            motionmap_dict=motionmap_dict, dataset=motionmap_dataset, conf=conf)
        print('[MAIN]: Completed training MotionMap model.\n')
    
        # Deep copy does not copy x_template hence we need to save it again
        motionmap_dict['x_template'] = x_template
    
    if conf.motionmap['evaluate']:
        
        print('\n[MAIN]: Evaluating MotionMap model on training split.')
        motionmap_dataset.preprocess('train', conf.num_frames)
        motionmap.evaluate(motionmap_dict=motionmap_dict,dataset=motionmap_dataset, conf=conf, split='train')
        print('[MAIN]: Completed evaluating MotionMap model on training split.\n')
        
        print('\n[MAIN]: Evaluating MotionMap model on testing split.')
        motionmap_dataset.preprocess('test', conf.num_frames)
        motionmap.evaluate(motionmap_dict=motionmap_dict, dataset=motionmap_dataset, conf=conf, split='test')
        print('[MAIN]: Completed evaluating MotionMap model on testing split.\n')

    # Exit code condition:
    if not (conf.multimodal['finetune'] or conf.multimodal['visualize'] or conf.multimodal['evaluate']):
        print('\n[MAIN]: Exiting code. No further models to train, visualize or evaluate.\n')
        exit()


    #3. Inference Model ---------------------------------------------------------------------------
    
    print('\n #### [MAIN]: Multimodal Inference ####\n')

    if conf.multimodal['finetune']:
        print('\n[MAIN]: Finetuning autoencoder model.')

        finetune_dataset = FineTuneDataset(motionmap_dataset, conf)
        finetune_dataset.preprocess('train', conf.num_frames)

        autoencoder_dict = multimodal.train(autoencoder_dict, finetune_dataset, conf)
        motionmap_dataset.codebook = finetune_dataset.codebook
        print('[MAIN]: Completed finetuning autoencoder model.\n')

    if conf.multimodal['visualize']:
        print('\n[MAIN]: Visualizing Multimodality with training split.')

        motionmap_dataset.preprocess('train', conf.num_frames)
        multimodal_dataset.preprocess('train', conf.num_frames)
        multimodal.visualize(
            autoencoder_dict, motionmap_dict, multimodal_dataset, motionmap_dataset, conf, split='train')
        print('[MAIN]: Completed visualizing multimodality with training split.\n')
        
        print('\n[MAIN]: Visualizing Multimodality with testing split.')
        motionmap_dataset.preprocess('test', conf.num_frames)
        multimodal_dataset.preprocess('test', conf.num_frames)
        multimodal.visualize(
            autoencoder_dict, motionmap_dict, multimodal_dataset, motionmap_dataset, conf, split='test')
        print('[MAIN]: Completed visualizing multimodality with testing split.\n')
        
    if conf.multimodal['evaluate']:

        print('\n[MAIN]: Evaluating Multimodality with testing split.')
        motionmap_dataset.preprocess('test', conf.num_frames)
        multimodal_dataset.preprocess('test', conf.num_frames)
        multimodal.evaluate(
            autoencoder_dict, motionmap_dict, multimodal_dataset, motionmap_dataset, conf, split='test')
        print('[MAIN]: Completed evaluating Multimodality with testing split.\n')
        
    
    save_training_package(autoencoder_dict, conf.save_path, type='autoencoder')
    save_training_package(motionmap_dict, conf.save_path, type='motionmap')


if __name__ == '__main__':
    main()