import os
import logging

from rich.progress import track

import torch
from matplotlib import pyplot as plt

from utilities import motion_transfer
from utilities import save_training_package, deepcopy_training_package


def train(motionmap_dict, dataset, conf):
    """
    Train function for the MotionMap model.

    This function trains the MotionMap model using the provided dataset and configuration settings.

    Workflow:
    1. Iterates through epochs and batches to train the model.
    2. Computes the loss using BCEWithLogitsLoss and backpropagates.
    3. Saves the model if the current epoch achieves a lower loss than previously saved.
    4. Saves visualizations of predicted and ground truth heatmaps for the first batch of each epoch.

    Parameters:
    -----------
    motionmap_dict : dict
        Dictionary containing the MotionMap model, optimizer, scheduler, and other training components.
    dataset : MotionmapDataset
        Dataset object for training the MotionMap model.
    conf : dict
        Configuration object containing experiment settings and paths.

    Returns:
    --------
    dict
        The best MotionMap model dictionary after training.
    """

    logging.info('Training MotionMap model.')

    dataset.strided = True
    dataset.use_augmentation = True
    
    end_epoch = conf.experiment_settings['epochs']
    batch_size = conf.experiment_settings['batch_size']

    start_epoch = motionmap_dict['epoch']
    motionmap_model = motionmap_dict['model']
    optim = motionmap_dict['optimizer']
    scheduler = motionmap_dict['scheduler']
    loss_saved = motionmap_dict['loss']
    x_template = motionmap_dict['x_template']

    motionmap_dict_best = deepcopy_training_package(motionmap_dict, conf, type='motionmap')

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=conf.num_workers)

    num_samples = len(dataset)

    pos_weight = torch.tensor([conf.motionmap['posWt']], device='cuda')
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    for e in range(start_epoch, end_epoch):
        logging.info('Epoch: {}'.format(e+1))

        os.makedirs(os.path.join(conf.save_path, 'motionmap/epoch_{}'.format(e)), exist_ok=True)

        save_images = True
        loss_epoch = 0.

        motionmap_model.train()

        for x_sample, hm in track(train_loader):

            x_sample = x_sample.cuda()
            hm = hm.cuda()

            # Transfer the motion to the reference skeleton
            x, _, _ = motion_transfer(
                skeletal_reference=x_template.expand(x_sample.shape[0], -1, -1, -1, -1),
                motion_reference=x_sample, dataset=conf.dataset)

            hm_pred = motionmap_model(x)

            loss = loss_fn(hm_pred, hm).sum()
            loss.backward()

            optim.step()
            optim.zero_grad()

            loss_epoch += loss.item()

            if save_images:
                for j in range(x_sample.shape[0]):
                    fig, axs = plt.subplots(2)
                    axs[0].imshow(hm[j].detach().cpu().numpy())
                    axs[1].imshow(torch.nn.Sigmoid()(hm_pred[j]).detach().cpu().numpy())
                    plt.savefig(os.path.join(conf.save_path, 'motionmap',
                                             'epoch_{}/{}.png'.format(e, j)), dpi=150)
                    plt.close()
                save_images = False

        loss_epoch /= num_samples
        scheduler.step(loss_epoch)
        logging.info('Loss epoch: {}\n'.format(loss_epoch))
        
        with open(os.path.join(conf.save_path, "loss_motionmap.txt"), "a+") as f:
            print('Loss epoch: {}\n'.format(loss_epoch), file=f)

        # Save the model at every epoch
        if loss_epoch < loss_saved:
            logging.info('Saving the model at the end of epoch: {}'.format(e+1))
            motionmap_dict['epoch'] = e + 1
            motionmap_dict['loss'] = loss_epoch

            save_training_package(motionmap_dict, conf.save_path, type='motionmap')
            
            loss_saved = loss_epoch
            motionmap_dict_best = deepcopy_training_package(motionmap_dict, conf, type='motionmap')

    logging.info('Training completed successfully.')
    logging.info('Models saved at: {}/motionmap.pt'.format(conf.save_path))
    logging.info('Best model epoch: {}'.format(motionmap_dict_best['epoch']))

    return motionmap_dict_best


@torch.no_grad()
def evaluate(motionmap_dict, dataset, conf, split):
    """
    Evaluate the MotionMap model.

    This function evaluates the MotionMap model on the specified dataset split by computing 
    the loss and saving visualizations of predicted and ground truth heatmaps.

    Workflow:
    1. Iterates through the dataset to compute the loss.
    2. Saves visualizations of predicted and ground truth heatmaps for the first batch.
    3. Logs and saves the evaluation loss.

    Parameters:
    -----------
    motionmap_dict : dict
        Dictionary containing the MotionMap model and related components.
    dataset : MotionmapDataset
        Dataset object for evaluation.
    conf : dict
        Configuration object containing experiment settings and paths.
    split : str
        Dataset split to evaluate (e.g., 'train', 'test').

    Returns:
    --------
    None
    """

    logging.info('Evaluating MotionMap model.')

    batch_size = conf.experiment_settings['batch_size']

    dataset.strided = False
    dataset.use_augmentation = False

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=conf.num_workers)

    loss_epoch = 0.
    num_samples = len(dataset)

    save_images = True
    os.makedirs(os.path.join(conf.save_path, '{}_motionmap'.format(split)), exist_ok=True)
    
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    x_template = motionmap_dict['x_template']

    motionmap_model = motionmap_dict['model']
    motionmap_model.eval()

    for x_sample, hm in track(loader):

        x_sample = x_sample.cuda()
        hm = hm.cuda()

        # Transfer the motion to the reference skeleton
        x, _, _ = motion_transfer(
            skeletal_reference=x_template.expand(x_sample.shape[0], -1, -1, -1, -1),
            motion_reference=x_sample, dataset=conf.dataset)

        hm_pred = motionmap_model(x)

        loss = loss_fn(hm_pred, hm).sum(dim=(1, 2))
        loss = loss.sum()
        loss_epoch += loss.item()

        if save_images:
            for j in range(x_sample.shape[0]):
                fig, axs = plt.subplots(2)
                axs[0].imshow(hm[j].detach().cpu().numpy())
                axs[1].imshow(torch.nn.Sigmoid()(hm_pred[j]).detach().cpu().numpy())
                plt.savefig(os.path.join(conf.save_path, '{}_motionmap'.format(split),
                                         '{}.png'.format(j)), dpi=200)
                plt.close()
            save_images = False

    loss_epoch /= num_samples
    logging.info('Loss ({}): {}\n'.format(split, loss_epoch))
        
    with open(os.path.join(conf.save_path, "loss_motionmap.txt"), "a+") as f:
        print('Loss ({}): {}\n'.format(split, loss_epoch), file=f)

    # Save the model
    save_training_package(motionmap_dict, conf.save_path, type='motionmap')

    logging.info('Evaluation completed successfully.')
    logging.info('Models saved at: {}/motionmap.pt'.format(conf.save_path))

    return None


@torch.no_grad()
def save_heatmaps(dataset, split, save_path):
    """
    Generate and save heatmaps for the specified dataset split.

    This function processes the dataset to generate heatmaps and saves them as a 
    PyTorch tensor file for later use.

    Parameters:
    -----------
    dataset : MotionmapDataset
        Dataset object for generating heatmaps.
    split : str
        Dataset split to process (e.g., 'train', 'test').
    save_path : str
        Path to save the generated heatmaps.

    Returns:
    --------
    None
    """
    print('Generating heatmaps for split: {}.'.format(split))

    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    hms = list()

    for _, hm in track(loader):
        hms.append(hm)
    hms = torch.cat(hms, dim=0)
    torch.save(hms, os.path.join(save_path, 'hms_{}.pt'.format(split)))
    print('Saved the heatmaps at: ', save_path)
