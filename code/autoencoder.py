import os
import random
import string
import logging
from rich.progress import track

import torch
import numpy as np

from utilities import motion_transfer
from utilities import save_training_package, deepcopy_training_package
from visualizer import Visualizer


def train(autoencoder_dict, dataset, conf):
    """
    Train function for the autoencoder model.

    This function trains the autoencoder model using the provided dataset and configuration settings.

    Workflow:
    1. Iterates through epochs and batches to train the model.
    2. Computes the loss using the uncertainty model and backpropagates.
    3. Saves the model if the current epoch achieves a lower loss than previously saved.

    Parameters:
    -----------
    autoencoder_dict : dict
        Dictionary containing the autoencoder model, optimizer, scheduler, and other training components.
    dataset : MultimodalDataset
        Dataset object for training the autoencoder.
    conf : dict
        Configuration object containing experiment settings and paths.

    Returns:
    --------
    dict
        The best autoencoder model dictionary after training.
    """

    logging.info('Training autoencoder.')
    
    dataset.strided = True
    dataset.use_augmentation = True
    
    end_epoch = conf.experiment_settings['epochs']
    batch_size = conf.experiment_settings['batch_size']

    start_epoch = autoencoder_dict['epoch']
    autoencoder_model = autoencoder_dict['model']
    optim = autoencoder_dict['optimizer']
    scheduler = autoencoder_dict['scheduler']
    loss_saved = autoencoder_dict['loss']
    uncertainty_model = autoencoder_dict['uncertainty']

    autoencoder_dict_best = deepcopy_training_package(autoencoder_dict, conf, type='autoencoder')

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=conf.num_workers)

    num_samples = len(dataset)

    for e in range(start_epoch, end_epoch // 2):
        logging.info('Epoch: {}'.format(e+1))

        loss_epoch = 0.

        autoencoder_model.train()
        uncertainty_model.train()

        for x, mm_y, _, _ in track(train_loader):

            x = x.cuda()
            mm_y = mm_y.cuda()

            # Described in Fig 3 of the paper
            mm_y, _, _ = motion_transfer(
                skeletal_reference=x, motion_reference=mm_y, dataset=conf.dataset)

            mm_y_hat, latent = autoencoder_model(x, mm_y)
            # We are predicting x + y
            mm_y = torch.cat([x, mm_y], dim=1)
            
            # Scaling for improved optimization only
            mm_y = mm_y * 10
            mm_y_hat = mm_y_hat * 10

            loss, sigma = uncertainty_model.loss(x=latent, y=mm_y, y_hat=mm_y_hat, epoch=e)
            loss_epoch += loss.mean(dim=(1, 2)).sum().item()
            loss = loss.mean()

            loss.backward()

            optim.step()
            optim.zero_grad()

        loss_epoch /= num_samples
        scheduler.step(loss_epoch)
        logging.info('Loss epoch: {}\n'.format(loss_epoch))
        
        with open(os.path.join(conf.save_path, "loss_autoencoder.txt"), "a+") as f:
            print('Loss epoch: {}\n'.format(loss_epoch), file=f)

        # Save the model at every epoch
        if loss_epoch < loss_saved:
            logging.info('Saving the model at the end of epoch: {}'.format(e+1))
            autoencoder_dict['epoch'] = e + 1
            autoencoder_dict['loss'] = loss_epoch

            save_training_package(autoencoder_dict, conf.save_path, type='autoencoder')
            
            loss_saved = loss_epoch
            autoencoder_dict_best = deepcopy_training_package(autoencoder_dict, conf, type='autoencoder')
    
    logging.info('Training completed successfully.')
    logging.info('Models saved at: {}/autoencoder.pt'.format(conf.save_path))
    logging.info('Best model epoch: {}'.format(autoencoder_dict_best['epoch']))

    return autoencoder_dict_best


@torch.no_grad()
def visualize(autoencoder_dict, dataset, conf, split):
    """
    Visualize function for the autoencoder model.

    This function visualizes the predictions of the autoencoder model by comparing 
    ground truth motion sequences with predicted sequences. It also generates 
    skeleton visualizations and GIFs for qualitative analysis.

    Workflow:
    1. Iterates through the dataset to generate predictions.
    2. Visualizes skeleton sequences and saves them as images and GIFs.
    3. Plots uncertainty for the latent space.

    Parameters:
    -----------
    autoencoder_dict : dict
        Dictionary containing the autoencoder model and related components.
    dataset : MultimodalDataset
        Dataset object for visualization.
    conf : dict
        Configuration object containing experiment settings and paths.
    split : str
        Dataset split to visualize (e.g., 'train', 'test').

    Returns:
    --------
    None
    """
    logging.info('Visualizing autoencoder.')

    batch_size = conf.experiment_settings['batch_size']

    autoencoder_model = autoencoder_dict['model']
    uncertainty_model = autoencoder_dict['uncertainty']
    autoencoder_model.eval()
    uncertainty_model.eval()

    dataset.strided = False
    dataset.use_augmentation = False

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=conf.num_workers)
    
    visualization = Visualizer(save_path=conf.save_path, dataset=dataset)

    for x, mm_y, _, metadata in track(loader):
        
        i = metadata['idx']
        
        # Iterate over batch
        for x_, mm_y_, i_ in zip(x, mm_y, i):
            # Add a dimension for batch size
            x_ = x_.cuda().unsqueeze(0)
            mm_y_ = mm_y_.cuda().unsqueeze(0)

            mm_y_, _, _ = motion_transfer(
                skeletal_reference=x_, motion_reference=mm_y_, dataset=conf.dataset)

            mm_y_hat, _ = autoencoder_model(x_, mm_y_)
            # It predicts X and Y which is why we need to remove X
            mm_y_hat = mm_y_hat[:, x_.shape[1]:]

            skeleton_list = [
                x_.squeeze(0).cpu().numpy(),
                torch.cat([x_.squeeze(0).cpu(), mm_y_.squeeze(0).cpu()], dim=0).numpy(),
                torch.cat([x_.squeeze(0).cpu(), mm_y_hat.squeeze(0).cpu()], dim=0).numpy()]

            # If this throws an error saying directory exists, then go buy a lottery ticket
            random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
            
            visualization.visualize_skeleton_compare_multi(
                sequences=skeleton_list,
                string="autoencoder_viz/{}/{}/{}".format(split, i_, random_string),
                return_array=False)
            
            visualization.visualize_skeleton_compare_multi_gif(
                sequences=skeleton_list,
                string="forecast_gif/{}/{}/{}".format(split, i_, random_string))

        # Plot uncertainty
        _, latent = autoencoder_model(x.cuda(), mm_y.cuda())
        uncertainty_model.plot_uncertainty(latent, epoch=autoencoder_dict['epoch'])
        
        break

    save_training_package(autoencoder_dict, conf.save_path, type='autoencoder')
    
    logging.info('Visualization completed successfully for one batch.')
    logging.info('Models saved at: {}/autoencoder.pt'.format(conf.save_path))

    return None


@torch.no_grad()
def evaluate(autoencoder_dict, dataset, conf, split):
    """
    Evaluate function for the autoencoder model.

    This function evaluates the performance of the autoencoder model by computing 
    the reconstruction loss on the specified dataset split.

    Workflow:
    1. Iterates through the dataset to compute the reconstruction loss.
    2. Saves the evaluation results and the model.

    Parameters:
    -----------
    autoencoder_dict : dict
        Dictionary containing the autoencoder model and related components.
    dataset : MultimodalDataset
        Dataset object for evaluation.
    conf : dict
        Configuration object containing experiment settings and paths.
    split : str
        Dataset split to evaluate (e.g., 'train', 'test').

    Returns:
    --------
    None
    """
    logging.info('Evaluating autoencoder.')

    batch_size = conf.experiment_settings['batch_size']

    dataset.strided = False
    dataset.use_augmentation = False

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=conf.num_workers)

    loss = 0.
    num_samples = len(dataset)

    autoencoder_model = autoencoder_dict['model']
    autoencoder_model.eval()

    for x, mm_y, _, _ in track(loader):

        x = x.cuda()
        mm_y = mm_y.cuda()

        mm_y, _, _ = motion_transfer(
            skeletal_reference=x, motion_reference=mm_y, dataset=conf.dataset)

        mm_y_hat, _ = autoencoder_model(x, mm_y)
        # It predicts X and Y which is why we need to remove X
        mm_y_hat = mm_y_hat[:, x.shape[1]:]

        # loss is calculated on y_hat only and not all of train like x
        loss += torch.pow(mm_y_hat - mm_y, 2).mean(dim=[1, 2, 3, 4]).sum().item()
        
    print('Loss epoch ({}): {}\n'.format(split, loss / num_samples))
    with open(os.path.join(conf.save_path, "loss_autoencoder.txt"), "a+") as f:
        print('Loss epoch ({}): {}\n'.format(split, loss / num_samples), file=f)

    # Save the model
    save_training_package(autoencoder_dict, conf.save_path, type='autoencoder')

    
    logging.info('Evaluation completed successfully.')
    logging.info('Models saved at: {}/autoencoder.pt'.format(conf.save_path))

    return None


@torch.no_grad()
def projection(autoencoder_dict, dataset, conf, viz=False):
    """
    Projection function for the autoencoder model.

    This function generates latent space projections of the dataset using the trained 
    autoencoder model. It can also visualize a subset of the projections for qualitative analysis.

    Workflow:
    1. If `viz` is False:
       - Computes latent space projections for the entire dataset.
       - Returns the projection array and corresponding action labels.
    2. If `viz` is True:
       - Samples a subset of the dataset for visualization.
       - Generates skeleton visualizations and saves them as numpy arrays.

    Parameters:
    -----------
    autoencoder_dict : dict
        Dictionary containing the autoencoder model and related components.
    dataset : MultimodalDataset
        Dataset object for generating projections.
    conf : dict
        Configuration object containing experiment settings and paths.
    viz : bool, optional
        Whether to visualize a subset of the projections (default is False).

    Returns:
    --------
    tuple or None
        If `viz` is False: Returns a tuple containing the projection array and action labels.
        If `viz` is True: Returns None after saving visualizations and projections.
    """

    save_training_package(autoencoder_dict, conf.save_path, type='autoencoder')

    batch_size = conf.experiment_settings['batch_size'] * 10

    autoencoder_model = autoencoder_dict['model']
    autoencoder_model.eval()

    dataset.use_augmentation = False
    dataset.strided = False

    num_samples = len(dataset)

    if not viz:
        logging.info('Obtaining projection.')

        loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=conf.num_workers)
        
        projection_array = torch.empty(num_samples, conf.encoder["projection"])
        actions_array = np.empty((2, num_samples), dtype=object)

        for x, _, y, metadata in track(loader):
            
            # The action label for x and y are the same since they are from the same video
            actions_array[:, metadata['idx']] = metadata['x_action']
            
            x = x.cuda()
            y = y.cuda()

            projection, _ = autoencoder_model.get_context_y(y=y)
            projection_array[metadata['idx']] = projection.cpu()
            
        actions_array = actions_array.T
        projection_array = projection_array.numpy()

        return projection_array, actions_array
    
    else:
        # Sampling only 2000 samples from the entire dataloader to visualize
        sample_idx = np.random.choice(num_samples, size=(2000,), replace=False).tolist()
        subset = torch.utils.data.Subset(dataset=dataset, indices=sample_idx)

        loader = torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=False)

        projection_array = list()
        visualize_array = list()
        
        visualization = Visualizer(save_path=conf.save_path, dataset=dataset)

        # x, mm_y, y, idx
        for x, _, y, _ in track(loader):

            x = x.cuda()
            y = y.cuda()

            projection, y_hat = autoencoder_model.get_context_y(x=x, y=y)
            y_hat = y_hat[:, x.shape[1]:]

            for i in range(y.shape[0]):
                skeleton_list = [y[i].cpu().numpy(), y_hat[i].cpu().numpy()]

                vis = visualization.visualize_skeleton_compare_multi(
                    sequences=skeleton_list,
                    string=None, return_array=True)

                visualize_array.append(vis)

            projection_array.append(projection.cpu())

        projection_array = torch.cat(projection_array, dim=0).numpy()
        projection_array = autoencoder_dict['reduction'].transform(projection_array)
        visualize_array = np.stack(visualize_array)

        np.save(
            os.path.join(
                conf.save_path, 'images_{}.npy'.format(len(visualize_array))), arr=visualize_array)
        np.save(
            os.path.join(
                conf.save_path, 'z_{}.npy'.format(len(projection_array))), arr=projection_array)
        
        logging.info('Saved numpy arrays for visualization.')

        return None