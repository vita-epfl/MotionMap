import os
import logging
from tqdm import tqdm
from copy import deepcopy
from rich.progress import track

import torch
import numpy as np
from matplotlib import pyplot as plt

from visualizer import Visualizer

from utilities import find_local_maxima, motion_transfer, mm_gt_train
from utilities import save_training_package, deepcopy_training_package
from metrics import compute_diversity, compute_ade, compute_fde, compute_mmade
from metrics import compute_mmfde, compute_anp, AverageMeter


@torch.no_grad()
def _inference(x, motionmap_dict, autoencoder_model, r, motionmap_dataset, hm_gt, save_path, i, viz=True):
    """
    Perform inference for multimodal prediction.

    This function generates multimodal predictions for a given input by using the predicted MotionMap 
    and Autoencoder models. It also visualizes heatmaps and predictions if specified.

    Parameters:
    -----------
    x : torch.Tensor
        Input skeletal sequence of shape (T, 1, J, 3), where T is the number of frames, 
        J is the number of joints, and D is the dimensionality.
    motionmap_dict : dict
        Dictionary containing the MotionMap model and related components.
    autoencoder_model : torch.nn.Module
        Autoencoder model used for decoding multimodal predictions.
    r : int
        Radius for finding local maxima in the heatmap.
    motionmap_dataset : MotionmapDataset
        Dataset object for MotionMap-related preprocessing and scaling.
    hm_gt : numpy.ndarray
        Ground truth heatmap for visualization.
    save_path : str
        Path to save visualizations and results.
    i : int
        Index of the current sample for saving visualizations.
    viz : bool, optional
        If True, visualizes heatmaps and predictions. Default is True.

    Returns:
    --------
    y_mm : torch.Tensor
        Predicted multimodal skeletal sequences of shape (M, T, 1, J, 3), where M is the number of modes.
    hm : torch.Tensor
        Predicted heatmap of shape (H, W), where H and W are the heatmap dimensions.
    local_maxima : torch.Tensor
        Local maxima coordinates in the heatmap space.
    h_y_mm : torch.Tensor or None
        Latent uncertainty predictions for the multimodal outputs, if available.

    Notes:
    ------
    - The function uses the input to generate a heatmap, finds local maxima, and decodes 
      multimodal predictions for each mode.
    - If `viz` is True, it saves visualizations of the predicted and ground truth heatmaps.
    """
    
    # Get models
    motionmap_model = motionmap_dict['model']
    x_template = motionmap_dict['x_template']
    
    # Move input to GPU and preprocess
    x = x.cuda().unsqueeze(0)
    x, _, _ = motion_transfer(
        skeletal_reference=x_template.expand(x.shape[0], -1, -1, -1, -1),
        motion_reference=x, 
        dataset=motionmap_dataset.dataset
    )
    
    # Get heatmap prediction
    hm = motionmap_model(x)
    hm = torch.nn.Sigmoid()(hm)
    hm = hm.squeeze(0).cpu().numpy()
    
    # Find local maxima
    local_maxima = find_local_maxima(hm, r=r)[0]
    local_maxima_plot = deepcopy(local_maxima)
    
    # Convert back to tensor and process
    local_maxima = torch.from_numpy(local_maxima).float()
    local_maxima = motionmap_dataset.invert_scaling(local_maxima).cuda()
    
    # Get predictions for each mode
    x = x.expand(local_maxima.shape[0], -1, -1, -1, -1)
    y_mm, h_y_mm = autoencoder_model.decode(x, local_maxima)
    y_mm = y_mm.cpu()
    
    if viz:
        # Visualize heatmaps
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(hm)
        axs[1].imshow(hm_gt)
        
        # Plot local maxima as crosses on the heatmap
        for maxima in local_maxima_plot:
            axs[0].plot(maxima[1], maxima[0], 'rx')
            
        os.makedirs(os.path.join(save_path, 'multimodal_hm'), exist_ok=True)
        plt.savefig(os.path.join(save_path, 'multimodal_hm', f'{i}.png'), dpi=150)
        plt.savefig(os.path.join(save_path, 'multimodal_hm', f'{i}.svg'), dpi=150)
        plt.close()

    
    return y_mm, hm, local_maxima, h_y_mm


def train(autoencoder_dict, dataset, conf):
    """
    Finetune the multimodal model: decoder and embedding post-processing only (not MotionMap).

    This function fine-tunes the multimodal model using the provided dataset and configuration. 
    It trains the model over multiple epochs, computes the loss, updates the model parameters, 
    and saves the best-performing model.

    Parameters:
    -----------
    autoencoder_dict : dict
        Dictionary containing the Autoencoder model, optimizer, scheduler, and other training components.
    dataset : FineTuneDataset
        Dataset object for training the multimodal model.
    conf : object
        Configuration object containing experiment settings and paths.

    Returns:
    --------
    dict
        A dictionary containing the best Autoencoder model and its associated training information.

    Notes:
    ------
    - The function resets the loss and scheduler if training starts from an early epoch.
    - The model is saved at the end of each epoch if the loss improves.
    - Loss values for each epoch are logged and saved to a text file.
    """

    logging.info('Finetuning multimodal model.')
    
    dataset.strided = True
    dataset.use_augmentation = True # +- shift only, no rotation or mirroring
    
    end_epoch = conf.experiment_settings['epochs']
    batch_size = conf.experiment_settings['batch_size']

    start_epoch = autoencoder_dict['epoch']
    autoencoder_model = autoencoder_dict['model']
    optim = autoencoder_dict['optimizer']
    scheduler = autoencoder_dict['scheduler']
    loss_saved = autoencoder_dict['loss']
    uncertainty_model = autoencoder_dict['uncertainty']

    # Finetune will load an earlier model and continue training.
    # However, this model will be trained on average embeddings, and will not be as good as the earlier one.
    # Since the earlier model has lower loss, the newer model will not be saved. Hence overwrite earlier saved loss.
    if start_epoch <= (end_epoch // 2):
        print('Resetting the loss and scheduler.')
        loss_saved = np.inf
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, factor=0.1, patience=10, cooldown=0, min_lr=1e-7, verbose=True)

    autoencoder_dict_best = deepcopy_training_package(autoencoder_dict, conf, type='autoencoder')

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=conf.num_workers)

    num_samples = len(dataset)

    for e in range(start_epoch, end_epoch):
        logging.info('Epoch: {}'.format(e+1))

        loss_epoch = 0.

        autoencoder_model.train()
        uncertainty_model.train()

        for x, mm_y, embedding, _ in track(train_loader):

            x = x.cuda()
            mm_y = mm_y.cuda()
            embedding = embedding.cuda()

            mm_y, _, _ = motion_transfer(
                skeletal_reference=x, motion_reference=mm_y, dataset=conf.dataset)
            
            mm_y_hat, latent = autoencoder_model.decode(x, embedding)
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
        
        with open(os.path.join(conf.save_path, "autoencoder.txt"), "a+") as f:
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
def visualize(autoencoder_dict, motionmap_dict, multimodal_dataset, motionmap_dataset, conf, split):
    """
    Visualize multimodal predictions and ground truth.

    This function generates visualizations of multimodal predictions and ground truth 
    for the specified dataset split. It also computes and displays evaluation metrics.

    Parameters:
    -----------
    autoencoder_dict : dict
        Dictionary containing the Autoencoder model and related components.
    motionmap_dict : dict
        Dictionary containing the MotionMap model and related components.
    multimodal_dataset : MultimodalDataset
        Dataset object for multimodal predictions.
    motionmap_dataset : MotionmapDataset
        Dataset object for MotionMap-related preprocessing and scaling.
    conf : object
        Configuration object containing experiment settings and paths.
    split : str
        Dataset split to visualize (e.g., 'train', 'test').

    Returns:
    --------
    None

    Notes:
    ------
    - Visualizations include skeleton comparisons and heatmaps for predictions and ground truth.
    - Metrics such as ADE, FDE, MMADE, MMFDE, and Diversity are computed for each sample.
    - Results are saved as images and GIFs in the specified save path.
    """
    
    visualization = Visualizer(save_path=conf.save_path, dataset=multimodal_dataset)
    
    if conf.multimodal['mm_gt_train'] and split == 'test':
        test_train_mm_, aux_dataset_m = mm_gt_train(conf)

    logging.info('Multimodal forecast pipeline.')

    batch_size = conf.experiment_settings['batch_size']

    autoencoder_model = autoencoder_dict['model']
    motionmap_model = motionmap_dict['model']
    uncertainty_model = autoencoder_dict['uncertainty']
    
    multimodal_dataset.strided = False
    motionmap_dataset.strided = False
    
    multimodal_dataset.use_augmentation = False
    motionmap_dataset.use_augmentation = False

    loader = torch.utils.data.DataLoader(
        multimodal_dataset, batch_size=batch_size, shuffle=True, num_workers=conf.num_workers)

    autoencoder_model.eval()
    motionmap_model.eval()
    uncertainty_model.eval()

    for x_batch, _, y_batch, metadata_batch in track(loader):

        i_batch = metadata_batch['idx']
        actions = metadata_batch['x_action'][1] if conf.dataset == 'Human36M' else metadata_batch['x_action'][0]

        # Iterate over the batch
        for x, y, i, action in zip(x_batch, y_batch, i_batch, actions):
            
            i = i.item()
            
            # Get multimodal ground truth 
            _, _, extra = multimodal_dataset.loader.dataset.__getitem__(i, random_select=False)
            mm_y = torch.from_numpy(extra['mm_gt'])

            # Overwrite multimodal ground truth from training set
            if conf.multimodal['mm_gt_train'] and split == 'test':
                if len(test_train_mm_[i]) == 0: # if there is no train data for this test data
                    continue
                else:
                    index_aux = test_train_mm_[i]
                    all_train_for_test_mm = [
                        aux_dataset_m.loader.dataset.__getitem__(i_.item(), random_select=True)[1] for i_ in index_aux]
                    all_train_for_test_mm = np.array(all_train_for_test_mm)
                    mm_y = torch.from_numpy(all_train_for_test_mm)

            # Get the heatmap ground truth
            _, hm_gt = motionmap_dataset.__getitem__(i)
            
            mm_y, _, _ = motion_transfer(
                skeletal_reference=x.expand(mm_y.shape[0], -1, -1, -1, -1),
                motion_reference=mm_y,
                dataset=conf.dataset)
            mm_y = mm_y.numpy()
            
            # Create list of motions for ground truth visualization
            gt_list = list()
            gt_list.append(x.numpy())
            gt_list.append(np.concatenate([x.numpy(), y.numpy()], axis=0))
            gt_list.extend([np.concatenate([x.numpy(), t.squeeze(0)], axis=0) \
                            for t in np.split(mm_y, mm_y.shape[0], axis=0)])

            # Create list of motions for prediction visualization
            pred_list = list()
            pred_list.append(x.numpy())
            pred_list.append(np.concatenate([x.numpy(), y.numpy()], axis=0))

            y_mm, hm, local_maxima, h_mm_y = _inference(
                x, motionmap_dict, autoencoder_model,
                conf.multimodal['peak_interval'], motionmap_dataset, hm_gt, conf.save_path, i, viz=True)

            pred_list.extend([t.squeeze(0).numpy() for t in torch.split(y_mm, 1, dim=0)])
            
            # Metric calculation
            metric_y_mm = y_mm[:, x.shape[0]:].numpy()    
            metric_y_mm = metric_y_mm.reshape(metric_y_mm.shape[0], metric_y_mm.shape[1], -1)
            
            sample_ade = compute_ade(metric_y_mm, y.numpy().reshape(y.shape[0], -1))
            sample_fde = compute_fde(metric_y_mm, y.numpy().reshape(y.shape[0], -1))
            sample_mm_ade = compute_mmade(
                metric_y_mm, y.numpy().reshape(y.shape[0], -1), mm_y.reshape(mm_y.shape[0], mm_y.shape[1], -1))
            sample_mm_fde = compute_mmfde(
                metric_y_mm, y.numpy().reshape(y.shape[0], -1), mm_y.reshape(mm_y.shape[0], mm_y.shape[1], -1))
            sample_diversity = compute_diversity(metric_y_mm)
            sample_num_predictions = metric_y_mm.shape[0]

            title = "ADE: {:.2f} FDE: {:.2f} MMADE: {:.2f} MMFDE: {:.2f} Diversity: {:.2f} Num Pred: {}".format(
                sample_ade, sample_fde, sample_mm_ade, sample_mm_fde, sample_diversity, sample_num_predictions)
            

            visualization.visualize_skeleton_compare_multi(
                sequences=gt_list, string='{}/{}/gt'.format(split, i), return_array=False)
            visualization.visualize_skeleton_compare_multi(
                sequences=pred_list, string='{}/{}/pred'.format(split, i), return_array=False, title=title)
            visualization.visualize_skeleton_compare_multi_gif(
                sequences=gt_list, string='{}/{}/gt_gif'.format(split, i))
            visualization.visualize_skeleton_compare_multi_gif(
                sequences=pred_list, string='{}/{}/pred_gif'.format(split, i))
            
            uncertainty_model.plot_uncertainty(h_mm_y, i)
            
        # Just do it for one batch
        break

    logging.info('Visualization completed successfully.')

    return None


@torch.no_grad()
def evaluate(autoencoder_dict, motionmap_dict, multimodal_dataset, motionmap_dataset, conf, split):
    """
    Evaluate the multimodal forecasting model.

    This function evaluates the multimodal forecasting model on the specified dataset split 
    by computing evaluation metrics and logging the results.

    Parameters:
    -----------
    autoencoder_dict : dict
        Dictionary containing the Autoencoder model and related components.
    motionmap_dict : dict
        Dictionary containing the MotionMap model and related components.
    multimodal_dataset : MultimodalDataset
        Dataset object for multimodal predictions.
    motionmap_dataset : MotionmapDataset
        Dataset object for MotionMap-related preprocessing and scaling.
    conf : object
        Configuration object containing experiment settings and paths.
    split : str
        Dataset split to evaluate (e.g., 'train', 'test').

    Returns:
    --------
    None

    Notes:
    ------
    - Metrics such as Diversity, ADE, FDE, MMADE, MMFDE, and ANP are computed for each sample.
    - Results are logged and saved to a text file in the specified save path.
    - If `mm_gt_train` is enabled and the split is 'test', multimodal ground truth from the training set is used.
    """
    
    if conf.multimodal['mm_gt_train'] and split == 'test':
        test_train_mm_, aux_dataset_m = mm_gt_train(conf)

    logging.info('Evaluating multimodal forecasting model.')
    
    batch_size = conf.experiment_settings['batch_size']
    
    autoencoder_model = autoencoder_dict['model']
    motionmap_model = motionmap_dict['model']

    multimodal_dataset.strided = False
    motionmap_dataset.strided = False
    multimodal_dataset.use_augmentation = False
    motionmap_dataset.use_augmentation = False

    loader = torch.utils.data.DataLoader(
        multimodal_dataset, batch_size=batch_size, shuffle=False, num_workers=conf.num_workers)

    autoencoder_model.eval()
    motionmap_model.eval()

    num_samples = len(multimodal_dataset)
    
    stats_func = {'Diversity': compute_diversity, 'ADE': compute_ade, 'FDE': compute_fde,
                  'MMADE': compute_mmade, 'MMFDE': compute_mmfde, 'ANP': compute_anp}
    stats_names = list(stats_func.keys())
    stats_meter = {x: AverageMeter() for x in stats_names}
       
    it = tqdm(loader)
    for it_n, (x_batch, _, y_batch, metadata_batch) in enumerate(it):

        i_batch = metadata_batch["idx"]
        
        for x, y, i in zip(x_batch, y_batch, i_batch):
        
            i = i.item()

            # Get multimodal ground truth
            _, _, extra = multimodal_dataset.loader.dataset.__getitem__(i, random_select=False)
            mm_y_gt = torch.from_numpy(extra['mm_gt'])
            
            if conf.multimodal['mm_gt_train'] and split == 'test':
                index_aux = test_train_mm_[i]
    
                if len(index_aux) != 0:
                    # We have multimodal ground truth from training set for the unseen test samples
                    all_train_for_test_mm = [
                        aux_dataset_m.loader.dataset.__getitem__(i_.item(), random_select=True)[1] for i_ in index_aux]
                    all_train_for_test_mm = np.array(all_train_for_test_mm)
                    mm_y_gt = torch.from_numpy(all_train_for_test_mm)
                    
                else:
                    continue

            # Get heatmap ground truth
            _, hm_gt = motionmap_dataset.__getitem__(i)
            
            mm_y_gt, _, _ = motion_transfer(
                skeletal_reference=x.expand(mm_y_gt.shape[0], -1, -1, -1, -1).cuda(),
                motion_reference=mm_y_gt.cuda(),
                dataset=conf.dataset)
            mm_y_gt = mm_y_gt.cpu().numpy()

            mm_y_pred, hm, local_maxima, _ = _inference(
                x, motionmap_dict, autoencoder_model,
                conf.multimodal['peak_interval'], motionmap_dataset,
                hm_gt, conf.save_path, i, viz=False)
            mm_y_pred = mm_y_pred.numpy()

            # Metric calculation
            # Since we predict X and Y, remove the X part from the prediction
            mm_y_pred = mm_y_pred[:, x.shape[0]:]

            #adding the zero velocity to y_mm_pred
            y_zero_vel = x[-1].expand(y.shape[0], -1, -1, -1).numpy()
            mm_y_pred = np.concatenate((y_zero_vel[None, ...], mm_y_pred), axis=0)

            # For evaluation
            y_gt = y.numpy().reshape(y.shape[0], -1)
            mm_y_gt = mm_y_gt.reshape(mm_y_gt.shape[0], mm_y_gt.shape[1], -1)
            mm_y_pred = mm_y_pred.reshape(mm_y_pred.shape[0], mm_y_pred.shape[1], -1)

            for stats in stats_names:
                val = 0

                # CHANGE everything to 48 (flatten last two dims: 16, 3) and squeeze (remove 1 in 100, 1, 16, 3)
                val = stats_func[stats](mm_y_pred, y_gt, mm_y_gt)
                
                # val /= len(pred) #in our case, we have only one prediction
                if val is not None:
                    stats_meter[stats].update(val)

        it.set_postfix(
            ordered_dict={
                "Diversity": f"{num_samples:04d} : {stats_meter['Diversity'].val:.4f}({stats_meter['Diversity'].avg:.4f})",
                "ADE": f"{num_samples:04d} : {stats_meter['ADE'].val:.4f}({stats_meter['ADE'].avg:.4f})",
                "FDE": f"{num_samples:04d} : {stats_meter['FDE'].val:.4f}({stats_meter['FDE'].avg:.4f})",
                "MMADE": f"{num_samples:04d} : {stats_meter['MMADE'].val:.4f}({stats_meter['MMADE'].avg:.4f})",
                "MMFDE": f"{num_samples:04d} : {stats_meter['MMFDE'].val:.4f}({stats_meter['MMFDE'].avg:.4f})",
                "ANP": f"{num_samples:04d} : {stats_meter['ANP'].val:.4f}({stats_meter['ANP'].avg:.4f}),",
                "batch_no": it_n#batch_no
            },
            refresh=False,
        )
        
        if it_n %20 == 0:
            print('-' * 80)
            for stats in stats_names:
                str_stats = f'{num_samples:04d} {stats}: {stats_meter[stats].val:.4f}({stats_meter[stats].avg:.4f})'
                print(str_stats)

    
    print('=' * 80)
    with open(os.path.join(conf.save_path, "multimodal.txt"), "a+") as f:
        print(split, file=f)
        print('=' * 80, file=f)
    
    for stats in stats_names:
        str_stats = f'Total {stats}: {stats_meter[stats].avg:.4f}'
        print(str_stats)

        with open(os.path.join(conf.save_path, "multimodal.txt"), "a+") as f:
            print(str_stats, file=f)   
    
    print('=' * 80)
    with open(os.path.join(conf.save_path, "multimodal.txt"), "a+") as f:
        print('=' * 80, file=f)
        print('\n', file=f)

    logging.info('Evaluation completed successfully.')
