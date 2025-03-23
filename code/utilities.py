import os
import sys
import fcntl
import logging
from collections import deque

import numpy as np
from sklearn.decomposition import PCA
from skimage.feature import peak_local_max
from scipy.spatial.transform import Rotation as R

import torch

from openTSNE import TSNE

from model.decoder import ResidualBehaviorNet
from model.heatmap import HeatmapDecoder
from model.uncertainty import UncertaintyNetwork

import cProfile
import pstats


def profile_getitem(dataset, index):
    """
    Profiles the execution time of the dataset's __getitem__ method for a given index.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    dataset.__getitem__(index)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()


def get_progress_bar():
    """
    Creates and returns a custom progress bar using the rich library.
    Thanks to: https://timothygebhard.de/posts/richer-progress-bars-for-rich/
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    # Define custom progress bar
    progress_bar = Progress(
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    return progress_bar


def load_mmgt(i, num_frames, dataset, dataset_name, threshold, mmgt_stride, all_mmgt_x):
    """
    Used with multiprocess to parallelize the computation of mm_gt
    :param i: Index of the dataset (parallelized)
    :param num_frames: Number of frames to consider
    :param dataset: Dataset object
    :param dataset_name: Name of the dataset
    :param threshold: Threshold to consider a mm_gt as a match
    :param mmgt_stride: Stride of the mm_gt
    :param all_mmgt_x: All the mm_gt to consider

    :return: Index of the mm_gt that are close to i
    """
    # Move x_i to cuda
    x_i = dataset(i)[0][-num_frames:, :, 1:, :]
    x_i = torch.from_numpy(x_i).cuda()

    # Expand x_i to match all_mmgt_x
    x_i_expanded = x_i.expand(len(all_mmgt_x), -1, -1, -1, -1)

    # Scale all mmgt to match i's skeleton size
    scaled_all_mmgt_x, _, _ = motion_transfer(
        skeletal_reference=x_i_expanded, motion_reference=all_mmgt_x, dataset=dataset_name)
        
    # To compute the distance
    scaled_all_mmgt_x = scaled_all_mmgt_x.reshape(len(all_mmgt_x), -1)
    x_i = x_i.reshape(1, -1)
                
    distance = torch.cdist(x_i, scaled_all_mmgt_x).squeeze()
    ind = torch.nonzero(distance < threshold).squeeze(1)
    saved_mm_gt = ind.cpu() *  mmgt_stride

    if i not in saved_mm_gt:
        saved_mm_gt = torch.cat((saved_mm_gt, torch.tensor([i])))

    return saved_mm_gt


def get_positive_definite_matrix(tensor, dim):
    """
    Multiplies matrix with its transpose to get positive semidefinite matrix
    """
    tensor = tensor[:, :dim ** 2]
    tensor = tensor.view(-1, dim, dim)
    return torch.matmul(tensor, tensor.mT)


def compare_models(model1, model2):
    """
    Compares two models to check if their state dictionaries are identical.
    """
    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()

    for key in model1_state_dict:
        if not torch.all(torch.eq(model1_state_dict[key], model2_state_dict[key])):
            return False

    return True


@torch.no_grad()
def find_local_maxima(matrix, r=4):
    """
    Finds local maxima in a matrix and their magnitudes.
    """
    # Find the coordinates of local maxima
    coordinates = peak_local_max(
        matrix, min_distance=r, threshold_rel=0.35, num_peaks=49, exclude_border=False)

    magnitudes = matrix[tuple(coordinates.T)]

    return coordinates, magnitudes


def select_points(points, r):
    """
    Selects points ensuring a minimum distance between them.
    """
    idx = np.arange(len(points))

    np.random.shuffle(idx)
    selected_idx = [idx[0]]

    for i in idx[1:]:
        if all(np.linalg.norm(points[i] - points[j]) >= r for j in selected_idx):
            selected_idx.append(i)

    return selected_idx


def create_training_package(conf, type=None):
    """
    Creates and initializes a training package based on the configuration and type.
    """
    model_dict = {'epoch': 0, 'loss': torch.inf}

    lr = conf.experiment_settings['lr']

    if type == 'autoencoder':
        # This is the same as BeLFusion, hence called ResidualBehaviorNet
        model = ResidualBehaviorNet(**conf.architecture['pose']).cuda()
        optim = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=0., amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, factor=0.1, patience=10, cooldown=0, min_lr=1e-7, verbose=True)
        reduction = None

        autoencoder_dict = {
            'model': model,
            'optimizer': optim,
            'scheduler': scheduler,
            'reduction': reduction,
        }
        model_dict.update(autoencoder_dict)

        uncertainty = UncertaintyNetwork(arch=conf.architecture['pose'], save_path=conf.save_path).cuda()
        
        optim = torch.optim.Adam(
            list(model.parameters()) + list(uncertainty.parameters()),
            lr=lr, weight_decay=0., amsgrad=True)
        
        uncertainty_dict = {
            'uncertainty': uncertainty,
            'optimizer': optim,
        }
        model_dict.update(uncertainty_dict)
        
    else:
        assert type == 'motionmap'
        model = HeatmapDecoder(conf.encoder['dimensions'], conf.motionmap['size'],
                               conf.architecture['pose']['encoder_arch'], context_length=3).cuda()
        optim = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=0., amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, factor=0.1, patience=10, cooldown=0, min_lr=1e-7, verbose=True)
        
        motionmap_dict = {
            'model': model,
            'optimizer': optim,
            'scheduler': scheduler,
        }
        model_dict.update(motionmap_dict)
    
    return model_dict


def load_training_package(model_dict, load_path, type=None):
    """
    Loads a training package from a checkpoint file.
    """
    if type == 'autoencoder': extension = 'autoencoder.pt'
    else: extension = 'motionmap.pt'

    checkpoint = torch.load(os.path.join(load_path, extension)) 

    print('Model ({}) was saved at epoch: {}'.format(type, checkpoint['epoch']))

    model_dict['epoch'] = checkpoint['epoch']
    model_dict['model'].load_state_dict(checkpoint['model_state_dict'])
    model_dict['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
    model_dict['scheduler'].load_state_dict(checkpoint['scheduler_state_dict'])
    model_dict['loss'] = checkpoint['loss']

    if type == 'autoencoder':
        model_dict['reduction'] = checkpoint['reduction']
        model_dict['uncertainty'].load_state_dict(checkpoint['uncertainty_state_dict'])
    
    return model_dict


def save_training_package(model_dict, save_path, type=None):
    """
    Saves the training package to a checkpoint file.
    """
    if type == 'autoencoder': extension = 'autoencoder.pt'
    else: extension = 'motionmap.pt'

    model_state_dict = {
        'epoch': model_dict['epoch'],
        'model_state_dict': model_dict['model'].state_dict(),
        'optimizer_state_dict': model_dict['optimizer'].state_dict(),
        'scheduler_state_dict': model_dict['scheduler'].state_dict(),
        'loss': model_dict['loss']
    }

    if type == 'autoencoder':
        model_state_dict['reduction'] = model_dict['reduction']
        model_state_dict['uncertainty_state_dict'] = model_dict['uncertainty'].state_dict()

    torch.save(model_state_dict, os.path.join(save_path, extension), pickle_protocol=4)


def deepcopy_training_package(model_dict, conf, type=None):
    """
    Creates a deep copy of a training package.
    """
    init = create_training_package(conf, type=type)
    
    init['epoch'] = model_dict['epoch']
    init['model'].load_state_dict(model_dict['model'].state_dict())
    init['optimizer'].load_state_dict(model_dict['optimizer'].state_dict())
    init['scheduler'].load_state_dict(model_dict['scheduler'].state_dict())
    init['loss'] = model_dict['loss']

    if type == 'autoencoder':
        init['reduction'] = model_dict['reduction']
        init['uncertainty'].load_state_dict(model_dict['uncertainty'].state_dict())

    return init


def _fit_pca(x, dimensions):
    """
    Fits PCA on the given data and returns the PCA object.
    """
    pca = PCA(n_components=dimensions)
    return pca.fit(x)


def _fit_tsne(x, dimensions):
    """
    Fits t-SNE on the given data and returns the t-SNE object.
    """
    tsne = TSNE(perplexity=100, initialization='pca', metric='cosine', n_jobs=-1, random_state=0).fit(x)
    tsne.affinities.set_perplexities([20])
    tsne.optimize(250)
    return tsne


def fit_reduction(x, dimensions, algorithm=None):
    """
    Fits a dimensionality reduction algorithm (PCA or t-SNE) on the data.
    """
    if algorithm == 'pca':
        return _fit_pca(x, dimensions=dimensions)
    else:
        assert algorithm == 'tsne'
        return _fit_tsne(x, dimensions=dimensions)


def find_nearest_non_empty(codebook, start):
    """
    Finds the nearest non-empty location in a codebook using BFS.
    """
    # Define the possible moves
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Create a queue for BFS
    queue = deque([start])

    # Create a set to store visited locations
    visited = set()

    while queue:
        location = queue.popleft()

        # If this location has not been visited before
        if location not in visited:
            # Mark the location as visited
            visited.add(location)

            # If the list at this location is not empty, return the location
            if len(codebook[location]):
                return location

            # Add the neighboring locations to the queue
            for move in moves:
                neighbor = (location[0] + move[0], location[1] + move[1])
                if neighbor in codebook:
                    queue.append(neighbor)

    # If no non-empty location is found, return None
    raise Exception


@torch.no_grad()
def normalize_skeleton(tensor, thigh_length=None):
    """
    Normalizes a skeleton tensor by dividing it by the thigh length.
    """
    assert tensor.dim() == 5
    # [BS,   25 or 30 or 100 or 120,    1,   16 or 21,     3]

    if thigh_length is None:
        
        # Human36M
        if tensor.shape[3] == 16:
            hip = tensor[:, -1, 0, 0]
            knee = tensor[:, -1, 0, 1] 
        
        # AMASS
        else:
            assert tensor.shape[3] == 21
            hip = tensor[:, -1, 0, 1]
            knee = tensor[:, -1, 0, 4]

        # Size of thigh length is batch size
        thigh_length = torch.norm(hip - knee, dim=-1)

    return tensor / thigh_length[:, None, None, None, None], thigh_length


@torch.no_grad()
def unnormalize_skeleton(tensor, thigh_length):
    """
    Unnormalizes a skeleton tensor by multiplying it with the thigh length.
    """
    assert tensor.dim() == 5
    assert thigh_length.dim() == 1

    return tensor * thigh_length[:, None, None, None, None]


@torch.no_grad()
def mirror_pose_sequence(x=None, y=None, mm_gt=None, prob=0.5):
    """
    Mirrors a pose sequence along specified axes with a given probability.
    """
    # Taken from h36m.py
    mirror_axis = [1, 1, 1]
    for m in [0, 1]: # 2 is not used because the person would be upside down
        
        # Invert each axis (x, y) separately
        if np.random.rand() < prob:
            mirror_axis[m] = -1
    
    mirror_axis = np.array(mirror_axis)

    if x is not None: x = x * mirror_axis
    if y is not None: y = y * mirror_axis
    if mm_gt is not None: mm_gt = mm_gt * mirror_axis

    return x, y, mm_gt


@torch.no_grad()
def rotate_pose_sequence(x=None, y=None, mm_gt=None, prob=0.5):
    """
    Rotates a pose sequence randomly around the z-axis with a given probability.
    """
    # Taken from h36m.py
    # apply random rotations with probability 1
    rotation_axes = ['z'] # 'x' and 'y' not used because the person could be upside down
    for a in rotation_axes:
        if np.random.rand() < prob:
            degrees = np.random.randint(0, 360)
            r = R.from_euler(a, degrees, degrees=True).as_matrix().astype(np.float32)
            if x is not None: x = (r @ x.reshape((-1, 3)).T).T.reshape(x.shape)
            if y is not None: y = (r @ y.reshape((-1, 3)).T).T.reshape(y.shape)
            if mm_gt is not None: mm_gt = (r @ mm_gt.reshape((-1, 3)).T).T.reshape(mm_gt.shape)
    
    return x, y, mm_gt


def interrupt_handler(save_path):
    """
    Creates a signal handler to save the interrupt state to a file.
    """
    def interrupt_sigterm(signum, frame):
        
        print("Received termination, adding '{}' to interrupt.txt".format(save_path))
        
        # Open the file in append mode and acquire an exclusive lock
        with open("interrupt.txt", "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)

            # Move the file pointer to the beginning of the file
            f.seek(0)
            
            # Read the existing paths
            locations = f.read().splitlines()
            
            if save_path not in locations:
                f.write(save_path + "\n")

            # Release the lock
            fcntl.flock(f, fcntl.LOCK_UN)
        
        sys.exit(1)
    
    return interrupt_sigterm


@torch.no_grad()
def cartesian_to_spherical(pose, parent_child_links):
    """
    pose is of shape (B, J, 3) where B is the batch size,
    J is the number of joints and 3 is the x, y, z coordinates

    Becareful that the calculation for parents should be done before the child
    """
    
    batch_size = pose.shape[0]
    num_links = len(parent_child_links)
    
    rho = torch.zeros((batch_size, num_links), device=pose.device)
    theta = torch.zeros((batch_size, num_links), device=pose.device)
    phi = torch.zeros((batch_size, num_links), device=pose.device)
    
    # Looping through each link
    for i, link in enumerate(parent_child_links):
        # -1 corresponds to the pelvis which is removed from __getitem__
        parent = pose[:, link[0], : ] if link[0] != -1 else torch.zeros((batch_size, 3), device=pose.device)
        child = pose[:, link[1], : ]

        vector = child - parent
        
        rho[:, i] = torch.linalg.norm(vector, dim=-1)
        theta[:, i] = torch.arccos(vector[:,2] / rho[:, i])
        phi[:, i] = torch.arctan2(vector[:, 1], vector[:, 0])
    
    return rho, theta, phi


@torch.no_grad()
def spherical_to_cartesian(rho, theta, phi, parent_child_links, num_joints):
    """
    #joints: (B*T, 16, 3)
    Converts spherical coordinates back to Cartesian coordinates for a pose.
    """
    BT = rho.shape[0]  # batch size * number of frames
    J = num_joints

    joints = torch.zeros((BT, J, 3), device = rho.device)
    
    for i, link in enumerate(parent_child_links):
        
        parent = link[0]
        child = link[1]
        
        x = rho[:, i] * torch.sin(theta[:, i]) * torch.cos(phi[:, i])
        y = rho[:, i] * torch.sin(theta[:, i]) * torch.sin(phi[:, i])
        z = rho[:, i] * torch.cos(theta[:, i])
        
        parent_joint = joints[:, parent] if parent != -1 else torch.zeros((BT, 3), device = rho.device)
                    
        joints[:, child] = parent_joint + torch.stack([x,y,z], dim=1)
    
    return joints


@torch.no_grad()
def motion_transfer(skeletal_reference, motion_reference, dataset): #(B,25,1,16,3) #(B,100,1,16,3)
    """
    Transfers motion from a reference sequence to a skeletal reference.
    """
    
    # Define the parent-child relation
    if dataset == 'Human36M':
        num_joints = 16
        parent_child_links = [
            [-1,0], [0,1], [1,2], [-1, 3], [3,4], [4,5], [-1, 6],
            [6,7], [7,8], [8,9], [7,10], [10,11], [11,12], [7,13], [13,14], [14,15]]

    else:
        assert dataset == 'AMASS'
        num_joints = 21
        parent_child_links = [
            [-1,1], [1,4], [4,7], [7,10], [-1,0], [0,3], [3,6], [6,9], [-1,2], [2,5],
            [5,8], [8,11], [11,14], [8,13], [13,16], [16,18], [18,20], [8,12], [12,15], [15,17], [17,19]]
    
    # Get the length of various links from the skeletal reference
    pose = skeletal_reference[:, -1, 0, :, :] # Taking the last frame of the sequence as reference, final shape: (B, 16, 3)
    rho_skeletal, _, _ = cartesian_to_spherical(pose, parent_child_links) # rho_skeletal: (B, number of links)

    # Get the angles of the motion reference
    motion_shape = motion_reference.shape
    rho_motion, theta_motion, phi_motion = cartesian_to_spherical(motion_reference.view(-1, motion_shape[3], motion_shape[4]), parent_child_links)
    
    # Repeat the skeletal reference for the number of frames in the sequence
    rho_skeletal = rho_skeletal.repeat_interleave(motion_shape[1], dim=0) #repeating for the number of frames in the sequence
    
    final_sequence = spherical_to_cartesian(rho_skeletal, theta_motion, phi_motion, parent_child_links, num_joints) 
    final_sequence = final_sequence.view(motion_shape) #reshaping back to (B, T, 1, 16, 3)
    
    return final_sequence, rho_skeletal[:, 1], rho_motion[:, 1] # Returning the final sequence and the thigh length (for plot title purposes)


@torch.no_grad()
def mm_gt_train(conf):
    """
    Loads multimodal ground truth (mm_gt) from the training set for test data.
    """
    # Load the multimodal ground truth from the training set for the test data
    from dataset import MultimodalDataset
    aux_dataset_m = MultimodalDataset(conf)
    
    print("Preprocessing aux_dataset_m...")
    aux_dataset_m.preprocess("train", conf.num_frames)
    
    #set shuffle and strided to False
    aux_dataset_m.strided = False
    aux_dataset_m.use_augmentation = False
    
    #load the mm_gt from the train set for the test data
    # Get the cache path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mmgt_cache_dir = os.path.join(parent_dir, "mmgt_cache")
    mmgt_file = os.path.join(mmgt_cache_dir, "train_MMGT_for_Test_{}.pt".format(conf.dataset))
    test_train_mm_ = torch.load(mmgt_file)
    return test_train_mm_, aux_dataset_m

