import os
import fcntl
import logging
import random
from itertools import product
from pathlib import Path


import torch
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
import torch.nn.functional as F
from rich.progress import track

from utilities import find_nearest_non_empty
from utilities import rotate_pose_sequence, mirror_pose_sequence
from dataloader import h36mdataloader, amassdataloader


class MotionMapDataset(torch.utils.data.Dataset):
    """
    A dataset class for generating motion maps.

    Attributes:
        dataset (str): Name of the dataset.
        z_reduced (dict): Reduced embeddings for train and test splits.
        actions_train (torch.Tensor): Action labels for training data.
        actions_test (torch.Tensor): Action labels for testing data.
        hm_size (list): Size of the heatmap.
        save_path (str): Path to save generated plots and heatmaps.
        stride (int): Stride for sampling data.
        mmgt_stride (int): Stride for multimodal ground truth.
        augmentation (int): Augmentation range for data.
        use_codebook (bool): Whether to use a codebook. (Which is true since projections >> 2)
        codebook (dict): Codebook mapping heatmap locations to embeddings.
    """

    def __init__(self, z_reduced, actions, conf, strided=True):
        """
        Initialize the MotionmapDataset object.

        Args:
            z_reduced (dict): Reduced embeddings for train and test splits.
            actions (list): List containing training and testing action labels.
            conf (object): Configuration object with dataset settings.
            strided (bool): Whether to use strided sampling.
        """

        print('Initializing MotionmapDataset object.')
        self.dataset = conf.dataset
        self.z_reduced = z_reduced
        self.hm_size = conf.motionmap['size'] # Currently integer, later is a list (for rectangular heatmaps)
        self.save_path = conf.save_path
            
        self.actions_train = actions[0]
        self.actions_test = actions[1]

        # Scaling the samples to size of heatmap
        self.origin_shift = None
        self.scale_matrix = None
        self.pad_heatmap = None

        self.current_split = None
        self.use_augmentation = None

        self.stride = conf.dataset_settings[self.dataset]['stride']
        self.mmgt_stride = conf.dataset_settings[self.dataset]['mmgt_stride']
        self.augmentation = conf.dataset_settings[self.dataset]['augmentation']

        self.set_scaling_matrix()
        self.pad_heatmap_parameters()

        for s in ['train', 'test']:
            z_hm = self.scale_embeddings(z_reduced[s][1])
            self.plot_hm_space_action(split=s, z_hm=z_hm[::self.mmgt_stride])
            self.get_heatmap_density(split=s, z_hm=z_hm)

        self.use_codebook = False
        if conf.encoder['projection'] not in [2, 3]:
            self.use_codebook = True
            self.codebook = self.get_codebook()

        self.strided = strided
            

    @torch.no_grad()
    def set_scaling_matrix(self):
        """
        Compute the scaling matrix to map embeddings to heatmap space.
        """
        print('Computing scaling.')
        # Select training embeddings
        mean = self.z_reduced['train'][1]

        assert mean.dim() == 2

        device = mean.device

        extent = torch.norm(torch.max(mean, dim=0)[0] - torch.min(mean, dim=0)[0])

        top_left = torch.min(mean, dim=0)[0]
        self.origin_shift = top_left - (0.02 * extent) # Keep 2% offset

        shifted_embeddings = mean - self.origin_shift

        bottom_right = torch.max(shifted_embeddings, dim=0)[0]
        bottom_right += (0.02 * extent) # Keep 2% offset

        self.hm_size = torch.tensor(
            [bottom_right[0] * self.hm_size / max(bottom_right[0], bottom_right[1]),
             bottom_right[1] * self.hm_size / max(bottom_right[0], bottom_right[1])]).to(device)

        self.hm_size = torch.ceil(self.hm_size).int()

        self.scale_matrix = torch.diag(self.hm_size / bottom_right)
        self.hm_size = self.hm_size.tolist()

        self.origin_shift = self.origin_shift.cpu()
        self.scale_matrix = self.scale_matrix.cpu()

        print('Scaled embeddings to fit heatmap of size: ', self.hm_size)


    @torch.no_grad()
    def scale_embeddings(self, z):
        """
        Map from 2D learnt representations to 2D Heatmap (without padding)
        """
        assert (self.origin_shift is not None) and (self.scale_matrix is not None), "Need to run set_scaling_matrix"
        assert z.dim() == 2

        # self.scale_matrix: 2 times 2
        scale_matrix = self.scale_matrix.to(z.device)
        origin_shift = self.origin_shift.to(z.device)

        z_hm = scale_matrix.expand(z.shape[0], -1, -1) @ (z - origin_shift).unsqueeze(2)

        return z_hm.squeeze(-1)


    @torch.no_grad()
    def pad_heatmap_parameters(self):
        """
        Compute padding parameters to make the heatmap square.
        """
        # ChatGPT help :)

        max_dim = max(self.hm_size[0], self.hm_size[1])
        # Calculate padding to be added to each side
        vertical_padding = (max_dim - self.hm_size[0]) // 2
        horizontal_padding = (max_dim - self.hm_size[1]) // 2
    
        # If the difference is odd, add the extra padding to the bottom or right
        top_padding = vertical_padding
        bottom_padding = vertical_padding + (max_dim - self.hm_size[0]) % 2
        left_padding = horizontal_padding
        right_padding = horizontal_padding + (max_dim - self.hm_size[1]) % 2

        self.pad_heatmap = (left_padding, right_padding, top_padding, bottom_padding)


    @torch.no_grad()
    def plot_hm_space(self, split, z_hm):
        """
        Plot the heatmap space as a scatter plot.

        Args:
            split (str): Dataset split ('train' or 'test').
            z_hm (torch.Tensor): Scaled embeddings for the split.
        """
        print('Plotting heatmap space for split: {}\tLen: {}'.format(split, len(z_hm)))
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.scatter(z_hm[:, 1].cpu().numpy(), z_hm[:, 0].cpu().numpy(), s=0.25)
        plt.savefig(os.path.join(self.save_path, 'scatter_Y_{}.png'.format(split)), dpi=200)
        plt.close()
        
    # Color coding actions for the hm_plot
    @torch.no_grad()
    def plot_hm_space_action(self, split, z_hm):
        """
        Plot heatmap space with color-coded actions.

        Args:
            split (str): Dataset split ('train' or 'test').
            z_hm (torch.Tensor): Scaled embeddings for the split.
        """
        # Extract actions from the second column
        action_indx = 1 if self.dataset == 'Human36M' else 0
        if split == 'train':
            actions = self.actions_train[:, action_indx]
        elif split == 'test':
            actions = self.actions_test[:, action_indx]  # Convert to numpy if it's a tensor

        # Define discrete colors (e.g., from TABLEAU_COLORS or a custom list)
        color_map = plt.get_cmap('tab20')
        unique_actions = np.unique(actions)

        if len(unique_actions) > color_map.N:
            raise ValueError('Not enough unique colors available for all actions.')
        
        # Map each unique action to a color
        action_to_color = {action: color_map(i / len(unique_actions)) for i, action in enumerate(unique_actions)}

        # Map each action in `actions` to a color
        action_colors = np.array([action_to_color[action] for action in actions])

        # Plot heatmap space with color-coded actions
        print('Plotting heatmap space for split: {}\tLen: {}'.format(split, len(z_hm)))
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.scatter(z_hm[:, 1].cpu().numpy(), z_hm[:, 0].cpu().numpy(), s=0.25, c=action_colors[::self.mmgt_stride], alpha=0.5)
        
        # Create legend
        legend_patches = [mpatches.Patch(color=action_to_color[action], label=f'{action}') for action in unique_actions]
        plt.legend(handles=legend_patches, title="Actions", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save the plot
        plt.savefig(os.path.join(self.save_path, 'scatter_Y_color_{}.pdf'.format(split)), dpi=200)
        plt.savefig(os.path.join(self.save_path, 'scatter_Y_color_{}.svg'.format(split)), dpi=200)
        plt.close()
        
    # Color coding actions for the hm_plot:
    @torch.no_grad()
    def plot_hm_space_subject(self, split, z_hm):
        """
        Plot heatmap space with color-coded subjects.

        Args:
            split (str): Dataset split ('train' or 'test').
            z_hm (torch.Tensor): Scaled embeddings for the split.
        """
        # Extract actions from the second column
        action_indx = 0 
        if split == 'train':
            actions = self.actions_train[:, action_indx]
        elif split == 'test':
            return 
        # Define discrete colors (e.g., from TABLEAU_COLORS or a custom list)
        color_list = list(mcolors.TABLEAU_COLORS.values())
        unique_actions = np.unique(actions)
        
        # Map each unique action to a color
        action_to_color = {action: color_list[i % len(color_list)] for i, action in enumerate(unique_actions)}

        # Map each action in `actions` to a color
        action_colors = np.array([action_to_color[action] for action in actions])

        # Plot heatmap space with color-coded actions
        print('Plotting heatmap space for split: {}\tLen: {}'.format(split, len(z_hm)))
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        scatter = plt.scatter(z_hm[:, 1].cpu().numpy(), z_hm[:, 0].cpu().numpy(), s=0.25, c=action_colors)
        
        # Create legend
        legend_patches = [mpatches.Patch(color=action_to_color[action], label=f'{action}') for action in unique_actions]
        plt.legend(handles=legend_patches, title="Subjects", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save the plot
        plt.savefig(os.path.join(self.save_path, 'scatter_Y_color_subject_{}.png'.format(split)), dpi=200)
        plt.close()
        

    @torch.no_grad()
    def get_heatmap_density(self, split, z_hm):
        """
        Compute and plot the density of embeddings in heatmap space.

        Args:
            split (str): Dataset split ('train' or 'test').
            z_hm (torch.Tensor): Scaled embeddings for the split.
        """
        kde = gaussian_kde(z_hm.cpu().numpy().T)
        u, v = np.mgrid[0: self.hm_size[0], 0: self.hm_size[1]]

        positions = np.vstack([u.ravel(), v.ravel()])
        density = np.reshape(kde(positions).T, u.shape)
        
        # Plot the density
        plt.imshow(density, origin='upper')
        plt.colorbar()
        plt.savefig(os.path.join(self.save_path, 'density_Y_{}.pdf'.format(split)), dpi=200)
        plt.close()


    @torch.no_grad()
    def _plot_heatmap(self, z, z_cov):
        """
        Generate a Gaussian heatmap for a given embedding.

        Args:
            z (torch.Tensor): Scaled embedding.
            z_cov (torch.Tensor): Covariance matrix for the Gaussian.

        Returns:
            torch.Tensor: Generated heatmap.
        """
        heatmap = torch.zeros(self.hm_size, dtype=torch.float32, device='cpu')
        z_rint = torch.round(z).int()

        # Size of Gaussian window, PLEASE KEEP DIVISIBLE BY 2
        size = 20
        assert size % 2 == 0

        # Check whether gaussian window intersects with image:
        if (z_rint[0] - (size // 2) >= self.hm_size[0]) or (z_rint[0] + (size // 2) <= 0) or \
           (z_rint[1] - (size // 2) >= self.hm_size[1]) or (z_rint[1] + (size // 2) <= 0):

            return heatmap
        
        else:
            # Define the grid
            us = torch.linspace(z_rint[0] - (size // 2), z_rint[0] + (size // 2), size + 1, device='cpu')
            vs = torch.linspace(z_rint[1] - (size // 2), z_rint[1] + (size // 2), size + 1, device='cpu')
            u, v = torch.meshgrid(us, vs, indexing='ij')

            # Flatten X and Y to create a 2xN array where N is the number of points
            uv = torch.stack((u.flatten(), v.flatten()), dim=0)
            z_cov_inv = torch.linalg.inv(z_cov)

            exponent = torch.exp(-0.5 * ((uv - torch.round(z).unsqueeze(1)).T @ z_cov_inv @ (uv - torch.round(z).unsqueeze(1))))
            z_hm = exponent.diagonal()
            z_hm = z_hm.reshape(u.shape)

            # Identify indices in im that will define the crop area
            top = max(0, z_rint[0] - (size // 2))
            bottom = min(self.hm_size[0], z_rint[0] + (size // 2) + 1)
            left = max(0, z_rint[1] - (size // 2))
            right = min(self.hm_size[1], z_rint[1] + (size // 2) + 1)

            heatmap[top:bottom, left:right] = z_hm[top - (z_rint[0] - (size // 2)): top - (z_rint[0] - (size // 2)) + (bottom - top),
                  left - (z_rint[1] - (size // 2)): left - (z_rint[1] - (size // 2)) + (right - left)]

            return heatmap
    

    @torch.no_grad()
    def _generate_heatmap(self, z):
        """
        Generate a heatmap for a batch of embeddings.

        Args:
            z (torch.Tensor): Batch of scaled embeddings.

        Returns:
            torch.Tensor: Generated heatmap.
        """
        heatmap = torch.zeros(self.hm_size, dtype=torch.float32, device='cpu')

        # Save computations by not plotting the same location twice
        locations_on_hm = list()

        for z_i in z:
            assert (z_i > 0).all() and (z_i < max(self.hm_size)).all()

            # Heatmap corresponding to this location already present
            if str(torch.round(z_i)) in locations_on_hm:
                continue

            hm_ = self._plot_heatmap(z_i, torch.tensor([[8., 0.], [0., 8.]]))
            heatmap = torch.maximum(heatmap, hm_)
            
            locations_on_hm.append(str(torch.round(z_i)))

        return heatmap


    @torch.no_grad()
    def preprocess(self, split, num_frames):
        """
        Preprocess the dataset for a given split.

        Args:
            split (str): Dataset split ('train' or 'test').
            num_frames (int): Number of frames to process.
        """
        print('Preprocessing for __getitem__.')
        assert split in ['train', 'test']

        if self.current_split == split:
            return

        self.current_split = split

        # We do this to get the length of the dataset
        if self.dataset == 'Human36M':
            loader = h36mdataloader(split=split, batch_size=1, mmgt_stride=self.mmgt_stride, num_frames=num_frames)
        else:
            assert self.dataset == 'AMASS'
            loader = amassdataloader(split=split, batch_size=1, mmgt_stride=self.mmgt_stride, num_frames=num_frames)

        num_samples = len(loader.dataset)
        self.dataset_idx = list(range(num_samples))

        self.loader = loader


    def get_codebook(self):
        """
        Create a codebook mapping heatmap locations to embeddings.

        Returns:
            dict: Codebook with heatmap locations as keys and embeddings as values.
        """
        # Written with the help of Copilot. Which probably took inspiration from someone else.
        logging.info('Creating the codebook')
        codebook = dict()

        top_padding = self.pad_heatmap[2]
        left_padding = self.pad_heatmap[0]

        padding = torch.tensor((top_padding, left_padding)).int()

        # Create a codebook
        max_dim = max(self.hm_size[0], self.hm_size[1])
        for location in product(range(max_dim), repeat=2):
            codebook[location] = []

        # Populate the codebook
        for z, embedding in track(zip(self.z_reduced['train'][1], self.z_reduced['train'][0])):
            z = z.unsqueeze(0)
            z = self.scale_embeddings(z)
            z_rint = torch.round(z).int().squeeze()

            # Add padding
            z_rint += padding
            z_rint = tuple(z_rint.tolist())
            assert z_rint in codebook.keys(), "{}".format(z_rint)
            codebook[z_rint].append(embedding)

        # How many samples in a codebook location
        n_samples_codebook = np.zeros((max_dim, max_dim))
        for location in product(range(max_dim), repeat=2):
            n_samples_codebook[location] = len(codebook[location])
        fig, ax = plt.subplots()
        cax = ax.imshow(n_samples_codebook, cmap='viridis')
        fig.colorbar(cax)
        plt.savefig(os.path.join(self.save_path, 'codebook.pdf'), dpi=150)
        plt.close()

        # Ensure there are no empty keys in the codebook
        # ...First create a new codebook since we do not want to modify the original
        # ...The original should not be modified since it affects the update process
        _codebook = dict()
        max_dim = max(self.hm_size[0], self.hm_size[1])
        for location in product(range(max_dim), repeat=2):
            _codebook[location] = []

        for location in product(range(max_dim), repeat=2):
            if len(codebook[location]) > 0:
                _codebook[location] = codebook[location]

            else:
                nearest_nonempty = find_nearest_non_empty(codebook, location)
                _codebook[location] = codebook[nearest_nonempty]

        codebook = _codebook

        # How many samples in a codebook location
        n_samples_codebook = np.zeros((max_dim, max_dim))
        for location in product(range(max_dim), repeat=2):
            n_samples_codebook[location] = len(codebook[location])
        fig, ax = plt.subplots()
        cax = ax.imshow(n_samples_codebook, cmap='viridis')
        # Add a colorbar
        fig.colorbar(cax)
        plt.savefig(os.path.join(self.save_path, 'codebook_updated.pdf'), dpi=150)
        plt.close()

        # Compute the mean and update the codebook location
        for location in product(range(max_dim), repeat=2):
            samples = codebook[location]
            samples = torch.stack(samples)
            
            mean = torch.mean(samples, dim=0)
            codebook[location] = mean

        return codebook


    @torch.no_grad()
    def invert_scaling(self, z_hm):
        """
        Go from 2D Heatmap to projection space
        This function works in tandem with local maxima
        This function has two behaviors depending on the use_codebook flag
        """
        assert z_hm.dim() == 2  # Number of modes x 2
      
        # Store embeddings
        embeddings = list()

        for z in z_hm:

            # Maximum should be 49 but I think never reached in practice
            if len(embeddings) == 49:
                    break
            else:
                z = tuple(z.tolist())
                embeddings.append(self.codebook[z])
                            
        return torch.stack(embeddings)
        
        
    def __getitem__(self, i):
        """
        Get a data sample and its corresponding heatmap.

        Args:
            i (int): Index of the sample.
        Returns:
            tuple: Input data, heatmap, and density.
        """
        if self.strided:
            idx = int(self.stride * i)
        else:
            idx = i

        if self.use_augmentation:
            offset = np.random.randint(-self.augmentation, self.augmentation + 1) 
            idx = max(0, min(idx + offset, len(self.dataset_idx) - 1))

        x, _, _ = self.loader.dataset.__getitem__(idx, random_select=True)
        mm_gt_idx = self.loader.dataset.mm_gt_indxs[idx].tolist()

        # Select training embeddings
        mm_z = self.z_reduced[self.current_split][1][mm_gt_idx]

        z_hm = self.scale_embeddings(mm_z)

        hm = self._generate_heatmap(z_hm)
        hm = F.pad(hm, pad=self.pad_heatmap, mode='constant', value=0)

        return x, hm #self.density[x_hm[0].int(), x_hm[1].int()]

 
    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        if self.strided:
            return (len(self.dataset_idx) // self.stride) \
                + bool(len(self.dataset_idx) % self.stride)
        else:
            return len(self.dataset_idx)


class MultimodalDataset(torch.utils.data.Dataset):
    """
    A dataset class for handling multimodal data.

    Attributes:
        dataset (str): Name of the dataset.
        recover_landmarks (callable): Function to recover landmarks.
        stride (int): Stride for sampling data.
        mmgt_stride (int): Stride for motion map ground truth.
        augmentation (int): Augmentation range for data.
    """

    def __init__(self, conf, strided=True):
        """
        Initialize the MultimodalDataset object.

        Args:
            conf (object): Configuration object with dataset settings.
            strided (bool): Whether to use strided sampling.
        """
        
        print('Initializing multimodal object.')
        self.dataset = conf.dataset

        self.recover_landmarks = None

        self.current_split = None
        self.use_augmentation = None

        self.stride = conf.dataset_settings[self.dataset]['stride']
        self.mmgt_stride = conf.dataset_settings[self.dataset]['mmgt_stride']
        self.augmentation = conf.dataset_settings[self.dataset]['augmentation']

        self.strided = strided


    @torch.no_grad()
    def preprocess(self, split, num_frames):
        """
        Preprocess the dataset for a given split.

        Args:
            split (str): Dataset split ('train' or 'test').
            num_frames (int): Number of frames to process.
        """
        
        print('Preprocessing for __getitem__.')
        assert split in ['train', 'test']

        self.current_split = split

        # We do this to get the length of the dataset
        if self.dataset == 'Human36M':
            loader = h36mdataloader(split=split, batch_size=1, mmgt_stride=self.mmgt_stride, num_frames=num_frames)
        else:
            assert self.dataset == 'AMASS'
            loader = amassdataloader(split=split, batch_size=1, mmgt_stride=self.mmgt_stride, num_frames=num_frames)

        self.recover_landmarks = loader.dataset.recover_landmarks
        num_samples = len(loader.dataset)
        self.dataset_idx = list(range(num_samples))

        self.loader = loader


    def __getitem__(self, i):
        """
        Get a data sample and its corresponding metadata.

        Args:
            i (int): Index of the sample.
            debug (bool): Whether to enable debug mode.

        Returns:
            tuple: Input data, ground truth, and metadata.
        """
        
        if self.strided:
            idx = int(self.stride * i)
        else:
            idx = i

        if self.use_augmentation:
            offset = np.random.randint(-self.augmentation, self.augmentation + 1) 
            idx = max(0, min(idx + offset, len(self.dataset_idx) - 1))
            
        x, y, extra = self.loader.dataset.__getitem__(idx, random_select=True)
        mm_y = extra['mm_gt']
        
        x_action = extra["metadata"]
        mm_y_action = extra["metadata_mmgt"]
        mm_gt_idx = extra['mm_gt_idx']
                
        if self.use_augmentation:
            x, y, mm_y = mirror_pose_sequence(x=x, y=y, mm_gt=mm_y, prob=0.5)
            x, y, mm_y = rotate_pose_sequence(x=x, y=y, mm_gt=mm_y, prob=0.5)
                
        
        return x.astype(np.float32), mm_y.astype(np.float32), y.astype(np.float32), \
            {'idx': idx, 'x_action': x_action, 'mm_y_action': mm_y_action, 'mm_gt_idx': mm_gt_idx}



    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        if self.strided:
            return (len(self.dataset_idx) // self.stride) \
                + bool(len(self.dataset_idx) % self.stride)
        else:
            return len(self.dataset_idx)
        

class FineTuneDataset(torch.utils.data.Dataset):
    """
    A dataset class for fine-tuning using motion map embeddings.

    Attributes:
        dataset (str): Name of the dataset.
        z_reduced (dict): Reduced embeddings for train and test splits.
        scale_matrix (torch.Tensor): Scaling matrix for embeddings.
        origin_shift (torch.Tensor): Origin shift for embeddings.
        hm_size (list): Size of the heatmap.
        padding (torch.Tensor): Padding for the heatmap.
        codebook (dict): Codebook mapping heatmap locations to embeddings.
    """

    def __init__(self, motionmap_dataset, conf):
        """
        Initialize the FineTuneDataset object.

        Args:
            motionmap_dataset (MotionmapDataset): Motion map dataset object.
            conf (object): Configuration object with dataset settings.
        """
        # Dataset name
        self.dataset = conf.dataset

        self.current_split = None
        self.strided = None
        self.use_augmentation = None

        # Set stride and augmentation
        self.stride = conf.dataset_settings[self.dataset]['stride']
        self.mmgt_stride = conf.dataset_settings[self.dataset]['mmgt_stride']
        self.augmentation = conf.dataset_settings[self.dataset]['augmentation']

        # Initializations from the hm dataset
        self.z_reduced = motionmap_dataset.z_reduced
        self.scale_matrix = motionmap_dataset.scale_matrix
        self.origin_shift = motionmap_dataset.origin_shift
        self.hm_size = motionmap_dataset.hm_size
        self.padding = torch.tensor((motionmap_dataset.pad_heatmap[2], motionmap_dataset.pad_heatmap[0])).int()
        self.codebook = motionmap_dataset.codebook

        self.conf = conf


    def preprocess(self, split, num_frames):
        """
        Preprocess the dataset for a given split.

        Args:
            split (str): Dataset split ('train' or 'test').
            num_frames (int): Number of frames to process.
        """
        print('Preprocessing for __getitem__.')
        assert split in ['train', 'test']

        self.current_split = split

        # We do this to get the length of the dataset
        if self.dataset == 'Human36M':
            loader = h36mdataloader(split=split, batch_size=1, mmgt_stride=self.mmgt_stride, num_frames=num_frames)
        else:
            assert self.dataset == 'AMASS'
            loader = amassdataloader(split=split, batch_size=1, mmgt_stride=self.mmgt_stride, num_frames=num_frames)

        num_samples = len(loader.dataset)
        self.dataset_idx = list(range(num_samples))

        self.loader = loader


    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        if self.strided:
            return (len(self.dataset_idx) // self.stride) \
                + bool(len(self.dataset_idx) % self.stride)
        else:
            return len(self.dataset_idx)
    

    def __getitem__(self, i):
        """
        Get a data sample and its corresponding embedding.

        Args:
            i (int): Index of the sample.

        Returns:
            tuple: Input data, ground truth, and embedding.
        """
        # Get idx after striding and augmentation
        if self.strided:
            idx = int(self.stride * i)
        else:
            idx = i

        if self.use_augmentation:
            offset = np.random.randint(-self.augmentation, self.augmentation + 1) 
            idx = max(0, min(idx + offset, len(self.dataset_idx) - 1))

        x, _, extra = self.loader.dataset.__getitem__(idx, random_select=True)
        mm_y = extra['mm_gt']
        mm_gt_idx = extra['mm_gt_idx']

        z = self.z_reduced[self.current_split][1][mm_gt_idx].clone()
        z = self.scale_matrix @ (z - self.origin_shift).unsqueeze(1)
        z = z.squeeze(-1)
        z = torch.round(z).int()
        z += self.padding

        z = tuple(z.tolist())
        embedding = self.codebook[z]

        return x.astype(np.float32), mm_y.astype(np.float32), embedding, idx
