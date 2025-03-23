"""
We have primarily modified these files in multmodal ground truth creation (get_mm_gt_inx)
We have also modified the __getitem__ function
"""

import os
import random


import zarr
import fcntl
import torch
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from functools import partial


from utilities import load_mmgt
from utils.skeleton import SkeletonAMASS
from base import BaseDataLoader, BaseMultiAgentDataset
from belfusion.data_loader.collate_functions import collate


THRESHOLD = 0.4
CACHE_PATH = os.path.join(Path(__file__).resolve().parents[3], 'mmgt_cache')
#CACHE_PATH = "/mnt/vita/scratch/vita-staff/users/meghshukla/GitHub/HumanPoseForecasting/motion_cache_chunk"


class AMASSDataset(BaseMultiAgentDataset):
    def __init__(self, annotations_folder, datasets, file_idces, 
                precomputed_folder, obs_length, pred_length, use_vel=False,
                stride=1, augmentation=0, segments_path=None, normalize_data=True, normalize_type='standardize',
                drop_root=False, dtype='float64', 
                da_mirroring=0.0, da_rotations=0.0, mmgt_stride=-1, num_frames=-1): # data augmentation strategies

        assert (datasets is not None and file_idces is not None) or segments_path is not None
        self.annotations_folder = annotations_folder
        self.segments_path = segments_path
        self.datasets, self.file_idces = datasets, file_idces
        assert self.file_idces == "all", "We only support 'all' files for now"
        self.use_vel = use_vel 
        self.drop_root = drop_root # for comparison against DLow/Smooth4Diverse
        self.dict_indices = {} # dict_indices[dataset][file_idx] indicated idx where dataset-file_idx annotations start.
        self.mm_indces = None
        self.metadata_class_idx = 0 # 0: dataset, 1: filename --> dataset is the class used for metrics computation
        self.idx_to_class = ['DFaust', 'DanceDB', 'GRAB', 'HUMAN4D', 'SOMA', 'SSM', 'Transitions']
        self.class_to_idx = {v: k for k, v in enumerate(self.idx_to_class)}
        self.mean_motion_per_class = [0.004860274970204714, 0.00815901767307159, 0.001774023530090276, 0.004391708416532331, 0.007596136106898701, 0.00575787090703614, 0.008530069935655568]

        assert da_mirroring >= 0.0 and da_mirroring <= 1.0 and da_rotations >= 0.0 and da_rotations <= 1.0, \
            "Data augmentation strategies must be in [0, 1]"
        
        self.augmentation = augmentation
        self.da_mirroring = da_mirroring
        self.da_rotations = da_rotations
        self.mmgt_stride = mmgt_stride

        super().__init__(precomputed_folder, obs_length, pred_length, augmentation=augmentation,
                         stride=stride, normalize_data=normalize_data,
                         normalize_type=normalize_type, dtype=dtype)
        
        #new:
        self.mm_gt_indxs = self.get_mm_gt_inx(num_frames=num_frames, threshold=THRESHOLD)

    
    def _get_hash_str(self, use_all=False):
        use_all = [str(self.obs_length), str(self.pred_length), str(self.stride), str(self.augmentation)] if use_all else []
        to_hash = "".join(tuple(self.datasets + list(self.file_idces) + 
                [str(self.drop_root), str(self.use_vel)] + use_all))
        return str(hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest())
            

    def _prepare_data(self, num_workers=8):
        if self.segments_path:
            self.segments, self.segment_idx_to_metadata = self._load_annotations_and_segments(self.segments_path, num_workers=num_workers)
            self.stride = 1
            self.augmentation = 0
            assert self.mmgt_stride == 1, "Stride for mm_gt computation must be set to 1 for test split"
        else:
            self.annotations = self._read_all_annotations(self.datasets, self.file_idces)
            self.segments, self.segment_idx_to_metadata = self._generate_segments()
            
    def _init_skeleton(self):
        # full list -> https://github.com/vchoutas/smplx/blob/43561ecabd23cfa70ce7b724cb831b6af0133e6e/smplx/joint_names.py#L166
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        left, right = [1, 4, 7, 10, 13, 16, 18, 20], [2, 5, 8, 11, 14, 17, 19, 21]
        self.skeleton = SkeletonAMASS(parents=parents,
                                 joints_left=left,
                                 joints_right=right,)
        self.removed_joints = {}
        self.kept_joints = np.array([x for x in range(22) if x not in self.removed_joints]) # 22
        self.skeleton.remove_joints(self.removed_joints)

    def _read_all_annotations(self, datasets, file_idces):
        if not os.path.exists(self.precomputed_folder):
            raise NotImplementedError("Preprocessing of AMASS dataset is not implemented yet. Please use the preprocessed data.")


        anns_all = []
        self.dict_indices = {}
        self.clip_idx_to_metadata = []
        counter = 0

        print("Loading datasets: ", datasets, file_idces)
        for dataset in datasets:
            self.dict_indices[dataset] = {}

            #print("Loading dataset: ", dataset)
            #print(os.path.join(self.precomputed_folder, dataset))
            z_poses = zarr.open(os.path.join(self.precomputed_folder, dataset, 'poses.zarr'), mode='r')
            z_trans = zarr.open(os.path.join(self.precomputed_folder, dataset, 'trans.zarr'), mode='r')
            z_index = zarr.open(os.path.join(self.precomputed_folder, dataset, 'poses_index.zarr'), mode='r')

            # we build the feature vectors for each dataset and file_idx
            #print(z_poses.shape, z_trans.shape, z_index.shape, z_index[-1])
            for file_idx in range(z_index.shape[0]):
                self.dict_indices[dataset][file_idx] = counter

                i0, i = z_index[file_idx]

                seq = z_poses[i0:i]
                #seq = np.concatenate([z_trans[i0:i][:, None], seq], axis=1) # add the root to the sequence
                seq[:, 1:] -= seq[:, :1] # we make them root-relative (root-> joint at first position)
                seq = seq[:, self.kept_joints, :]

                self.dict_indices[dataset][file_idx] = counter
                self.clip_idx_to_metadata.append((dataset, file_idx))
                counter += 1

                anns_all.append(seq[:, None].astype(self.dtype)) # datasets axis expanded

        self._generate_statistics_full(anns_all)
        return anns_all

    def _load_annotations_and_segments(self, segments_path, num_workers=8):
        assert os.path.exists(segments_path), "The path specified for segments does not exist: %s" % segments_path
        df = pd.read_csv(segments_path)
        # columns -> dataset,file,file_idx,pred_init,pred_end
        datasets, file_idces = list(df["dataset"].unique()), list(df["file_idx"].unique())
        self.annotations = self._read_all_annotations(datasets, "all")#file_idces)
        
        segments = [(self.dict_indices[row["dataset"]][row["file_idx"]], 
                    row["pred_init"] - self.obs_length, 
                    row["pred_init"] + self.pred_length - 1) 
                        for i, row in df.iterrows()]

        segment_idx_to_metadata = [(row["dataset"], row["file_idx"]) for i, row in df.iterrows()]
                        
        #print(segments)
        #print(self.dict_indices)
        return segments, segment_idx_to_metadata

    def get_custom_segment(self, dataset, file_idx, frame_num):
        counter = self.dict_indices[dataset][file_idx]
        obs, pred = self._get_segment(counter, frame_num, frame_num + self.seg_length - 1)
        return obs, pred

    def recover_landmarks(self, data, rrr=True, fill_root=False):
        if self.normalize_data:
            data = self.denormalize(data)
        # data := (BatchSize, SegmentLength, NumPeople, Landmarks, Dimensions)
        # or data := (BatchSize, NumSamples, DiffusionSteps, SegmentLength, NumPeople, Landmarks, Dimensions)
        # the idea is that it does not matter how many dimensions are before NumPeople, Landmarks, Dimension => always working right
        if rrr:
            assert data.shape[-2] == len(self.kept_joints) or (data.shape[-2] == len(self.kept_joints)-1 and fill_root), "Root was dropped, so original landmarks can not be recovered"
            if data.shape[-2] == len(self.kept_joints)-1 and fill_root:
                # we fill with a 'zero' imaginary root
                size = list(data.shape[:-2]) + [1, data.shape[-1]] # (BatchSize, SegmentLength, NumPeople, 1, Dimensions)
                return np.concatenate((np.zeros(size), data), axis=-2) # same, plus 0 in the root position
            data[..., 1:, :] += data[..., :1, :]
        return data

    def denormalize(self, x):
        if self.drop_root:
            if x.shape[-2] == len(self.kept_joints)-1:
                return super().denormalize(x, idces=list(range(1, len(self.kept_joints))))
            elif x.shape[-2] == len(self.kept_joints):
                return super().denormalize(x, idces=list(range(len(self.kept_joints))))
            else:
                raise Exception(f"'x' can't have shape != {len(self.kept_joints)-1} or {len(self.kept_joints)}")
        return super().denormalize(x)

    def __getitem__(self, idx, random_select=False):

        obs, pred, extra = super(AMASSDataset, self).__getitem__(idx)
        obs, pred = obs[..., 1:, :], pred[..., 1:, :]

        mm_gt = -1
        
        mm_gt_idices = self.mm_gt_indxs[idx]

        if random_select:
            mm_gt_idx = random.choice(mm_gt_idices)
            _, mm_gt, mm_gt_extra = super(AMASSDataset, self).__getitem__(mm_gt_idx)
            mm_gt = mm_gt[..., 1:, :]
            mm_gt_action_subject = (mm_gt_extra["metadata"][0], str(mm_gt_extra["metadata"][0]))# 'metadata': ('BMLhandball', 141)
            
            extra["mm_gt_idx"] = mm_gt_idx

        else:
            mm_gt = np.array([super(AMASSDataset, self).__getitem__(i)[1][...,1:,:] 
                            for i in mm_gt_idices])
            # mm_gt_action_subject is not needed for quantitative evaluations
            mm_gt_action_subject = [(None, None)] * len(mm_gt_idices)

        extra["mm_gt"] = mm_gt
        extra["metadata_mmgt"] = mm_gt_action_subject
        extra["metadata"] = (extra["metadata"][0], str(extra["metadata"][0])) # 'metadata': ('BMLhandball', 141)
        
        return obs, pred, extra


    @torch.no_grad()
    def get_mm_gt_inx(self, num_frames, threshold):
        """
        Computes and retrieves multi-modal ground truth (MMGT) indices for the dataset.

        Args:
            num_frames (int): The number of frames to consider for each segment.
            threshold (float): The threshold value for determining neighbors.

        Returns:
            dict: A dictionary where keys are dataset indices and values are lists of MMGT indices.

        This method attempts to load precomputed MMGT indices from a cache. If not found, it computes
        the MMGT indices by comparing segments of the dataset based on the given number of frames and
        threshold. The dataset is split into chunks for the same. The idea is that different executions
        of the program can compute different chunks in parallel, with the program finishing first 
        having the responsibility to merge the chunks into a single file. The chunks should be
        deleted after the merge, but I am not sure. The final file is saved in the CACHE_PATH.

        Steps:
        1. Check if cached MMGT indices exist. If found, load and return them.
        2. If not cached, divide the dataset into chunks and process each chunk to compute MMGT indices.
        3. Save the computed indices in chunks and merge them into a single file.
        4. Return the computed MMGT indices.

        Notes:
        - The method uses GPU acceleration for faster computation.
        - The `mmgt_stride` attribute must be set before calling this method.
        """

        len_dataset = super(AMASSDataset, self).__len__()
        
        assert self.mmgt_stride != -1, "Stride for mm_gt computation must be set"
        
        try:
            saved_mm_gt = torch.load(os.path.join(
                CACHE_PATH,
                'Amass_Frames_{}_Threshold_{}_len_{}.pt'.format(num_frames, threshold, len_dataset)))

            print('Loaded neighbours from: {}'.format(os.path.join(
                CACHE_PATH,
                'Amass_Frames_{}_Threshold_{}_len_{}.pt'.format(num_frames, threshold, len_dataset))))
            
        except FileNotFoundError:
            print('Could not find saved mm_gt_indxs.')
            print('Finding neighbors using Frames: {}\tThreshold: {}'.format(num_frames, threshold))

            # Create a folder if it does not exist to store chunks
            chunk_location = os.path.join(
                CACHE_PATH, 'chunks', 'Amass_Frames_{}_Threshold_{}_len_{}'.format(
                    num_frames, threshold, len_dataset))
            os.makedirs(chunk_location, exist_ok=True)
            
            all_mmgt_x = None

            # Create chunks of the mm_gt and save them
            chunk_size = 1000
            chunks = []
            for i in range(0, len_dataset, chunk_size):
                start = i
                end = min(i + chunk_size - 1, len_dataset - 1)
                chunks.append((start, end))
            random.shuffle(chunks)

            total_chunks = len(chunks)

            while len(chunks) > 0:
                # Read all saved chunks
                with open(os.path.join(chunk_location, "saved.txt"), "a+") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.seek(0)
                    locations = f.read().splitlines()
                    fcntl.flock(f, fcntl.LOCK_UN)

                try:
                    # Remove chunks that have already been processed
                    while str(chunks[0]) in locations:
                        chunks.pop(0)
                except IndexError:
                    print('All chunks have been processed')
                    break

                print('Processing chunks: {}\t\tNumber of chunks left: {}/{}'.format(
                    chunks[0], len(chunks), total_chunks))
            
                saved_mm_gt = dict()

                # First, we collect pivots for mmgt
                if all_mmgt_x is None:
                    all_mmgt_x = np.array(
                        [super(AMASSDataset, self).__getitem__(idx)[0][-num_frames:, :, 1:, :]
                        for idx in range(len_dataset) if idx % self.mmgt_stride == 0])
                    
                    all_mmgt_x = torch.from_numpy(all_mmgt_x).cuda()

                    print('MMGT pivots shape: ', all_mmgt_x.shape)
                

                partial_load_mmgt = partial(load_mmgt,
                    num_frames=num_frames, dataset=super(AMASSDataset, self).__getitem__,
                    dataset_name="AMASS", threshold=threshold,
                    mmgt_stride=self.mmgt_stride, all_mmgt_x=all_mmgt_x)
                
                for i in range(chunks[0][0], chunks[0][1] + 1):
                    saved_mm_gt[i] = partial_load_mmgt(i)

                torch.save(
                    saved_mm_gt,
                    os.path.join(chunk_location, '{}_{}.pt'.format(chunks[0][0], chunks[0][1])))
                
                print('Saved neighbours {} to {} in: {}'.format(
                    chunks[0][0], chunks[0][1], chunk_location))
                
                with open(os.path.join(chunk_location, "saved.txt"), "a+") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.write("{}\n".format(chunks[0]))
                    fcntl.flock(f, fcntl.LOCK_UN)

            # We have finished pieces, now we merge them
            assert len(chunks) == 0, "Some chunks were not processed"

            # Load the chunks
            saved_mm_gt = dict()
            for i in range(0, len_dataset, chunk_size):
                start = i
                end = min(i + chunk_size - 1, len_dataset - 1)
                saved_mm_gt.update(torch.load(
                    os.path.join(chunk_location, '{}_{}.pt'.format(start, end))))
                
            # Save them in CACHE_PATH
            torch.save(saved_mm_gt,
                os.path.join(
                    CACHE_PATH,
                    'Amass_Frames_{}_Threshold_{}_len_{}.pt'.format(num_frames, threshold, len_dataset)))

            print('Saved neighbours in: {}'.format(CACHE_PATH))

        return saved_mm_gt

    
class AMASSDataLoader(BaseDataLoader):
    def __init__(self, batch_size, annotations_folder, precomputed_folder, obs_length, pred_length, validation_split=0.0, datasets=None, file_idces='all', 
                    stride=1, shuffle=True, num_workers=1, num_workers_dataset=1, augmentation=0, segments_path=None, use_vel=False, seed=0,
                    normalize_data=True, normalize_type='standardize', drop_root=False, drop_last=True, dtype='float64', samples_to_track=None,
                    da_mirroring=0.0, da_rotations=0.0, mmgt_stride=-1, num_frames=-1):
                    
        self.dataset = AMASSDataset(annotations_folder, datasets, file_idces, precomputed_folder, obs_length, pred_length, 
                                            stride=stride, augmentation=augmentation, segments_path=segments_path, use_vel=use_vel,
                                            normalize_data=normalize_data, normalize_type=normalize_type, drop_root=drop_root, dtype=dtype,
                                            da_mirroring=da_mirroring, da_rotations=da_rotations, mmgt_stride=mmgt_stride,
                                            num_frames=num_frames)
    
        # super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, seed, drop_last=drop_last, samples_to_track=samples_to_track) #new
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, seed, drop_last=drop_last, samples_to_track=samples_to_track, collate_fn=collate)
