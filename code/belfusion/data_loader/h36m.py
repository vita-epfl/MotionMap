"""
We have primarily modified these files in multmodal ground truth creation (get_mm_gt_inx)
We have also modified the __getitem__ function
"""

import os
import random


import fcntl
import torch
import cdflib
import hashlib
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
from functools import partial


from utilities import load_mmgt
from utils.skeleton import SkeletonH36M
from base import BaseDataLoader, BaseMultiAgentDataset
from belfusion.data_loader.collate_functions import collate


#CACHE_PATH = "/mnt/vita/scratch/vita-staff/users/meghshukla/GitHub/HumanPoseForecasting/motion_cache_chunk"
THRESHOLD = 0.5
CACHE_PATH = os.path.join(Path(__file__).resolve().parents[3], 'mmgt_cache')


class H36MDataset(BaseMultiAgentDataset):
    def __init__(self, annotations_folder, subjects, actions, 
                precomputed_folder, obs_length, pred_length, use_vel=False,
                stride=1, augmentation=0, segments_path=None, normalize_data=True, normalize_type='standardize',
                drop_root=False, dtype='float64', 
                da_mirroring=0.0, da_rotations=0.0, mmgt_stride=-1, num_frames=-1): # data augmentation strategies

        assert (subjects is not None and actions is not None) or segments_path is not None
        self.annotations_folder = annotations_folder
        self.segments_path = segments_path
        self.subjects, self.actions = subjects, actions
        self.use_vel = use_vel 
        self.drop_root = drop_root # for comparison against DLow/Smooth4Diverse
        self.dict_indices = {} # dict_indices[subject][action] indicated idx where subject-action annotations start.
        self.metadata_class_idx = 1 # 0: subject, 1: action --> action is the class used for metrics computation
        self.idx_to_class = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        self.class_to_idx = {v: k for k, v in enumerate(self.idx_to_class)}
        self.mean_motion_per_class = [0.004533339312024582, 0.005071772030221925, 0.003968115058494981, 0.00592599384929542, 0.003590651675618232, 0.004194935839372698, 0.005625120976387903, 0.0024796492124910586, 0.0035406092427418797, 0.003602172245980421, 0.004347639393585013, 0.004222595821256223, 0.007537553520400006, 0.007066049169369122, 0.006754175094952483]

        assert da_mirroring >= 0.0 and da_mirroring <= 1.0 and da_rotations >= 0.0 and da_rotations <= 1.0, \
            "Data augmentation strategies must be in [0, 1]"
        
        self.augmentation = augmentation
        self.da_mirroring = da_mirroring
        self.da_rotations = da_rotations
        self.mmgt_stride = mmgt_stride

        super().__init__(precomputed_folder, obs_length, pred_length, augmentation=augmentation,
                         stride=stride, normalize_data=normalize_data,
                         normalize_type=normalize_type, dtype=dtype)
        

        if ("S9" in subjects and "S11" in subjects) and (self.segments_path):
            print("____test split____")
            self.mm_gt_indxs = self.get_mm_gt_inx(num_frames=num_frames, threshold=THRESHOLD)
            # This is because test split already is strided based on the segments csv.
            assert mmgt_stride == 1, "Stride for mm_gt computation must be set to 1 for test split"
        else:
            print("____train split____")
            self.mm_gt_indxs = self.get_mm_gt_inx(num_frames=num_frames, threshold=THRESHOLD) 

    
    def _get_hash_str(self, use_all=False):
        use_all = [str(self.obs_length), str(self.pred_length), str(self.stride), str(self.augmentation)] if use_all else []
        to_hash = "".join(tuple(self.subjects + list(self.actions) + 
                [str(self.drop_root), str(self.use_vel)] + use_all))
        return str(hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest())

    def _prepare_data(self, num_workers=8):
        if self.segments_path:
            self.segments, self.segment_idx_to_metadata = self._load_annotations_and_segments(self.segments_path, num_workers=num_workers)
            self.stride = 1
            self.augmentation = 0
        else:
            self.annotations = self._read_all_annotations(self.subjects, self.actions)
            self.segments, self.segment_idx_to_metadata = self._generate_segments()
            
    def _init_skeleton(self):
        self.skeleton = SkeletonH36M(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                          16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                                 joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                 joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
        self.removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
        self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
        self.skeleton.remove_joints(self.removed_joints)
        self.skeleton._parents[11] = 8
        self.skeleton._parents[14] = 8

    def _read_all_annotations(self, subjects, actions):
        preprocessed_path = os.path.join(self.precomputed_folder, 'data_3d_h36m.npz')
        if not os.path.exists(preprocessed_path):
            # call function that preprocesses dataset from dataset folder
            preprocess_dataset(self.annotations_folder, output_path=preprocessed_path) # borrowed from VideoPose3D repository

        # we load from already preprocessed dataset
        data_o = np.load(preprocessed_path, allow_pickle=True)['positions_3d'].item()
        data_f = dict(filter(lambda x: x[0] in subjects, data_o.items()))
        if actions != 'all': # if not all, we only keep the data from the selected actions, for each participant
            for subject in list(data_f.keys()):
                #data_f[key] = dict(filter(lambda x: all([a in x[0] for a in actions]), data_f[key].items())) # OLD and wrong
                data_f[subject] = dict(filter(lambda x: any([a in x[0] for a in actions]), data_f[subject].items()))
                if len(data_f[subject]) == 0: # no actions for subject => delete
                    data_f.pop(subject)
                    print(f"Subject '{subject}' has no actions available from '{actions}'.")
        else:
            print(f"All actions loaded from {subjects}.")

        # we build the feature vectors for each participant and action
        for subject in data_f.keys():
            for action in data_f[subject].keys():
                seq = data_f[subject][action][:, self.kept_joints, :]
                if self.use_vel:
                    v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                    v = np.append(v, v[[-1]], axis=0)
                seq[:, 1:] -= seq[:, :1] # we make them root-relative (root-> joint at first position)
                if self.use_vel:
                    seq = np.concatenate((seq, v), axis=1) # shape -> 17+1 (vel only from root joint)
                data_f[subject][action] = seq
        self.data = data_f


        anns_all = []
        self.dict_indices = {}
        self.clip_idx_to_metadata = []
        counter = 0
        for subject in self.data:
            self.dict_indices[subject] = {}

            for action in self.data[subject]:
                self.dict_indices[subject][action] = counter
                self.clip_idx_to_metadata.append((subject, action.split(" ")[0]))
                counter += 1

                anns_all.append(self.data[subject][action][:, None].astype(self.dtype)) # participants axis expanded
        
        self._generate_statistics_full(anns_all)

        return anns_all

    def _load_annotations_and_segments(self, segments_path, num_workers=8):
        assert os.path.exists(segments_path), "The path specified for segments does not exist: %s" % segments_path
        df = pd.read_csv(segments_path)
        subjects, actions = list(df["subject"].unique()), list(df["action"].unique())
        self.annotations = self._read_all_annotations(subjects, actions)
        
        segments = [(self.dict_indices[row["subject"]][row["action"]], 
                    row["pred_init"] - self.obs_length, 
                    row["pred_init"] + self.pred_length - 1) 
                        for i, row in df.iterrows()]

        segment_idx_to_metadata = [(row["subject"], row["action"].split(" ")[0]) for i, row in df.iterrows()]
                        
        #print(segments)
        #print(self.dict_indices)
        return segments, segment_idx_to_metadata

    def get_custom_segment(self, subject, action, frame_num):
        counter = self.dict_indices[subject][action]
        obs, pred = self._get_segment(counter, frame_num, frame_num + self.seg_length - 1)
        return obs, pred

    def recover_landmarks(self, data, rrr=True, fill_root=False):
        if self.normalize_data:
            data = self.denormalize(data)
        # data := (BatchSize, SegmentLength, NumPeople, Landmarks, Dimensions)
        # or data := (BatchSize, NumSamples, DiffusionSteps, SegmentLength, NumPeople, Landmarks, Dimensions)
        # the idea is that it does not matter how many dimensions are before NumPeople, Landmarks, Dimension => always working right
        if rrr:
            assert data.shape[-2] == 17 or (data.shape[-2] == 16 and fill_root), "Root was dropped, so original landmarks can not be recovered"
            if data.shape[-2] == 16 and fill_root:
                # we fill with a 'zero' imaginary root
                size = list(data.shape[:-2]) + [1, data.shape[-1]] # (BatchSize, SegmentLength, NumPeople, 1, Dimensions)
                return np.concatenate((np.zeros(size), data), axis=-2) # same, plus 0 in the root position
            data[..., 1:, :] += data[..., :1, :]
        return data

    def denormalize(self, x):
        if self.drop_root:
            if x.shape[-2] == 16:
                return super().denormalize(x, idces=list(range(1, 17)))
            elif x.shape[-2] == 17:
                return super().denormalize(x, idces=list(range(17)))
            else:
                raise Exception("'x' can't have shape != 16 or 17")
        return super().denormalize(x)

    def __getitem__(self, idx, random_select=False):

        obs, pred, extra = super(H36MDataset, self).__getitem__(idx)
        obs, pred = obs[..., 1:, :], pred[..., 1:, :]
        
        mm_gt = -1
        
        mm_gt_idices = self.mm_gt_indxs[idx]
        
        if random_select:
            mm_gt_idx = random.choice(mm_gt_idices)
            _, mm_gt, mm_gt_extra = super(H36MDataset, self).__getitem__(mm_gt_idx)
            mm_gt = mm_gt[..., 1:, :]
            mm_gt_action_subject = mm_gt_extra["metadata"]
            
            extra["mm_gt_idx"] = mm_gt_idx

        else:
            mm_gt = np.array([super(H36MDataset, self).__getitem__(i)[1][...,1:,:] 
                            for i in mm_gt_idices]) #(25, 1, 17, 3) #obs
            # mm_gt_action_subject is not needed for quantitative evaluations
            mm_gt_action_subject = [(None, None)] * len(mm_gt_idices)
        
        extra["mm_gt"] = mm_gt 
        extra["metadata_mmgt"] = mm_gt_action_subject

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
        len_dataset = super(H36MDataset, self).__len__()

        assert self.mmgt_stride != -1, "Stride for mm_gt computation must be set"

        try:
            saved_mm_gt = torch.load(os.path.join(
                CACHE_PATH,
                'Human36M_Frames_{}_Threshold_{}_len_{}.pt'.format(num_frames, threshold, len_dataset)))

            print('Loaded neighbours from: {}'.format(os.path.join(
                CACHE_PATH,
                'Human36M_Frames_{}_Threshold_{}_len_{}.pt'.format(num_frames, threshold, len_dataset))))
        
        except FileNotFoundError:
            print('Could not find saved mm_gt_indxs.')
            print('Finding neighbors using Frames: {}\tThreshold: {}'.format(num_frames, threshold))

            # Create a folder if it does not exist to store chunks
            chunk_location = os.path.join(
                CACHE_PATH, 'chunks', 'Human36M_Frames_{}_Threshold_{}_len_{}'.format(
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
                        [super(H36MDataset, self).__getitem__(idx)[0][-num_frames:, :, 1:, :]
                        for idx in range(len_dataset) if idx % self.mmgt_stride == 0])
                    
                    all_mmgt_x = torch.from_numpy(all_mmgt_x).cuda()

                    print('MMGT pivots shape: ', all_mmgt_x.shape)

                partial_load_mmgt = partial(load_mmgt,
                    num_frames=num_frames, dataset=super(H36MDataset, self).__getitem__,
                    dataset_name="Human36M", threshold=threshold,
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
                    'Human36M_Frames_{}_Threshold_{}_len_{}.pt'.format(num_frames, threshold, len_dataset)))
            
            print('Saved neighbours in: {}'.format(CACHE_PATH))

        return saved_mm_gt
    
                               
class H36MDataLoader(BaseDataLoader):
    def __init__(self, batch_size, annotations_folder, precomputed_folder, obs_length, pred_length, validation_split=0.0, subjects=None, actions=None, 
                    stride=1, shuffle=True, num_workers=1, num_workers_dataset=1, augmentation=0, segments_path=None, use_vel=False, seed=0,
                    normalize_data=True, normalize_type='standardize', drop_root=False, drop_last=True, dtype='float64', samples_to_track=None,
                    da_mirroring=0.0, da_rotations=0.0, mmgt_stride=-1, num_frames=-1):
                    
        self.dataset = H36MDataset(annotations_folder, subjects, actions, precomputed_folder, obs_length, pred_length, 
                                            stride=stride, augmentation=augmentation, segments_path=segments_path, use_vel=use_vel,
                                            normalize_data=normalize_data, normalize_type=normalize_type, drop_root=drop_root, dtype=dtype,
                                            da_mirroring=da_mirroring, da_rotations=da_rotations, mmgt_stride=mmgt_stride,
                                            num_frames=num_frames)
    
        # super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, seed, drop_last=drop_last, samples_to_track=samples_to_track) #new
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, seed, drop_last=drop_last, samples_to_track=samples_to_track, collate_fn=collate)


OUTPUT_3D = 'data_3d_h36m'
SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

def preprocess_dataset(dataset_folder, output_path=OUTPUT_3D, subjects=SUBJECTS):
    
    if os.path.exists(output_path):
        print('The dataset already exists at', output_path)
        exit(0)
        
    print('Converting original Human3.6M dataset from', dataset_folder, '(CDF files)')
    output = {}
    
    for subject in tqdm(subjects):
        output[subject] = {}
        
        #new
        # file_list = glob(os.path.join(dataset_folder, f'Poses_D3_Positions_{subject}', subject, 'MyPoseFeatures', 'D3_Positions', '*.cdf'))
        file_list = glob(os.path.join(dataset_folder, subject, 'MyPoseFeatures', 'D3_Positions', '*.cdf'))
        
        
        assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
        for f in file_list:
            action = os.path.splitext(os.path.basename(f))[0]
            
            if subject == 'S11' and action == 'Directions':
                continue # Discard corrupted video
                
            # Use consistent naming convention
            canonical_name = action.replace('TakingPhoto', 'Photo') \
                                    .replace('WalkingDog', 'WalkDog')
            
            hf = cdflib.CDF(f)
            positions = hf['Pose'].reshape(-1, 32, 3)
            positions /= 1000 # Meters instead of millimeters
            output[subject][canonical_name] = positions.astype('float32')
    
    print(f'Saving into "{output_path}"...')
    np.savez_compressed(output_path, positions_3d=output)
    print('Done.')
