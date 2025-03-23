import os
from belfusion.data_loader.h36m import H36MDataLoader
from belfusion.data_loader.amass import AMASSDataLoader


def amassdataloader(split='train', batch_size=128, obs_length=30, pred_length=120, mmgt_stride=-1, num_frames=-1):
    """
    Creates and returns a data loader for the AMASS dataset.

    Args:
        split (str): The dataset split to use ('train' or 'test').
        batch_size (int): The batch size for the data loader.
        obs_length (int): The number of observation frames.
        pred_length (int): The number of prediction frames.
        mmgt_stride (int): Stride for multi-modal ground truth calculation. Defaults to -1.
        num_frames (int): Total number of frames to use. Must be set.

    Returns:
        AMASSDataLoader: A data loader instance for the AMASS dataset.
    """
    
    assert num_frames != -1, "num_frames is not set for mm_gt calculation"

    from config import PATH
    
    if split == 'train':
        training = ["ACCAD", "BMLhandball", "BMLmovi", "BMLrub", "CMU", "EKUT",
                    "EyesJapanDataset", "KIT", "PosePrior", "TCDHands", "TotalCapture"]
        validation = ["HumanEva", "HDM05", "SFU", "MoSh"]
        datasets = training + validation
        segments_path = None

    else:
        assert split == 'test', "Wrong split selected"
        datasets = ['DFaust', 'DanceDB', 'GRAB', 'HUMAN4D', 'SOMA', 'SSM', 'Transitions']
        segments_path = os.path.join(PATH, "auxiliar/datasets/AMASS/segments_test.csv")
        mmgt_stride = 1
    
    annotations_folder = os.path.join(PATH, "datasets/single/AMASS")
    precomputed_folder = os.path.join(PATH, "auxiliar/datasets/AMASS")

    loader = AMASSDataLoader(batch_size, annotations_folder, precomputed_folder, 
                obs_length, pred_length, drop_root=True, 
                datasets=datasets, file_idces="all", drop_last=False,
                stride=1, shuffle=False, augmentation=0, normalize_data=False,
                dtype="float32", da_mirroring=0., da_rotations=0., segments_path=segments_path,
                mmgt_stride=mmgt_stride, num_frames=num_frames) 
    
    return loader


def h36mdataloader(split='train', batch_size=128, obs_length=25, pred_length=100, mmgt_stride=-1, num_frames=-1):
    """
    Creates and returns a data loader for the Human3.6M dataset.

    Args:
        split (str): The dataset split to use ('train' or 'test').
        batch_size (int): The batch size for the data loader.
        obs_length (int): The number of observation frames.
        pred_length (int): The number of prediction frames.
        mmgt_stride (int): Stride for multi-modal ground truth calculation. Defaults to -1.
        num_frames (int): Total number of frames to use. Must be set.

    Returns:
        H36MDataLoader: A data loader instance for the Human3.6M dataset.
    """
    
    assert num_frames != -1, "num_frames is not set for mm_gt calculation"

    from config import PATH

    if split == 'train':
        subjects = ["S1", "S5", "S6", "S7", "S8"]
        segments_path = None

    else:
        assert split == 'test', "Wrong split selected" 
        subjects = ["S9", "S11"]
        segments_path = os.path.join(PATH, "auxiliar/datasets/Human36M/segments_test.csv")
        mmgt_stride = 1


    annotations_folder = os.path.join(PATH, "datasets/Human36M/")
    precomputed_folder = os.path.join(PATH, "auxiliar/datasets/Human36M/")

    # Stride: 1 because we want to compute mm_gt for all frames.
    # Multimodal and heatmap dataloader set the strides in their __getitem__.

    # Augmentation is zero because of inconsistency with mm_gt.
    # Multimodal and heatmap dataloader set the augmentation in their __getitem__.

    # Batch size doesn't really matter since we index the underlying dataset anyways.

    # first time we call the data_loader, it will preprocess all subjects + generate statistics
    # with the subjects that we want to use for training
    loader = H36MDataLoader(batch_size, annotations_folder, precomputed_folder, 
                obs_length, pred_length, drop_root=True, 
                subjects=subjects, actions="all", drop_last=False,
                stride=1, shuffle=False, augmentation=0, normalize_data=False,
                dtype="float32", da_mirroring=0., da_rotations=0., segments_path=segments_path,
                mmgt_stride=mmgt_stride, num_frames=num_frames)
    
    return loader
