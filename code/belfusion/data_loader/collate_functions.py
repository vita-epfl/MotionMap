import torch

def collate(batch):
    x_r, y_r, mm_gt, mm_gt_split = [], [], [], [0]
    for x, y, extra in batch:
        x_r.append(x)
        y_r.append(y)
        mm_gt.append(extra["mm_gt"])
        mm_gt_split.append(mm_gt_split[-1] + len(extra["mm_gt"]))
    return torch.utils.data.dataloader.default_collate(x_r),torch.utils.data.dataloader.default_collate(y_r),mm_gt


def collate_multimodal(batch):
    x_r, y_r, mm_y_r, i_r = [], [], [], []
    for x, y, mm_y, i in batch:
        x_r.append(x)
        y_r.append(y)
        mm_y_r.append(torch.from_numpy(mm_y))
        i_r.append(i)
    return torch.utils.data.dataloader.default_collate(x_r), \
           torch.utils.data.dataloader.default_collate(y_r), \
           mm_y_r, \
           torch.utils.data.dataloader.default_collate(i_r)


def collate_heatmap(batch):
    training_tuple, x_tuple, x_on_hm, y_tuple, density = zip(*batch)

    # For fixed-size tensors, we can directly stack them
    x_sample = torch.stack([item[0] for item in training_tuple])
    hm = torch.stack([item[1] for item in training_tuple])

    x = torch.stack([item[0] for item in x_tuple])
    x_mean = torch.stack([item[1] for item in x_tuple])
    x_cov = torch.stack([item[2] for item in x_tuple])

    x_hm = torch.stack([item[0] for item in x_on_hm])
    x_hm_cov = torch.stack([item[1] for item in x_on_hm])

    mm_y = [item[0] for item in y_tuple]
    z_hm = [item[1] for item in y_tuple]
    z_hm_cov = [item[2] for item in y_tuple]

    density = density

    return (x_sample, hm), \
           (x, x_mean, x_cov), \
           (x_hm, x_hm_cov), \
           (mm_y, z_hm, z_hm_cov), \
           density