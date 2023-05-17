import torch
import numpy as np
from typing import Tuple, List


@torch.no_grad()
def warp(current_hm: torch.Tensor, foreground: torch.Tensor, foreground_flow3d: torch.Tensor, 
         pixel_size: float,
         point_cloud_range: np.ndarray,
         hm_threshold: float = 0.1,
         num_neighbors: int = 5) -> torch.Tensor:
    """
    Hard-code for batch_size == 1
    Args:
        current_hm: (1, n_cls, H, W)
        foreground: (N, 1 + 3 + C) - batch_idx, x, y, z, C channels
        foreground_flow3d: (N, 3) - scene-flow_x, scene-flow_y, scene-flow_z
        pixel_size: in meters of heatmap (= original voxel_size * heatmap stride)
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        hm_threshold: below threshold is considered zero
        num_neighbors: num of nearest foreground in BEV to be searched for each nonzero location of current_hm
    
    Returns:
        warped_current_hm: (1, n_cls, H, W)
    """
    batch_size, n_cls, height, width = current_hm.shape
    assert batch_size == 1, 'not support batch_size > 1'
    assert torch.all(foreground[:, 0].long() == 0), 'first column must be batch_idx, and must be 0 because only support batch_size == 1'
    
    if point_cloud_range[0] == point_cloud_range[1]:
        mask_in_pc_range = torch.logical_and(foreground[:, 1: 3] > point_cloud_range[0], 
                                             foreground[:, 1: 3] < point_cloud_range[3] - 1e-3).all(dim=1)
        fg_bev_xy = (foreground[mask_in_pc_range, 1: 3] - point_cloud_range[0]) / pixel_size  # (N_fg, 2)
        print('fg_bev_xy: ', fg_bev_xy.shape)
    else:
        raise NotImplementedError
    
    list_hm_xy, list_hm_val = find_nonzero_location_in_feat_map(current_hm, hm_threshold)

    warped_current_hm = torch.zeros_like(current_hm)
    for ch_idx, (hm_xy, hm_val) in enumerate(zip(list_hm_xy, list_hm_val)):
        # find k-nearest neighbor w/ fg_bev_xy
        dist = torch.square(hm_xy.unsqueeze(1) - fg_bev_xy.unsqueeze(0)).sum(dim=-1)  
        # (N_pos_current, 1, 2) - (1, N_fg, 2) -> (N_pos_current, N_fg, 2) -> (N_pos_current, N_fg) 
        _, neighbor_indices = torch.topk(dist, k=num_neighbors, dim=1, largest=False)  # (N_pos_current, num_neighbors) 

        # extract flow from foreground neighbors
        hm_flow = foreground_flow3d[neighbor_indices.reshape(-1), :2].reshape(hm_xy.shape[0], num_neighbors, 2).mean(dim=1)  # (N_pos_current, 2)
        hm_flow = hm_flow / pixel_size  # convert from flow in 3D to flow on BEV

        # move
        hm_xy = hm_xy + hm_flow

        # bilinear scatter
        warped_current_hm[0, ch_idx] = bilinear_scatter(hm_xy, hm_val, height, width)

    return warped_current_hm


@torch.no_grad()
def find_nonzero_location_in_feat_map(featmap: torch.Tensor, threshold: float = 0.1) -> Tuple[List[torch.Tensor]]:
    """
    Args:
        featmap: (1, C, H, W)
        threshold: below threshold is considered zero

    Returns:
        list_xy: [(N, 2)] - xy-coord of nonzero value of featmap | len == C
        list_val: [(N,)] - value of nonzero value of featmap | len == C
    """
    batch_size, channels, height, width = featmap.shape
    assert batch_size == 1, 'not support batch_size > 1'
    all_x = torch.arange(0, width)
    all_y = torch.arange(0, height)
    yy, xx = torch.meshgrid(all_y, all_x)  # (H, W), (H, W)

    mask_postive = featmap > threshold  # (1, C, H, W)

    list_xy, list_val = list(), list()
    for ch_idx in range(channels):
        list_xy.append(
            torch.stack((xx[mask_postive[0, ch_idx]], yy[mask_postive[0, ch_idx]]), dim=1)  # (N_pos_current, 2)
        )
        
        current_channel = featmap[0, ch_idx]  # (H, W)
        list_val.append(
            current_channel[mask_postive[0, ch_idx]]  # (N_pos_current)
        )

    return list_xy, list_val


@torch.no_grad()
def bilinear_scatter(xy: torch.Tensor, val: torch.Tensor, height: int, width: int):
    def column_prod(t: torch.Tensor):
        assert t.shape == (t.shape[0], 2)
        return t[:, 0] * t[:, 1]
    
    out = xy.new_zeros(height, width)
    
    x1y2 = torch.floor(xy).long()
    x2y1 = torch.ceil(xy).long()
    weight_x1y2 = column_prod(x2y1 - xy)  # (N,)
    weight_x2y1 = column_prod(xy - x1y2)  # (N,)
    

    x2y2 = torch.stack([torch.ceil(xy[:, 0]), torch.floor(xy[:, 1])], dim=1).long()
    x1y1 = torch.stack([torch.floor(xy[:, 0]), torch.ceil(xy[:, 1])], dim=1).long()
    weight_x2y2 = column_prod(torch.abs(xy - x1y1))  # (N,)
    weight_x1y1 = column_prod(torch.abs(xy - x2y2))  # (N,)

    list_xy_int = [x1y2, x2y1, x2y2, x1y1]
    list_weight = [weight_x1y2, weight_x2y1, weight_x2y2, weight_x1y1]

    sum_ = 0
    for weight in list_weight:
        sum_ = sum_ + weight  # (N,)

    for xy_int, weight_ in zip(list_xy_int, list_weight):
        xy_int[:, 0] = torch.clamp(xy_int[:, 0], min=0, max=width - 1)
        xy_int[:, 1] = torch.clamp(xy_int[:, 1], min=0, max=height - 1)
        out[xy_int[:, 1], xy_int[:, 0]] += val * weight_ / sum_

    return out
