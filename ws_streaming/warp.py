# Copyright (c) OpenMMLab. All rights reserved.
# src: https://github.com/open-mmlab/mmflow/blob/master/mmflow/ops/warp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def coords_grid(flow: Tensor) -> Tensor:
    """Generate shifted coordinate grid based based input flow.

    Args:
        flow (Tensor): Estimated optical flow. (B, 2, H, W)

    Returns:
        Tensor: The coordinate that shifted by input flow and scale in the
            range [-1, 1].
    """
    B, _, H, W = flow.shape
    xx = torch.arange(0, W, device=flow.device, requires_grad=False)
    yy = torch.arange(0, H, device=flow.device, requires_grad=False)
    coords = torch.meshgrid(yy, xx)
    coords = torch.stack(coords[::-1], dim=0).float()
    grid = coords[None].repeat(B, 1, 1, 1) + flow
    grid[:, 0, ...] = grid[:, 0, ...] * 2. / max(W - 1, 1) - 1.
    grid[:, 1, ...] = grid[:, 1, ...] * 2. / max(H - 1, 1) - 1.
    grid = grid.permute(0, 2, 3, 1)
    return grid


class Warp(nn.Module):
    """Warping layer to warp feature using optical flow.

    Args:
        mode (str): interpolation mode to calculate output values. Options are
            'bilinear' and 'nearest'. Defaults to 'bilinear'.
        padding_mode (str): padding mode for outside grid values. Options are
            'zero', 'border' and 'reflection'. Defaults to 'zeros'.
        align_corners (bool): If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s corner
            pixels. If set to False, they are instead considered as referring
            to the corner points of the input’s corner pixels, making the
            sampling more resolution agnostic. Default to False.
    """

    def __init__(self,
                 mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 use_mask: bool = True) -> None:

        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.use_mask = use_mask

    def forward(self, feat: Tensor, flow: Tensor) -> Tensor:
        """Forward function for warp.

        Args:
            feat (Tensor): Input feature
            flow (Tensor): Input optical flow.

        Returns:
            Tensor: The output feature that was generated by warping input
                feature based input flow.
        """

        grid = coords_grid(flow)
        out = F.grid_sample(
            feat,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners)

        mask = torch.ones(feat.size(), device=feat.device, requires_grad=False)
        if self.use_mask:
            mask = F.grid_sample(
                mask,
                grid,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners)
            mask = (mask > 0.9999).float()
        return out * mask

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(mode={self.mode}, '
        s += f'padding_mode={self.padding_mode}, '
        s += f'align_corners={self.align_corners},'
        s += f'use_mask={self.use_mask})'
        return s