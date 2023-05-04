import torch
from einops import rearrange
from utils.visualization_toolsbox import PointsPainter, print_dict


def main(chosen_batch_idx: int, show_most_recent_sweep_only: bool, show_correction: bool):
    batch_dict = torch.load('artifact/dataset_trainTrue_bs10_dataloaderIdx3.pth')
    print_dict(batch_dict, name='batch_dict')
    points = batch_dict['points']  # (N, 8) - batch_idx, x, y, z, intensity, time, sweep_idx, instance_idx
    
    points = points[points[:, 0].long() == chosen_batch_idx]
    boxes = batch_dict['gt_boxes'][chosen_batch_idx]
    print('points: ', points.shape)
    
    unq_sweep_ids = torch.unique(points[:, -2].long(), sorted=True)
    print('unq_sweep_ids: ', unq_sweep_ids)

    if show_most_recent_sweep_only:
        points = points[points[:, -2].long() == unq_sweep_ids[-1].item()]
        print('points: ', points.shape)
    elif show_correction:
        instances_tf = batch_dict['instances_tf'][chosen_batch_idx]  # (N_box, N_sweep, 3, 4)
        print('instances_tf: ', instances_tf.shape)

        instances_tf = rearrange(instances_tf, 'N_box N_sweep C1 C2 -> (N_box N_sweep) C1 C2')

        mask_fg = points[:, -1] > -1
        assert torch.any(mask_fg)

        num_sweeps = 10
        points_inst_sw_idx = points[mask_fg, -1].long() * num_sweeps + points[mask_fg, -2].long()
        unique_points_inst_sw_idx, unique_points_inst_sw_idx_inv = torch.unique(points_inst_sw_idx, return_inverse=True)

        locals_tf = instances_tf[unique_points_inst_sw_idx]
        points_tf = locals_tf[unique_points_inst_sw_idx_inv]  # (N_pts, 3, 4)

        points_corr = torch.clone(points)
        points_corr[mask_fg, 1: 4] = torch.matmul(points_tf[:, :3, :3], points_corr[mask_fg, 1: 4].unsqueeze(-1)).squeeze(-1) + points_tf[:, :3, -1]
    


    # ==================
    print('showing original points')
    painter = PointsPainter(points[:, 1: 4], boxes[:, :7])
    painter.show()

    if show_correction:
        print('showing corrected points')
        painter = PointsPainter(points_corr[:, 1: 4], boxes[:, :7])
        painter.show()


if __name__ == '__main__':
    main(chosen_batch_idx=5,
         show_most_recent_sweep_only=False,
         show_correction=True)
