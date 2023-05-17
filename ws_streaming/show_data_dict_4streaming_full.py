import torch
import numpy as np
import subprocess

from utils.visualization_toolsbox import PointsPainter, print_dict


def main(chosen_batch_idx: int, 
         pc_range: np.ndarray, 
         show_full: bool,
         show_data_dict: bool, 
         copy_from_azog: bool):
    files_name = ('batch_dict_4streaming_full.pth', 'data_dict_4streaming_full.pth')
    if copy_from_azog:
        domain = 'jupyter-dao-mq@azog.ls2n.ec-nantes.fr'
        root_at_domain = '/home/jupyter-dao-mq/workspace/learn-to-align-bev/streaming_ws/artifact'
        for filename in files_name:
            src_file = f'{domain}:{root_at_domain}/{filename}'
            cmd_out = subprocess.run(['scp', src_file, './artifact/'], stdout=subprocess.PIPE)
    
    if show_full:
        print('showing full 30 sweeps')
        batch_dict = torch.load(f'./artifact/{files_name[0]}')
        print_dict(batch_dict, 'batch_dict')
        assert chosen_batch_idx < batch_dict['batch_size']
        points = batch_dict['points']
        points = points[points[:, 0].astype(int) == chosen_batch_idx]
        print('points.shape: ', points.shape)
        points = points[np.logical_and(points[:, 1: 4] > pc_range[:3], points[:, 1: 4] < pc_range[3:]).all(axis=1)]


        # gt_boxes = batch_dict['gt_boxes'][chosen_batch_idx]

        painter = PointsPainter(points[:, 1: 4])
        painter.show()
        print('==================================')

    if show_data_dict:
        print('showing data dict')
        data_dict = torch.load(f'./artifact/{files_name[1]}', map_location=torch.device('cpu'))
        print_dict(data_dict, 'data_dict')

        points = data_dict['points'].detach()
        pts_all_cls_prob = torch.sigmoid(data_dict['points_cls_logit'].detach())  # (N, 3)
        points_cls_prob, points_cls_indices = torch.max(pts_all_cls_prob, dim=1)  # (N,), (N,)
        mask_bg = torch.logical_and(points_cls_prob > 0.3, points_cls_indices == 0)
        mask_fg = torch.logical_not(mask_bg)

        mask_current_sample = points[:, 0].long() == chosen_batch_idx
        points = points[mask_current_sample]
        mask_fg = mask_fg[mask_current_sample]

        print('points.shape: ', points.shape)


        painter = PointsPainter(points[:, 1: 4])
        points_color = points.new_zeros(points.shape[0], 3)
        points_color[mask_fg, 0] = 1.0
        painter.show(xyz_color=points_color)
        print('==================================')



if __name__ == '__main__':
    main(chosen_batch_idx=0,
         pc_range=np.array([-51.2, -51.2, -5., 51.2, 51.2, 3.]),
         show_full=True,
         show_data_dict=True,
         copy_from_azog=False)


