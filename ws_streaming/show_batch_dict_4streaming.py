import torch
import numpy as np
import subprocess
import torch_scatter
from einops import rearrange
import matplotlib.pyplot as plt

from utils.visualization_toolsbox import PointsPainter, print_dict
from warp import Warp


def main(chosen_batch_idx: int, 
         pc_range: np.ndarray, 
         copy_from_azog: bool,
         show_full: bool,
         show_stream: bool,
         show_hm: bool):
    
    files_name = ('batch_dict_4streaming_pmfusion.pth', 
                  'out_batch_dict_4streaming_chunk0_pmfusion.pth',
                  'out_batch_dict_4streaming_chunk1_pmfusion.pth', 
                  'out_batch_dict_4streaming_chunk2_pmfusion.pth')

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
        points = points[np.logical_and(points[:, 1: 4] > pc_range[:3], points[:, 1: 4] < pc_range[3:]).all(axis=1)]


        # gt_boxes = batch_dict['gt_boxes'][chosen_batch_idx]

        painter = PointsPainter(points[:, 1: 4])
        painter.show()
        print('==================================')

    if show_stream:
        for chunk_idx in (0, 1, 2):
            print(f'chunk index: {chunk_idx}')

            output = torch.load(f'./artifact/out_batch_dict_4streaming_chunk{chunk_idx}_pmfusion.pth', 
                                map_location=torch.device('cpu'))
            points_input = output['points_input']
            points_corrected = output['points_corrected']
            mask_fg = output['mask_fg']
            pts_streaming_indicator = output['pts_streaming_indicator']
            pred_dicts = output['pred_dicts']

            # TODO: take points of this batch
            points_batch_mask = points_input[:, 0].long() == chosen_batch_idx
            points_input = points_input[points_batch_mask]
            points_corrected = points_corrected[points_batch_mask]
            mask_fg = mask_fg[points_batch_mask]
            pts_streaming_indicator = pts_streaming_indicator[points_batch_mask]
            
            pred_dicts = pred_dicts[chosen_batch_idx]
            print_dict(pred_dicts, 'pred_dicts')
            
            print('show input: now')
            mask_now = pts_streaming_indicator.long() == 0
            painter = PointsPainter(points_input[mask_now, 1: 4])
            painter.show()

            print('show input: now + prev_fg')
            painter = PointsPainter(points_input[:, 1: 4])
            mask_prev_fg = torch.logical_not(mask_now)
            points_color = points_input.new_zeros(points_input.shape[0], 3)
            points_color[mask_prev_fg, 0] = 1.0
            painter.show(xyz_color=points_color)

            print('show corrected: color coded by fg')
            painter = PointsPainter(points_corrected[:, 1: 4])
            points_color = points_corrected.new_zeros(points_corrected.shape[0], 3)
            points_color[mask_fg, 0] = 1  # fg -> red
            painter.show(xyz_color=points_color)

            if chunk_idx == 2:
                print('show predict boxes')
                pred_boxes = pred_dicts['pred_boxes']
                pred_scores = pred_dicts['pred_scores']
                
                mask_valid_pred = pred_scores > 0.3
                pred_boxes = pred_boxes[mask_valid_pred]
                painter = PointsPainter(points_corrected[:, 1: 4], pred_boxes[:, :7])
                points_color = points_corrected.new_zeros(points_corrected.shape[0], 3)
                points_color[mask_fg, 0] = 1  # fg -> red
                painter.show(xyz_color=points_color)

            print('==================================')

            # break

    if show_hm:
        for chunk_idx in (0, 1, 2):
            print(f'chunk index: {chunk_idx}')

            output = torch.load(f'./artifact/out_batch_dict_4streaming_chunk{chunk_idx}_pmfusion.pth', 
                                map_location=torch.device('cpu'))
            print_dict(output, 'output')

            hm_car = output['pred_heatmaps'][0].sigmoid()
            print('hm_car: ', hm_car.shape)


            # TODO: get points, flow3d, project to BEV
            output_next = torch.load(f'./artifact/out_batch_dict_4streaming_chunk{chunk_idx + 1}_pmfusion.pth', 
                                     map_location=torch.device('cpu'))
            flow3d = output_next['points_flow3d']  # (N_next, 3)
            pts_streaming_indicator = output_next['pts_streaming_indicator']  # (N_next)

            hm_car_next = output_next['pred_heatmaps'][0].sigmoid()
            
            # extract flow of current foreground
            flow3d = flow3d[pts_streaming_indicator.long() == 1]
            assert flow3d.shape[0] == output['mask_fg'].sum().item()

            # scatter flow3d to BEV
            mask_fg = output['mask_fg']
            corrected_current_fg = output['points_corrected'][mask_fg]
            
            batch_idx = corrected_current_fg[:, 0].long()

            height, width = hm_car.shape[-2:]
            
            pc_range = torch.from_numpy(pc_range).float()
            
            bev_coord_xy_float = (corrected_current_fg[:, 1: 3] - pc_range[0]) /  (0.2 * 4)  # (N_fg, 2)
            print('bev_coord_xy_float: ', bev_coord_xy_float.shape)
            
            all_x = torch.arange(0, width)
            all_y = torch.arange(0, height)
            yy, xx = torch.meshgrid(all_y, all_x)
            mask_positive_hm = hm_car > 0.1  # (B, C, H, W)
            pos_x, pos_y = xx[mask_positive_hm[chosen_batch_idx, 0]], yy[mask_positive_hm[chosen_batch_idx, 0]]
            heat_now = hm_car[chosen_batch_idx, 0, pos_y, pos_x]

            nonzero_xy = torch.stack([pos_x, pos_y], dim=1)  # (N_pos, 2)
            print('nonzero_xy: ', nonzero_xy.shape)

            # find k-nearest neighbor w/ bev_coord_xy_float
            dist = torch.square(nonzero_xy.unsqueeze(1) - bev_coord_xy_float.unsqueeze(0)).sum(dim=-1)  
            # (N_pos, 1, 2) - (1, N_fg, 2) -> (N_pos, N_fg, 2) -> (N_pos, N_fg) 
            k = 10
            _, neighbor_indices = torch.topk(dist, k=k, dim=1, largest=False)  # (N_pos, 5) 
            print('neighbor_indices: ', neighbor_indices.shape)

            nonzero_xy_flow = flow3d[neighbor_indices.reshape(-1), :2].reshape(nonzero_xy.shape[0], k, 2).max(dim=1)[0]  # (N_pos, 2)
            nonzero_xy_flow = nonzero_xy_flow / (0.2 * 4.0)
            print('nonzero_xy_flow: ', nonzero_xy_flow.shape)

            nonzero_xy = nonzero_xy + nonzero_xy_flow
            out = []
            for i in range(4):
                warped_hm_car = torch.zeros(height, width)
                if i == 0:
                    _x = torch.floor(nonzero_xy[:, 0]) 
                    _y = torch.floor(nonzero_xy[:, 1]) 
                elif i == 1:
                    _x = torch.ceil(nonzero_xy[:, 0]) 
                    _y = torch.ceil(nonzero_xy[:, 1])
                elif i == 2:
                    _x = torch.floor(nonzero_xy[:, 0]) 
                    _y = torch.ceil(nonzero_xy[:, 1]) 
                elif i == 3:
                    _x = torch.ceil(nonzero_xy[:, 0]) 
                    _y = torch.floor(nonzero_xy[:, 1])
                
                warped_hm_car[_y.long(), _x.long()] = heat_now
                out.append(warped_hm_car.unsqueeze(0))
            
            out = torch.cat(out, dim=0)
            warped_hm_car = torch.max(out, dim=0)[0]
            warped_hm_car = warped_hm_car.unsqueeze(0).unsqueeze(0)
            
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(hm_car[chosen_batch_idx, 0, 0: 30, 45: 65])
            ax[0].set_title('hm car')

            ax[1].imshow(hm_car_next[chosen_batch_idx, 0, 0: 30, 45: 65])
            ax[1].set_title('hm car next')

            ax[2].imshow(warped_hm_car[0, 0, 0: 30, 45: 65])
            ax[2].set_title('warped hm car')


            plt.show()
            break




if __name__ == '__main__':
    main(chosen_batch_idx=1,
         pc_range=np.array([-51.2, -51.2, -5., 51.2, 51.2, 3.]),
         copy_from_azog=False, 
         show_full=False, 
         show_stream=False,
         show_hm=True)
