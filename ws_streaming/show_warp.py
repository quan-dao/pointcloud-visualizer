import torch
import numpy as np
import subprocess
import matplotlib.pyplot as plt

from utils.visualization_toolsbox import PointsPainter, print_dict
from warp_heatmap import warp


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

        painter = PointsPainter(points[:, 1: 4])
        painter.show()
        print('==================================')

    if show_hm:
        historical_hms = []
        for chunk_idx in range(3):
            print(f'chunk index: {chunk_idx}')
            out_current = torch.load(f'./artifact/out_batch_dict_4streaming_chunk{chunk_idx}_pmfusion.pth', 
                                     map_location=torch.device('cpu'))
            out_next = torch.load(f'./artifact/out_batch_dict_4streaming_chunk{chunk_idx + 1}_pmfusion.pth', 
                                  map_location=torch.device('cpu')) if chunk_idx < 2 else None
            
            current_hm = out_current['pred_heatmaps'][0].sigmoid()
            historical_hms.append(current_hm)
            
            if out_next is None:
                break

            # find current foreground
            fg_current = out_current['points_corrected'][out_current['mask_fg']]  # (N_fg_curr, 1 + 3 + C)
            
            # find scene flow of current foreground
            flow3d = out_next['points_flow3d']  # (N_next, 3)
            stream_indicator = out_next['pts_streaming_indicator']  # (N_next,)  
            # sanity check
            fg_current_flow3d = flow3d[stream_indicator.long() == 1]  # (N_fg_curr, 3)
            assert fg_current_flow3d.shape[0] == fg_current.shape[0]

            # warp
            for hm_idx in range(len(historical_hms)):
                flow_discount_factor = np.exp(-(len(historical_hms) - 1.0 - hm_idx) * 0.).item()
                print('flow_discount_factor: ', flow_discount_factor)
                historical_hms[hm_idx] = warp(historical_hms[hm_idx], 
                                              fg_current, 
                                              fg_current_flow3d * flow_discount_factor, 
                                              pixel_size=0.2 * 4,
                                              point_cloud_range=pc_range,
                                              num_neighbors=10)

            # fig, ax = plt.subplots(1, len(historical_hms) + 1)
            # for hm_idx, hm in enumerate(historical_hms):
            #     ax[hm_idx].imshow(hm[0, 0]), ax[hm_idx].set_title(f'{hm_idx}'), ax[hm_idx].set_aspect('equal')

            # next_hm = out_next['pred_heatmaps'][0].sigmoid()
            # ax[-1].imshow(next_hm[0, 0]), ax[-1].set_title('target hm'), ax[-1].set_aspect('equal')
            
            # plt.show()

            print(f'end of chunk: {chunk_idx}')
            print('---')
            # break
        
        # =====================
        fused_hm = torch.zeros_like(historical_hms[0])
        discount = np.exp(-(len(historical_hms) - 1.0 - np.arange(len(historical_hms))) * 2.0) 
        discount /= discount.sum()
        for hm_idx, (hm, dis_factor) in enumerate(zip(historical_hms, discount)):
            fused_hm += hm * dis_factor


        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(fused_hm[0, 0]), ax[0].set_title('fused'), ax[0].set_aspect('equal')

        out_final = torch.load('./artifact/out_batch_dict_4streaming_chunk2_pmfusion.pth', 
                               map_location=torch.device('cpu'))
        final_hm = out_final['pred_heatmaps'][0].sigmoid()
        ax[1].imshow(final_hm[0, 0]), ax[1].set_title('final'), ax[1].set_aspect('equal')
        plt.show()


if __name__ == '__main__':
    main(
        chosen_batch_idx=1,
        pc_range=np.array([-51.2, -51.2, -5., 51.2, 51.2, 3.]),
        copy_from_azog=False, 
        show_full=False, 
        show_stream=False,
        show_hm=True
    )
