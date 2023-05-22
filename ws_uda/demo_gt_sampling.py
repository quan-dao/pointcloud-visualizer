import numpy as np
import torch
from nuscenes import NuScenes

from utils.visualization_toolsbox import PointsPainter, print_dict
from nuscenes_temporal_utils import *


def main():
    nusc = NuScenes(dataroot='/home/user/dataset/nuscenes/', verbose=False, version='v1.0-mini')

    scene = nusc.scene[0]
    sample_tk = scene['first_sample_token']
    for _ in range(3):
        sample = nusc.get('sample', sample_tk)
        sample_tk = sample['next']
    
    # get token of the current LiDAR
    current_sample_tk = sample_tk
    current_sample = nusc.get('sample', current_sample_tk)
    current_lidar_tk = current_sample['data']['LIDAR_TOP']
    current_se3_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(nusc, current_lidar_tk))
    
    # get pointcloud
    sweeps_info = get_sweeps_token(nusc, current_lidar_tk, n_sweeps=10, return_time_lag=True, return_sweep_idx=True)
    points = list()
    for (lidar_tk, timelag, sweep_idx) in sweeps_info:
        pcd = get_one_pointcloud(nusc, lidar_tk)
        pcd = np.pad(pcd, pad_width=[(0, 0), (0, 2)], constant_values=-1)
        pcd[:, -2] = timelag
        pcd[:, -1] = sweep_idx

        # map pcd to current frame (EMC)
        glob_se3_past =  get_nuscenes_sensor_pose_in_global(nusc, lidar_tk)
        current_se3_past = current_se3_glob @ glob_se3_past
        apply_se3_(current_se3_past, points_=pcd)
        points.append(pcd)
        
    points = np.concatenate(points, axis=0)

    # ======================================
    painter = PointsPainter(xyz=points[:, :3])
    painter.show()




if __name__ == '__main__':
    main()
