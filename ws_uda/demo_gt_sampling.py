import numpy as np
import torch
from nuscenes import NuScenes
import pickle

from utils.visualization_toolsbox import PointsPainter, print_dict
from nuscenes_temporal_utils import *


np.random.seed(666)


def main(num_sweeps=10):
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
    sweeps_info = get_sweeps_token(nusc, current_lidar_tk, n_sweeps=num_sweeps, return_time_lag=True, return_sweep_idx=True)
    points = list()
    list_current_se3_past = list()
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
        list_current_se3_past.append(current_se3_past)
        
    points = np.concatenate(points, axis=0)


    # sample points from database
    database_root = Path('./gt_boxes_database_lyft')
    database_car = database_root / 'car'
    car_trajs_path = list(database_car.glob('*.pkl'))
    car_trajs_path.sort()

    # filter traj by length
    invalid_traj_ids = list()
    for traj_idx, traj_path in enumerate(car_trajs_path):
        with open(traj_path, 'rb') as f:
            traj_info = pickle.load(f)
        if len(traj_info) < num_sweeps:
            invalid_traj_ids.append(traj_idx)

    invalid_traj_ids.reverse()
    for _i in invalid_traj_ids:
        del car_trajs_path[_i]

    copied_pts, copied_boxes = list(), list()
    for j in range(10):
        _pts, _boxes = load_1traj(car_trajs_path[j], list_target_se3_current_lidar=list_current_se3_past, num_sweeps=num_sweeps)
        
        copied_pts.append(_pts)
        copied_boxes.append(_boxes)

    copied_pts = np.concatenate(copied_pts)
    copied_boxes = np.concatenate(copied_boxes)

    print('copied_pts: ', copied_pts.shape)
    print('copied_boxes: ', copied_boxes.shape)

    # ======================================
    print('showing before')
    painter = PointsPainter(xyz=points[:, :3])
    painter.show()

    print('showing after')
    num_original = points.shape[0]
    points = np.concatenate([points[:, :4], copied_pts])
    points_color = np.zeros((points.shape[0], 3))
    points_color[num_original:, 0] = 1.0  # red for copied points
    painter = PointsPainter(xyz=points[:, :3], boxes=copied_boxes)
    painter.show(xyz_color=points_color)




if __name__ == '__main__':
    main()
