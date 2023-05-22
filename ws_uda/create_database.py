import numpy as np
from pyquaternion import Quaternion
from pathlib import Path
from typing import List, Dict
import pickle
from pprint import pprint
from nuscenes import NuScenes
from nuscenes_temporal_utils import get_available_scenes, quaternion_to_yaw, get_one_pointcloud, get_nuscenes_sensor_pose_in_global, \
    apply_se3_, make_se3, map_name_from_general_to_detection


DATABASE_ROOT = Path('./gt_boxes_database')


def process_1scene(nusc: NuScenes, scene_token: str, classes_of_interest=set(('car', 'pedestrian', 'bicycle')), database_root=DATABASE_ROOT):
    scene = nusc.get('scene', scene_token)
    seen_instances_token = set()
    sample_tk = scene['first_sample_token']
    while sample_tk != '':
        sample = nusc.get('sample', sample_tk)
        for anno_tk in sample['anns']:
            sample_anno = nusc.get('sample_annotation', anno_tk)
            # det_name = map_name_from_general_to_detection[sample_anno['category_name']]
            det_name = sample_anno['category_name']
            # Filter 
            # by name
            if det_name not in classes_of_interest:
                continue
            # by num lidar 
            # if sample_anno['num_lidar_pts'] < 5:
            #     continue
            # by instance_token
            if sample_anno['instance_token'] in seen_instances_token:
                continue

            # get points on this traj & save to disk
            seen_instances_token.add(sample_anno['instance_token'])
            traj_info = get_points_on_trajectory(nusc, sample_anno['instance_token'])
            with open(database_root / f"{det_name}" / f"{sample_anno['instance_token']}.pkl", 'wb') as f:
                pickle.dump(traj_info, f)

        # move to next
        sample_tk = sample['next']

    return


def get_points_on_trajectory(nusc: NuScenes, instance_token: str) -> List[Dict]:
    """
    out = [
        {'sample_token': '',
         'points': np.array([])},

        {'sample_token': '',
         'points': np.array([])}
    ]
    """
    instance = nusc.get('instance', instance_token)
    anno_tk = instance['first_annotation_token']
    out = list()
    while anno_tk != '':
        sample_anno = nusc.get('sample_annotation', anno_tk)
        sample = nusc.get('sample', sample_anno['sample_token'])

        # box
        box = np.array([
            *sample_anno['translation'],
            sample_anno['size'][1], sample_anno['size'][0], sample_anno['size'][2],
            quaternion_to_yaw(Quaternion(sample_anno['rotation']))
        ])  # [x, y, z, dx, dy, dz, yaw]
        glob_se3_box = make_se3(box[:3], yaw=box[6])
        lidar_se3_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(nusc, sample['data']['LIDAR_TOP']))
        box_se3_lidar = np.linalg.inv(lidar_se3_glob @ glob_se3_box)
        
        # ---------------
        # get points in box
        pcd = get_one_pointcloud(nusc, sample['data']['LIDAR_TOP'])  # (N, 4) in LiDAR
        apply_se3_(box_se3_lidar, points_=pcd)
        mask_in_box = np.all(np.abs(pcd[:, :3] / box[3: 6]) < (0.5 + 1e-2), axis=1)  # (N,)
        
        # store points & box
        out.append({
            'sample_tk': sample_anno['sample_token'],
            'points': pcd[mask_in_box],  # (N, 4) - x, y, z, intensity in LiDAR
            'box': box  # (7,) - [x, y, z, dx, dy, dz, yaw]
        })

        # move to next
        anno_tk = sample_anno['next']
    
    return out


if __name__ == '__main__':
    from lyft_dataset_sdk.lyftdataset import LyftDataset

    lyft = LyftDataset(data_path='/home/user/dataset/lyft/trainval/', 
                       json_path='/home/user/dataset/lyft/trainval/data/', 
                       verbose=True)
    
    scene = lyft.scene[0]

    # nusc = NuScenes(dataroot='/home/user/dataset/nuscenes/', verbose=False, version='v1.0-mini')
    # scene = nusc.scene[0]
    process_1scene(lyft, scene['token'])
    # sample_tk = scene['first_sample_token']
    # sample = lyft.get('sample', sample_tk)
    # sample_anno = lyft.get('sample_annotation', sample['anns'][0])
    # pprint(sample_anno)
