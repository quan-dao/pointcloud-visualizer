import numpy as np
from nuscenes import NuScenes 
from pprint import pprint
from pyquaternion import Quaternion
from nuscenes_temporal_utils import *

from utils.visualization_toolsbox import PointsPainter


def main():
    nusc = NuScenes(dataroot='/home/user/dataset/nuscenes/', verbose=False, version='v1.0-mini')
    scene = nusc.scene[0]

    sample_tk = scene['first_sample_token']
    sample = nusc.get('sample', sample_tk)
    box = None
    for ann_tk in sample['anns']:
        sample_anno = nusc.get('sample_annotation', ann_tk)
        det_name = map_name_from_general_to_detection[sample_anno['category_name']]
        if det_name == 'truck':
            pprint(sample_anno)
            box = np.array([
                *sample_anno['translation'],
                sample_anno['size'][1], sample_anno['size'][0], sample_anno['size'][2],
                quaternion_to_yaw(Quaternion(sample_anno['rotation']))
            ]).reshape(1, -1)  # [x, y, z, dx, dy, dz, yaw] | in global
            lidar_se3_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(nusc, sample['data']['LIDAR_TOP']))
            apply_se3_(lidar_se3_glob, boxes_=box)
            box = box.reshape(-1)
            break
    


    # find points in box
    pcd = get_one_pointcloud(nusc, sample['data']['LIDAR_TOP'])  # (N, 4) in LiDAR

    painter = PointsPainter(xyz=pcd[:, :3], boxes=box.reshape(1, -1))
    painter.show()

    lidar_se3_box = make_se3(box[:3], yaw=box[6])
    box_se3_lidar = np.linalg.inv(lidar_se3_box)
    print('box_se3_lidar:\n', box_se3_lidar)
    # return
    apply_se3_(box_se3_lidar, points_=pcd)
    
    painter = PointsPainter(xyz=pcd[:, :3])
    painter.show()

    mask_in_box = np.all(np.abs(pcd[:, :3] / box[3: 6]) < (0.5 + 2e-2), axis=1)  # (N,)
    print('num_in_box: ', mask_in_box.sum())


if __name__ == '__main__':
    main()
