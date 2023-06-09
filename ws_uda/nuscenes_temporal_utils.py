import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion
from pathlib import Path
from typing import List, Dict, Union
import pickle


map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


def tf(translation, rotation):
    """
    Build transformation matrix
    """
    if not isinstance(rotation, Quaternion):
        assert isinstance(rotation, list) or isinstance(rotation, np.ndarray), f"{type(rotation)} is not supported"
        rotation = Quaternion(rotation)
    tf_mat = np.eye(4)
    tf_mat[:3, :3] = rotation.rotation_matrix
    tf_mat[:3, 3] = translation
    return tf_mat


def apply_tf(tf: np.ndarray, points: np.ndarray):
    assert points.shape[1] == 3
    assert tf.shape == (4, 4)
    points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    out = tf @ points_homo.T
    return out[:3, :].T


def rotation_matrix_to_yaw(rot: np.ndarray) -> float:
    return np.arctan2(rot[1, 0], rot[0, 0])


def apply_se3_(se3_tf: np.ndarray, 
               points_: np.ndarray = None, 
               boxes_: np.ndarray = None, boxes_has_velocity: bool = False, 
               vector_: np.ndarray = None) -> None:
    """
    Inplace function

    Args:
        se3_tf: (4, 4) - homogeneous transformation
        points_: (N, 3 + C) - x, y, z, [others]
        boxes_: (N, 7 + 2 + C) - x, y, z, dx, dy, dz, yaw, [vx, vy], [others]
        boxes_has_velocity: make boxes_velocity explicit
        vector_: (N, 2 [+ 1]) - x, y, [z]
    """
    if points_ is not None:
        assert points_.shape[1] >= 3, f"points_.shape: {points_.shape}"
        points_[:, :3] = points_[:, :3] @  se3_tf[:3, :3].T + se3_tf[:3, -1]

    if boxes_ is not None:
        assert boxes_.shape[1] >= 7, f"boxes_.shape: {boxes_.shape}"
        boxes_[:, :3] = boxes_[:, :3] @  se3_tf[:3, :3].T + se3_tf[:3, -1]
        boxes_[:, 6] += rotation_matrix_to_yaw(se3_tf[:3, :3])
        boxes_[:, 6] = np.arctan2(np.sin(boxes_[:, 6]), np.cos(boxes_[:, 6]))
        if boxes_has_velocity:
            boxes_velo = np.pad(boxes_[:, 7: 9], pad_width=[(0, 0), (0, 1)], constant_values=0.0)  # (N, 3) - vx, vy, vz
            boxes_velo = boxes_velo @ se3_tf[:3, :3].T
            boxes_[:, 7: 9] = boxes_velo[:, :2]

    if vector_ is not None:
        if vector_.shape[1] == 2:
            vector = np.pad(vector_, pad_width=[(0, 0), (0, 1)], constant_values=0.)
            vector_[:, :2] = (vector @ se3_tf[:3, :3].T)[:, :2]
        else:
            assert vector_.shape[1] == 3, f"vector_.shape: {vector_.shape}"
            vector_[:, :3] = vector_ @ se3_tf[:3, :3].T

    return


def get_nuscenes_sensor_pose_in_ego_vehicle(nusc: NuScenes, curr_sd_token: str):
    curr_rec = nusc.get('sample_data', curr_sd_token)
    curr_cs_rec = nusc.get('calibrated_sensor', curr_rec['calibrated_sensor_token'])
    ego_from_curr = tf(curr_cs_rec['translation'], curr_cs_rec['rotation'])
    return ego_from_curr


def get_nuscenes_sensor_pose_in_global(nusc: NuScenes, curr_sd_token: str):
    ego_from_curr = get_nuscenes_sensor_pose_in_ego_vehicle(nusc, curr_sd_token)
    curr_rec = nusc.get('sample_data', curr_sd_token)
    curr_ego_rec = nusc.get('ego_pose', curr_rec['ego_pose_token'])
    glob_from_ego = tf(curr_ego_rec['translation'], curr_ego_rec['rotation'])
    glob_from_curr = glob_from_ego @ ego_from_curr
    return glob_from_curr


def get_sweeps_token(nusc: NuScenes, curr_sd_token: str, n_sweeps: int, return_time_lag=True, return_sweep_idx=False) -> list:
    ref_sd_rec = nusc.get('sample_data', curr_sd_token)
    ref_time = ref_sd_rec['timestamp'] * 1e-6
    sd_tokens_times = []
    for s_idx in range(n_sweeps):
        curr_sd = nusc.get('sample_data', curr_sd_token)
        if not return_sweep_idx:
            sd_tokens_times.append((curr_sd_token, ref_time - curr_sd['timestamp'] * 1e-6))
        else:
            sd_tokens_times.append((curr_sd_token, ref_time - curr_sd['timestamp'] * 1e-6, n_sweeps - 1 - s_idx))
        # s_idx: the higher, the closer to the current
        # move to previous
        if curr_sd['prev'] != '':
            curr_sd_token = curr_sd['prev']

    # organize from PAST to PRESENCE
    sd_tokens_times.reverse()

    if return_time_lag:
        return sd_tokens_times
    else:
        return [token for token, _ in sd_tokens_times]


def get_one_pointcloud(nusc: NuScenes, sweep_token: str) -> np.ndarray:
    """
    Args:
        nusc:
        sweep_token: sample data token

    Return:
        pointcloud: (N, 4) - x, y, z, reflectant
    """
    pcfile = nusc.get_sample_data_path(sweep_token)  # TODO: bug here
    pc = np.fromfile(pcfile, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]  # (x, y, z, intensity)
    return pc


def get_available_scenes(nusc: NuScenes) -> List[Dict]:
    available_scenes = []
    print('total scene num:', len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            # if not sd_rec['next'] == '':
            #     sd_rec = nusc.get('sample_data', sd_rec['next'])
            # else:
            #     has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes


def quaternion_to_yaw(q: Quaternion) -> float:
    return np.arctan2(q.rotation_matrix[1, 0], q.rotation_matrix[0, 0])


def make_rotation_around_z(yaw: float) -> np.ndarray:
    cos, sin = np.cos(yaw), np.sin(yaw)
    out = np.array([
        [cos, -sin, 0.],
        [sin, cos, 0.],
        [0., 0., 1.]
    ])
    return out


def make_se3(translation: Union[List[float], np.ndarray], yaw: float = None, rotation_matrix: np.ndarray = None):
    if yaw is None:
        assert rotation_matrix is not None
    else:
        assert rotation_matrix is None
    
    if rotation_matrix is None:
        rotation_matrix = make_rotation_around_z(yaw)

    out = np.zeros((4, 4))
    out[-1, -1] = 1.0

    out[:3, :3] = rotation_matrix

    if not isinstance(translation, np.ndarray):
        translation = np.array(translation)
    out[:3, -1] = translation.reshape(3)

    return out


def load_1traj(path_traj: Path, num_sweeps: int = 10, beam_ratio: int = 2):
    with open(path_traj, 'rb') as f:
        traj_info = pickle.load(f)
    traj_len = len(traj_info)
    
    start_idx = np.random.randint(low=0, high=max(traj_len - num_sweeps, 0))
    end_idx = min(start_idx + num_sweeps, traj_len)
    
    points, boxes, mask_keep_points = list(), list(), list()
    for idx in range(start_idx, end_idx):
        info = traj_info[idx]
        
        box_in_glob = info['box_in_glob']  # in glob
        
        # ----
        # points | in box -> in glob
        pts = info['points']  # in box
        glob_se3_box = make_se3(box_in_glob[:3], yaw=box_in_glob[6])
        apply_se3_(glob_se3_box, points_=pts)

        # ---
        # downsample based on points' beam idx
        if 'points_beam_idx' in info:
            points_beam_idx = info['points_beam_idx']
            mask_keep = (points_beam_idx % beam_ratio) == 0  # (N,)
        else:
            mask_keep = np.ones(pts.shape[0], dtype=bool)

        points.append(pts)
        boxes.append(box_in_glob.reshape(1, -1))
        mask_keep_points.append(mask_keep)

    points = np.concatenate(points, axis=0)  # in glob
    boxes = np.concatenate(boxes, axis=0)  # in glob
    mask_keep_points = np.concatenate(mask_keep_points)

    glob_se3_last_box = make_se3(boxes[-1, :3], yaw=boxes[-1, 6])
    # map points and boxes to last_box
    apply_se3_(np.linalg.inv(glob_se3_last_box), points_=points, boxes_=boxes)
    
    # map points and boxes from last_box to lidar
    last_box_in_lidar = traj_info[-1]['box_in_lidar']
    lidar_se3_last_box = make_se3(last_box_in_lidar[:3], yaw=last_box_in_lidar[6])
    apply_se3_(lidar_se3_last_box, points_=points, boxes_=boxes)

    return points, boxes, mask_keep_points
