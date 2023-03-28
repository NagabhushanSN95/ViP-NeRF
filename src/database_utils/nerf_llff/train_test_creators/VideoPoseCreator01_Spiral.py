# Shree KRISHNAya Namaha
# Creates spiral poses
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import datetime
import time
import traceback
from pathlib import Path

import numpy
import pandas
import simplejson

this_filepath = Path(__file__)
this_filename = this_filepath.stem
this_filenum = int(this_filename[16:18])


def normalize(x):
    return x / numpy.linalg.norm(x)


def view_matrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(numpy.cross(vec1_avg, vec2))
    vec1 = normalize(numpy.cross(vec2, vec0))
    m = numpy.stack([vec0, vec1, vec2, pos], 1)
    bottom = numpy.array([0, 0, 0, 1])[None].astype('float32')
    matrix = numpy.concatenate([m, bottom], axis=0)
    return matrix


def compute_average_pose(poses: numpy.ndarray):
    # compute average pose in camera2world system
    rot_mats = poses[:, :3, :3]
    rot_inverted = numpy.transpose(rot_mats, axes=[0, 2, 1])
    translations = poses[:, :3, 3:]
    rotated_translations = -rot_inverted @ translations
    avg_translation = numpy.mean(rotated_translations, axis=0)[:, 0]

    vec2 = normalize(rot_inverted[:, :3, 2].sum(0))
    up = rot_inverted[:, :3, 1].sum(0)
    avg_pose_c2w = view_matrix(vec2, up, avg_translation)
    avg_pose = numpy.linalg.inv(avg_pose_c2w)  # convert avg_pose to world2camera system
    return avg_pose


def change_coordinate_system(poses: numpy.ndarray, p: numpy.ndarray):
    changed_poses = []
    for pose in poses:
        r = pose[:3, :3]
        t = pose[:3, 3:]
        rc = p.T @ r @ p
        tc = p @ t
        changed_pose = numpy.concatenate([numpy.concatenate([rc, tc], axis=1), pose[3:]], axis=0)
        changed_poses.append(changed_pose)
    changed_poses = numpy.stack(changed_poses)
    return changed_poses


def recenter_poses(poses):
    avg_pose = compute_average_pose(poses)
    centered_poses = avg_pose[None] @ numpy.linalg.inv(poses)

    # Convert from Colmap/RE10K convention to NeRF convention: (x,-y,-z) to (x,y,z)
    perm_matrix = numpy.eye(3)
    perm_matrix[1, 1] = -1
    perm_matrix[2, 2] = -1
    std_poses = change_coordinate_system(centered_poses, perm_matrix)

    return std_poses, avg_pose


def recenter_poses_nerf(poses):
    import numpy as np

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = view_matrix(vec2, up, center)
    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = numpy.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in numpy.linspace(0., 2. * numpy.pi * rots, N + 1)[:-1]:
        c = numpy.dot(c2w[:3, :4], numpy.array([numpy.cos(theta), -numpy.sin(theta), -numpy.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - numpy.dot(c2w[:3, :4], numpy.array([0, 0, -focal, 1.])))
        render_poses.append(numpy.concatenate([view_matrix(z, up, c), hwf], 1))
    return render_poses


def create_video_poses(poses: numpy.ndarray, num_frames: int, num_rotations: int, bds, bd_factor):
    _, avg_pose = recenter_poses(poses.copy())

    # Implement pre-processing done by NeRF on poses
    c2w_mats = numpy.linalg.inv(poses)
    poses = c2w_mats[:, :3, :4].transpose([1,2,0]).astype('float32')
    poses = numpy.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :]], 1)
    # poses = numpy.transpose(poses, [2, 0, 1])
    poses = numpy.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # [-u, r, -t] -> [r, u, -t]
    poses = numpy.moveaxis(poses, -1, 0).astype(numpy.float32)

    sc = 1. if bd_factor is None else 1. / (float(bds.min()) * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    poses = recenter_poses_nerf(poses)
    poses = poses.astype('float32')

    c2w = poses_avg(poses)
    # print('recentered', c2w.shape)
    # print(c2w[:3, :4])

    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = numpy.percentile(numpy.abs(tt), 90, 0)
    c2w_path = c2w

    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=num_rotations, N=num_frames)
    render_poses = numpy.stack(render_poses).astype('float32')

    # Invert the processing that will be done during testing
    perm_matrix = numpy.eye(3)
    perm_matrix[1:] *= -1
    cv_poses = change_coordinate_system(render_poses, perm_matrix)

    video_poses = numpy.linalg.inv(numpy.linalg.inv(avg_pose)[None] @ cv_poses)
    video_poses[:, :3, 3] /= sc

    center_pose = poses_avg(video_poses)
    video_poses = [center_pose] + video_poses.tolist()
    video_poses = numpy.stack(video_poses)
    return video_poses


def save_configs(output_dirpath: Path, configs: dict):
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = simplejson.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs != old_configs:
            raise RuntimeError('Configs mismatch while resuming resuming video pose generation.')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def save_video_poses(configs: dict):
    root_dirpath = Path('../../../../')
    database_dirpath = root_dirpath / 'data/databases/NeRF_LLFF/data/'

    set_num = configs['set_num']
    num_frames = configs['num_frames']
    num_rotations = configs['num_rotations']
    bd_factor = configs['bd_factor']

    output_dirpath = database_dirpath / f'train_test_sets/set{set_num:02}/video_poses{this_filenum:02}'
    output_dirpath.mkdir(parents=True, exist_ok=False)
    save_configs(output_dirpath, configs)

    train_videos_path = database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
    train_videos_data = pandas.read_csv(train_videos_path)

    scene_names = numpy.unique(train_videos_data['scene_name'])
    for scene_name in scene_names:
        trans_mats_path = database_dirpath / f'all/database_data/{scene_name}/CameraExtrinsics.csv'
        trans_mats = numpy.loadtxt(trans_mats_path.as_posix(), delimiter=',').reshape((-1, 4, 4))

        bds_path = database_dirpath / f'all/database_data/{scene_name}/DepthBounds.csv'
        bds = numpy.loadtxt(bds_path.as_posix(), delimiter=',')

        video_poses = create_video_poses(trans_mats, num_frames, num_rotations, bds, bd_factor)
        video_poses_flat = numpy.reshape(video_poses, (video_poses.shape[0], -1))

        output_path = output_dirpath / f'{scene_name}.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        numpy.savetxt(output_path.as_posix(), video_poses_flat, delimiter=',')
    video_frame_nums = numpy.arange(num_frames)
    output_path = output_dirpath / 'VideoFrameNums.csv'
    numpy.savetxt(output_path.as_posix(), video_frame_nums, fmt='%i', delimiter=',')
    return


def demo1():
    configs = {
        'PosesCreator': this_filename,
        'set_num': 1,
        'num_frames': 120,
        'bd_factor': 0.75,
        'num_rotations': 2,
    }
    save_video_poses(configs)

    configs = {
        'PosesCreator': this_filename,
        'set_num': 2,
        'num_frames': 120,
        'bd_factor': 0.75,
        'num_rotations': 2,
    }
    save_video_poses(configs)

    configs = {
        'PosesCreator': this_filename,
        'set_num': 3,
        'num_frames': 120,
        'bd_factor': 0.75,
        'num_rotations': 2,
    }
    save_video_poses(configs)

    configs = {
        'PosesCreator': this_filename,
        'set_num': 4,
        'num_frames': 120,
        'bd_factor': 0.75,
        'num_rotations': 2,
    }
    save_video_poses(configs)
    return


def main():
    demo1()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
