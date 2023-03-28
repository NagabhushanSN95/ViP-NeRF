# Shree KRISHNAya Namaha
# Creates poses using original video trajectory
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


def create_video_poses(trans_mats: numpy.ndarray):
    num_frames = trans_mats.shape[0]
    video_poses = list(trans_mats)

    # Add center frame pose at the beginning
    center_pose = video_poses[num_frames // 2]
    video_poses = [center_pose] + video_poses
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
            from deepdiff import DeepDiff
            raise RuntimeError(f'Configs mismatch while resuming testing: {DeepDiff(old_configs, configs)}')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def save_video_poses(configs: dict):
    root_dirpath = Path('../../../../')
    database_dirpath = root_dirpath / 'data/databases/RealEstate10K/data/'

    set_num = configs['set_num']

    output_dirpath = database_dirpath / f'train_test_sets/set{set_num:02}/video_poses{this_filenum:02}'
    output_dirpath.mkdir(parents=True, exist_ok=False)
    save_configs(output_dirpath, configs)

    train_videos_path = database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
    train_videos_data = pandas.read_csv(train_videos_path)

    scene_nums = numpy.unique(train_videos_data['scene_num'])
    for scene_num in scene_nums:
        trans_mats_path = root_dirpath / f'data/test/database_data/{scene_num:05}/CameraExtrinsics.csv'
        trans_mats = numpy.loadtxt(trans_mats_path.as_posix(), delimiter=',').reshape((-1, 4, 4))
        video_poses = create_video_poses(trans_mats)
        video_poses_flat = numpy.reshape(video_poses, (video_poses.shape[0], -1))

        output_path = output_dirpath / f'{scene_num:05}.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        numpy.savetxt(output_path.as_posix(), video_poses_flat, delimiter=',')
    video_frame_nums = numpy.concatenate([numpy.arange(0, 50, 1), numpy.arange(49, -1, -1)])[None]
    output_path = output_dirpath / 'VideoFrameNums.csv'
    numpy.savetxt(output_path.as_posix(), video_frame_nums, fmt='%i', delimiter=',')
    return


def demo1():
    configs = {
        'PosesCreator': this_filename,
        'set_num': 1,
    }
    save_video_poses(configs)

    configs = {
        'PosesCreator': this_filename,
        'set_num': 2,
    }
    save_video_poses(configs)

    configs = {
        'PosesCreator': this_filename,
        'set_num': 3,
    }
    save_video_poses(configs)

    configs = {
        'PosesCreator': this_filename,
        'set_num': 4,
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
