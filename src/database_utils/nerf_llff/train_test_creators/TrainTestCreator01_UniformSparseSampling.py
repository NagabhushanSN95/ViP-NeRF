# Shree KRISHNAya Namaha
# Uses every 8th frame as the test frame as per the usual convention. Among the remaining frames, divides them into
# n+1 groups and picks n frames at the intersections
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


def sample_sparse_train_frames(frame_nums: list, num_frames: int):
    if num_frames != -1:
        selected_indices = numpy.round(numpy.linspace(-1, len(frame_nums), num_frames+2)).astype('int')[1:-1]
        selected_frame_nums = numpy.array(frame_nums)[selected_indices]
    else:
        selected_frame_nums = frame_nums
    return selected_frame_nums


def create_scene_frames_data(scene_name, frame_nums):
    frames_data = [[scene_name, frame_num] for frame_num in frame_nums]
    return frames_data


def create_data_frame(frames_data: list):
    frames_array = numpy.array(frames_data)
    frames_data = pandas.DataFrame(frames_array, columns=['scene_name', 'pred_frame_num'])
    return frames_data


def create_train_test_set(configs: dict):
    root_dirpath = Path('../../../../')
    database_dirpath = root_dirpath / 'data/databases/NeRF_LLFF/data/'

    set_num = configs['set_num']
    num_train_frames = configs['num_train_frames']

    set_dirpath = database_dirpath / f'train_test_sets/set{set_num:02}'
    set_dirpath.mkdir(parents=True, exist_ok=True)

    scenes_dirpath = database_dirpath / 'all/database_data/'
    scene_names = sorted(map(lambda path: path.stem, scenes_dirpath.iterdir()))
    train_data, val_data, test_data = [], [], []

    for scene_name in scene_names:
        frames_dirpath = scenes_dirpath / f'{scene_name}/rgb'
        frame_nums = sorted(map(lambda path: int(path.stem), frames_dirpath.iterdir()))
        test_frame_nums = list(range(0, len(frame_nums), 8))
        train_frame_nums = list(set(frame_nums) - set(test_frame_nums))
        train_frame_nums = sample_sparse_train_frames(train_frame_nums, num_train_frames)
        val_frame_nums = [test_frame_nums[len(test_frame_nums) // 2]]
        train_data.extend(create_scene_frames_data(scene_name, train_frame_nums))
        test_data.extend(create_scene_frames_data(scene_name, test_frame_nums))
        val_data.extend(create_scene_frames_data(scene_name, val_frame_nums))
    train_data = create_data_frame(train_data)
    train_data_path = set_dirpath / 'TrainVideosData.csv'
    train_data.to_csv(train_data_path, index=False)

    test_data = create_data_frame(test_data)
    test_data_path = set_dirpath / 'TestVideosData.csv'
    test_data.to_csv(test_data_path, index=False)

    val_data = create_data_frame(val_data)
    val_data_path = set_dirpath / 'ValidationVideosData.csv'
    val_data.to_csv(val_data_path, index=False)

    configs_path = set_dirpath / 'Configs.json'
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)

    return


def demo1():
    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 1,
        'num_train_frames': -1,
    }
    create_train_test_set(configs)

    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 2,
        'num_train_frames': 2,
    }
    create_train_test_set(configs)

    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 3,
        'num_train_frames': 3,
    }
    create_train_test_set(configs)

    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 4,
        'num_train_frames': 4,
    }
    create_train_test_set(configs)
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
