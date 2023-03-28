# Shree KRISHNAya Namaha
# Creates train and test sets.
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import datetime
import time
import traceback
from pathlib import Path
from typing import List

import numpy
import pandas
import simplejson

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def create_frames_data(scene_nums: List[int], frame_nums: List[int]):
    num_scenes = len(scene_nums)
    num_frames = len(frame_nums)

    scene_nums1 = numpy.repeat(numpy.array(scene_nums)[:, None], repeats=num_frames, axis=1).ravel()
    frame_nums1 = numpy.repeat(numpy.array(frame_nums)[None, :], repeats=num_scenes, axis=0).ravel()
    frames_array = numpy.stack([scene_nums1, frame_nums1], axis=1)
    frames_data = pandas.DataFrame(frames_array, columns=['scene_num', 'pred_frame_num'])
    return frames_data


def create_train_test_set(configs: dict):
    root_dirpath = Path('../../../../')
    database_dirpath = root_dirpath / 'data/databases/RealEstate10K/data/'

    set_num = configs['set_num']
    scene_nums = configs['scene_nums']

    train_views_density = configs['train_views_density']
    if train_views_density == 'sparse':
        train_frame_nums = [10, 20, 30, 0, 40]
        test_frame_nums = list(set(range(50)) - set(train_frame_nums))
        train_frame_nums = sorted(train_frame_nums[:configs['num_train_frames']])
    elif train_views_density == 'dense':
        test_frame_nums = list(range(0, 50, 5))
        train_frame_nums = list(set(range(50)) - set(test_frame_nums))
    else:
        raise RuntimeError(f'Unknown train views density: {train_views_density}')
    validation_frame_nums = test_frame_nums[::len(test_frame_nums)//5][1:4]

    set_dirpath = database_dirpath / f'train_test_sets/set{set_num:02}'
    set_dirpath.mkdir(parents=True, exist_ok=True)

    train_data = create_frames_data(scene_nums, train_frame_nums)
    train_data_path = set_dirpath / 'TrainVideosData.csv'
    train_data.to_csv(train_data_path, index=False)

    test_data = create_frames_data(scene_nums, test_frame_nums)
    test_data_path = set_dirpath / 'TestVideosData.csv'
    test_data.to_csv(test_data_path, index=False)

    val_data = create_frames_data(scene_nums, validation_frame_nums)
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
        'scene_nums': [0, 1, 3, 4, 6],
        'train_views_density': 'dense',
    }
    create_train_test_set(configs)

    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 2,
        'scene_nums': [0, 1, 3, 4, 6],
        'train_views_density': 'sparse',
        'num_train_frames': 2,
    }
    create_train_test_set(configs)

    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 3,
        'scene_nums': [0, 1, 3, 4, 6],
        'train_views_density': 'sparse',
        'num_train_frames': 3,
    }
    create_train_test_set(configs)

    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 4,
        'scene_nums': [0, 1, 3, 4, 6],
        'train_views_density': 'sparse',
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
