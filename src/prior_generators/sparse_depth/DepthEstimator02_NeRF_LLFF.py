# Shree KRISHNAya Namaha
# Estimates depth on NeRF-LLFF scenes
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import datetime
import json
import time
import traceback
from pathlib import Path

import numpy
import pandas
import simplejson
import skimage.io
from deepdiff import DeepDiff
from tqdm import tqdm

import Tester01 as Tester

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def read_image(path: Path):
    image = skimage.io.imread(path.as_posix())
    return image


def save_configs(output_dirpath: Path, configs: dict):
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = json.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs != old_configs:
            raise RuntimeError(f'Configs mismatch while resuming generation: {DeepDiff(old_configs, configs)}')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def start_generation(gen_configs: dict):
    root_dirpath = Path('../../../')
    database_dirpath = root_dirpath / 'data/databases' / gen_configs['database_dirpath']
    tmp_dirpath = root_dirpath / 'tmp'

    output_dirpath = database_dirpath / f"all/estimated_depths/DE{gen_configs['gen_num']:02}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    save_configs(output_dirpath, gen_configs)

    set_num = gen_configs['gen_set_num']
    video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_names = numpy.unique(video_data['scene_name'].to_numpy())

    tester = Tester.ColmapTester(tmp_dirpath)
    res_suffix = gen_configs['resolution_suffix']

    for scene_name in tqdm(scene_names):
        bounds_path = output_dirpath / f'{scene_name}/EstimatedBounds.csv'
        if bounds_path.exists():
            continue

        frame_nums = video_data.loc[video_data['scene_name'] == scene_name]['pred_frame_num'].to_numpy()
        frames = [read_image(database_dirpath / f'all/database_data/{scene_name}/rgb{res_suffix}/{frame_num:04}.png') for frame_num in frame_nums]
        frames = numpy.stack(frames)
        intrinsics_path = database_dirpath / f'all/database_data/{scene_name}/CameraIntrinsics{res_suffix}.csv'
        extrinsics_path = database_dirpath / f'all/database_data/{scene_name}/CameraExtrinsics.csv'
        intrinsics = numpy.loadtxt(intrinsics_path.as_posix(), delimiter=',').reshape((-1, 3, 3))[frame_nums]
        extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))[frame_nums]

        depth_data_list, bounds_data = tester.estimate_sparse_depth(frames, extrinsics, intrinsics)
        if depth_data_list is None:
            continue

        for i, frame_num in enumerate(frame_nums):
            depth_path = output_dirpath / f'{scene_name}/estimated_depths{res_suffix}/{frame_num:04}.csv'
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            depth_data_list[i].to_csv(depth_path, index=False)
        bounds_data.to_csv(bounds_path, index=False)
    return


def demo1():
    """
    For a gen set
    :return:
    """
    gen_configs = {
        'generator': this_filename,
        'gen_num': 2,
        'gen_set_num': 2,
        'database_name': 'NeRF_LLFF',
        'database_dirpath': 'NeRF_LLFF/data',
        'resolution_suffix': '_down4',
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': this_filename,
        'gen_num': 3,
        'gen_set_num': 3,
        'database_name': 'NeRF_LLFF',
        'database_dirpath': 'NeRF_LLFF/data',
        'resolution_suffix': '_down4',
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': this_filename,
        'gen_num': 4,
        'gen_set_num': 4,
        'database_name': 'NeRF_LLFF',
        'database_dirpath': 'NeRF_LLFF/data',
        'resolution_suffix': '_down4',
    }
    start_generation(gen_configs)
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
