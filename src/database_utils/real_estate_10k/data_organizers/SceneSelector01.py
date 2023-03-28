# Shree KRISHNAya Namaha
# Run this file first to select videos to be downloaded. Then run DataExtractor to extract the data.
# It allows choosing a percentage of videos to contain primarily motion in x or y direction and remaining videos with
# arbitrary motion.
# Author: Nagabhushan S N
# Modified: 12/01/2022

import datetime
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy
import pandas
import simplejson
from deepdiff import DeepDiff
from tqdm import tqdm

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class VideosSelector:
    def __init__(self, extracted_dirpath: Path, configs, output_dirpath: Path):
        self.extracted_dirpath = extracted_dirpath
        self.configs = configs
        self.split_name = configs['split_name']
        self.src_dirpath = self.extracted_dirpath / self.split_name
        self.step_size = configs['step_size']
        self.start_offset = configs['start_offset']
        self.end_offset = configs['end_offset']
        self.num_scenes = configs['num_scenes']
        self.num_frames_per_scene = configs['num_frames_per_scene']
        self.segment_filter = self.get_segment_filter(configs['segment_filter']['name'])
        self.cache_segments_data = configs['cache_segments_data']
        self.output_dirpath = output_dirpath

        self.scenes_list = sorted([item.stem for item in self.src_dirpath.iterdir()])
        return

    def compile_segments(self):
        if self.cache_segments_data:
            filtered_cache_filepath = self.output_dirpath / f'Cache/FilteredSegmentsData.csv'
            random_cache_filepath = self.output_dirpath / f'Cache/RandomSegmentsData.csv'
            if filtered_cache_filepath.exists() and random_cache_filepath.exists():
                filtered_segments_data = pandas.read_csv(filtered_cache_filepath)
                random_segments_data = pandas.read_csv(random_cache_filepath)
                return filtered_segments_data, random_segments_data

        filtered_segments, random_segments = [], []
        for scene_name in tqdm(self.scenes_list):
            scene_data_path = self.extracted_dirpath / self.split_name / f'{scene_name}/CameraData.txt'
            with open(scene_data_path.as_posix(), 'r') as scene_data_file:
                scene_data = scene_data_file.readlines()[1:]
                scene_data = numpy.array([line.split(' ') for line in scene_data])
                scene_data = scene_data[self.start_offset: scene_data.shape[0]-self.end_offset]
            num_segments = len(scene_data) - (self.num_frames_per_scene - 1) * self.step_size
            if num_segments <= 0:
                continue

            random_segment_index = numpy.random.randint(0, num_segments)
            random_timestamp = scene_data[random_segment_index, 0]
            random_segments.append([scene_name, random_timestamp])

            rel_distances = self.compute_distances(scene_data)
            rel_distances_segment_wise = [rel_distances[i:i+(self.num_frames_per_scene-1)*self.step_size: self.step_size] for i in range(num_segments)]
            rel_distances_segment_wise = numpy.array(rel_distances_segment_wise)
            filtered_segment_data = self.segment_filter(scene_data[:num_segments, 0], rel_distances_segment_wise)
            num_selected_segments = filtered_segment_data.shape[0]
            if num_selected_segments <= 0:
                continue
            selected_scene_data = numpy.array([scene_name]*num_selected_segments)[:, None]
            filtered_segment_data = numpy.concatenate([selected_scene_data, filtered_segment_data], axis=1)
            filtered_segments.append(filtered_segment_data)

        filtered_segments_data = numpy.concatenate(filtered_segments, axis=0)
        filtered_segments_data = pandas.DataFrame(filtered_segments_data, columns=['scene_name', 'start_timestamp', 'average_translation'])
        random_segments_data = numpy.array(random_segments)
        random_segments_data = pandas.DataFrame(random_segments_data, columns=['scene_name', 'start_timestamp'])

        if self.cache_segments_data:
            filtered_cache_filepath.parent.mkdir(parents=True, exist_ok=True)
            filtered_segments_data.to_csv(filtered_cache_filepath, index=False)
            random_segments_data.to_csv(random_cache_filepath, index=False)
        return filtered_segments_data, random_segments_data

    def select_segments(self, filtered_segments_data: pandas.DataFrame, random_segments_data: pandas.DataFrame):
        num_segments = self.configs['num_scenes']
        num_filtered_segments = num_segments * self.configs['percentage_xy_motion_scenes'] // 100
        num_random_segments = num_segments - num_filtered_segments

        # Choose filtered segments
        sorted_segments_data = filtered_segments_data.sort_values(by='average_translation', ascending=False)
        unique_segments_data = sorted_segments_data.drop_duplicates('scene_name')[['scene_name', 'start_timestamp']]
        selected_filtered_segments_data = unique_segments_data[:num_filtered_segments]

        # Choose random segments
        random_segments_data = pandas.concat([selected_filtered_segments_data, random_segments_data], axis=0)
        unique_segments_data = random_segments_data.drop_duplicates('scene_name')
        unique_random_segments_data = unique_segments_data[num_filtered_segments:]
        selected_random_segments_data = unique_random_segments_data[:num_random_segments]

        selected_segments_data = pandas.concat([selected_filtered_segments_data, selected_random_segments_data], axis=0)

        selected_segments_data = selected_segments_data.sort_values(by='scene_name', ascending=True)
        selected_filtered_segments_data = selected_filtered_segments_data.sort_values(by='scene_name', ascending=True)
        selected_random_segments_data = selected_random_segments_data.sort_values(by='scene_name', ascending=True)
        return selected_segments_data, selected_filtered_segments_data, selected_random_segments_data

    def compute_distances(self, scene_data: numpy.ndarray):
        num_frames = scene_data.shape[0]
        transformations = scene_data[:, 7:].reshape(num_frames, 3, 4).astype(numpy.float)
        last_row = numpy.array([[0, 0, 0, 1]] * num_frames)[:, None, :]
        transformations = numpy.concatenate([transformations, last_row], axis=1)
        rel_transformations = self.compute_relative_transformations(transformations, self.step_size)
        rel_translations = rel_transformations[:, :3, 3]
        rel_distances = numpy.linalg.norm(rel_translations, axis=1, keepdims=True)
        all_rel_distances = numpy.concatenate([rel_translations, rel_distances], axis=1)
        all_rel_distances = numpy.abs(all_rel_distances)
        return all_rel_distances

    @staticmethod
    def compute_relative_transformations(transformations: numpy.ndarray, step_size: int) -> numpy.ndarray:
        transformations1 = transformations[:-step_size]
        transformations1_inv = numpy.linalg.inv(transformations1)
        transformations2 = transformations[step_size:]
        relative_trans = numpy.matmul(transformations2, transformations1_inv)
        return relative_trans

    def get_segment_filter(self, name: str) -> Callable:
        if name is None:
            segment_filter = None
        elif name == 'segment_filter01':
            segment_filter = self.segment_filter01
        else:
            raise RuntimeError(f'Unknown segment filter: {name}')
        return segment_filter

    def segment_filter01(self, timestamps: numpy.ndarray, distances: numpy.ndarray):
        translation_segments = numpy.min(distances[:, :, 3], axis=1) >= self.configs['segment_filter']['translation_threshold']
        xy_motion_segments = numpy.any((distances[:, :, 2] < distances[:, :, 0]) | (distances[:, :, 2] < distances[:, :, 1]), axis=1)
        selected_segments = translation_segments & xy_motion_segments
        selected_timestamps = timestamps[selected_segments]
        selected_distances = numpy.mean(distances[selected_segments, :, 3], axis=1)
        selected_segment_data = numpy.stack([selected_timestamps, selected_distances], axis=1)
        return selected_segment_data


def save_configs(output_dirpath: Path, new_configs: dict):
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = simplejson.load(configs_file)
        for key in old_configs.keys():
            if key not in new_configs.keys():
                new_configs[key] = old_configs[key]
        for key in new_configs.keys():
            if key not in old_configs.keys():
                old_configs[key] = new_configs[key]
        if DeepDiff(old_configs, new_configs):
            raise RuntimeError('Configs mismatch while resuming testing')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(new_configs, configs_file, indent=4)
    return


def demo1():
    configs = {
        'SceneSelector': this_filename,
        'split_name': 'test',
        'identifier': 1,
        'num_scenes': 10,
        'percentage_xy_motion_scenes': 50,
        'step_size': 1,
        'start_offset': 15,
        'end_offset': 0,
        'num_frames_per_scene': 50,
        'segment_filter': {
            'name': 'segment_filter01',
            'translation_threshold': 0.01,
        },
        'cache_segments_data': True,
    }

    root_dirpath = Path('../../../../')
    extracted_dirpath = root_dirpath / 'data/databases/RealEstate10K/data/extracted_data'
    processed_dirpath = root_dirpath / 'data/databases/RealEstate10K/data/processed_data'
    output_dirpath = processed_dirpath / f"{configs['split_name']}{configs['identifier']:02}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    save_configs(output_dirpath, configs)

    selector = VideosSelector(extracted_dirpath, configs, output_dirpath)
    filtered_segments_data, random_segments_data = selector.compile_segments()
    selected_data = selector.select_segments(filtered_segments_data, random_segments_data)
    selected_segments_data, selected_filtered_segments_data, selected_random_segments_data = selected_data

    selected_segments_path = output_dirpath / 'Cache/AllScenesData.csv'
    selected_filtered_segments_path = output_dirpath / 'Cache/FilteredScenesData.csv'
    selected_random_segments_path = output_dirpath / 'Cache/RandomScenesData.csv'
    selected_segments_data.to_csv(selected_segments_path, index=False)
    selected_filtered_segments_data.to_csv(selected_filtered_segments_path, index=False)
    selected_random_segments_data.to_csv(selected_random_segments_path, index=False)
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
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
