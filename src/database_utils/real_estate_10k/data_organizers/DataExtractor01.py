# ShreeRamaJayam
# Extracts videos from the url links, for training
# Creates train data 'Train' of the foll format: 2k scenes, from the 'train' folder, each 12 frames
# train_rdtvs_v2 is designed for 10step sampling ( 41 frames extracted from each scenes corres to frame n-2, n, n+1 and n+2)
# Author: KV
# Modified: 23/09/2021

import datetime
import os
import shutil
import time
import traceback
from enum import Enum
from pathlib import Path

import numpy
import pandas
import simplejson
import skimage.io
from deepdiff import DeepDiff
from skimage.transform import resize
from tqdm import tqdm

# from pytube import YouTube

this_filepath = Path(__file__)
this_filename = this_filepath.stem


# ------------- Enums for easier data passing ---------- #
class DataFeatures(Enum):
    FRAME = 'frame'
    INTRINSIC = 'intrinsic'
    EXTRINSIC = 'extrinsic'


class DataExtractor:
    def __init__(self, extracted_dirpath: Path, configs, tmp_dirpath: Path, output_dirpath: Path):
        self.extracted_dirpath = extracted_dirpath
        self.configs = configs
        self.tmp_dirpath = tmp_dirpath
        self.output_dirpath = output_dirpath
        return

    def extract_data(self, features, indices=None):
        """
        Loads data for the scenes to be downloaded, as specified by indices
        :param features:
        :param indices:
        :return:
        """
        scenes_data_path = self.output_dirpath / 'Cache/AllScenesData.csv'
        scenes_data = pandas.read_csv(scenes_data_path)
        if indices is not None:
            scenes_data = scenes_data.iloc[indices]
        for scene_num, start_timestamp in tqdm(scenes_data.to_numpy()):
            scene_name = f'{scene_num:05}'

            # Check if this scene has already been extracted
            intrinsics_path = self.output_dirpath / f'{scene_name}/CameraIntrinsics.csv'
            if intrinsics_path.exists():
                continue

            scene_datapath = self.extracted_dirpath / f'{self.configs["split_name"]}/{scene_name}/CameraData.txt'
            with open(scene_datapath.as_posix(), 'r') as scene_data_file:
                scene_data = [line.strip() for line in scene_data_file.readlines()]
            url = scene_data[0]
            scene_data = [line.split(' ') for line in scene_data[1:]]
            scene_data = numpy.array(scene_data)
            start_line_num = numpy.where(scene_data[:, 0].astype('int') == start_timestamp)[0][0]
            step_size = self.configs['step_size']
            end_line_num = start_line_num + self.configs['num_frames_per_scene'] * step_size
            frames_data = scene_data[start_line_num:end_line_num:step_size]

            frames, extrinsics, intrinsics = None, None, None
            if DataFeatures.FRAME in features:
                timestamps = frames_data[:, 0].astype('int')
                frames = self.download_frames(scene_name, url, timestamps)
                if frames is None:
                    continue
            if DataFeatures.EXTRINSIC in features:
                extrinsics = self.compute_extrinsic_matrices(frames_data[:, 7:19])
            if DataFeatures.INTRINSIC in features:
                intrinsics = self.compute_intrinsic_matrices(frames_data[:, 1:5])

            self.save_data(scene_name, features, frames, intrinsics, extrinsics)
        return

    def download_frames(self, scene_name: str, url: str, timestamps: numpy.ndarray):
        if self.tmp_dirpath.exists():
            shutil.rmtree(self.tmp_dirpath)
        self.tmp_dirpath.mkdir(parents=True, exist_ok=False)

        # Download the video if it hasn't been downloaded previously
        video_dirpath = self.extracted_dirpath / f'{self.configs["split_name"]}/{scene_name}'
        if len(list(video_dirpath.rglob(f'./{scene_name}.*'))) == 0:
            video_filepath = video_dirpath / 'video.mp4'
            cmd = f'youtube-dl -o {video_filepath.absolute().as_posix()} {url}'
            return_code = os.system(cmd)
        else:
            print(f'video {scene_name} already downloaded')
            return_code = 0

        if return_code == 0:
            # Get the saved video path
            video_filepath = list(video_dirpath.rglob('video.*'))[0]
            resolution = self.configs['resolution']
            # Extract required frames
            frames = []
            for timestamp in timestamps:
                timestamp = int(timestamp / 1000)
                str_hour = str(int(timestamp / 3600000)).zfill(2)
                str_min = str(int(int(timestamp % 3600000) / 60000)).zfill(2)
                str_sec = str(int(int(int(timestamp % 3600000) % 60000) / 1000)).zfill(2)
                str_mill = str(int(int(int(timestamp % 3600000) % 60000) % 1000)).zfill(3)
                str_timestamp = str_hour + ":" + str_min + ":" + str_sec + "." + str_mill
                tmp_frame_filepath = self.tmp_dirpath / f'{timestamp}.png'
                cmd = f'/opt/FFmpeg/ffmpeg-4.4-i686-static/ffmpeg -loglevel quiet -ss {str_timestamp} -i {video_filepath.as_posix()} -vframes 1 -f image2 {tmp_frame_filepath.as_posix()}'
                return_code = os.system(cmd)
                if return_code == 0:
                    frame = skimage.io.imread(tmp_frame_filepath.as_posix())
                    if resolution is not None:
                        resized_frame = skimage.transform.resize(frame, output_shape=resolution, preserve_range=True, anti_aliasing=True)
                        frame = numpy.round(resized_frame).astype('uint8')
                    frames.append(frame)
                else:
                    frames = None
                    break
            frames = numpy.stack(frames) if frames is not None else None
        else:
            print(f'Unable to download scene: {scene_name}')
            frames = None

        shutil.rmtree(self.tmp_dirpath)
        return frames

    def compute_intrinsic_matrices(self, intrinsics_data: numpy.ndarray):
        intrinsics_data = intrinsics_data.astype('float32')
        h, w = self.configs['resolution']
        num_frames = intrinsics_data.shape[0]
        intrinsic_matrices = numpy.zeros(shape=(num_frames, 9), dtype=numpy.float32)
        fx, fy, px, py = [x.squeeze() for x in numpy.split(intrinsics_data, 4, axis=1)]
        intrinsic_matrices[:, 0] = w * fx
        intrinsic_matrices[:, 4] = h * fy
        intrinsic_matrices[:, 2] = w * px
        intrinsic_matrices[:, 5] = h * py
        intrinsic_matrices[:, 8] = 1
        return intrinsic_matrices

    @staticmethod
    def compute_extrinsic_matrices(extrinsics_data: numpy.ndarray):
        extrinsics_data = extrinsics_data.astype('float32')
        num_frames = extrinsics_data.shape[0]
        last_row = numpy.zeros(shape=(num_frames, 4), dtype=numpy.float32)
        last_row[:, 3] = 1
        extrinsic_matrices = numpy.concatenate([extrinsics_data, last_row], axis=1)
        return extrinsic_matrices

    def save_data(self, scene_name, features, frames, intrinsics, extrinsics):
        scene_dirpath = self.output_dirpath / scene_name
        scene_dirpath.mkdir(parents=True, exist_ok=True)

        if DataFeatures.FRAME in features:
            self.save_frames(scene_dirpath, frames)

        if DataFeatures.EXTRINSIC in features:
            self.save_extrinsics(scene_dirpath, extrinsics)

        if DataFeatures.INTRINSIC in features:
            self.save_intrinsics(scene_dirpath, intrinsics)
        return

    @staticmethod
    def save_frames(scene_dirpath: Path, frames: numpy.ndarray, frame_nums: numpy.ndarray = None, resolution: tuple = None):
        rgb_dirpath = scene_dirpath / 'rgb'
        rgb_dirpath.mkdir(parents=True, exist_ok=False)
        if frame_nums is None:
            frame_nums = numpy.arange(frames.shape[0])
        for frame_num, frame in zip(frame_nums, frames):
            frame_path = rgb_dirpath / f'{frame_num:04}.png'
            skimage.io.imsave(frame_path.as_posix(), frame)
        return

    @staticmethod
    def save_intrinsics(scene_dirpath: Path, intrinsics: numpy.ndarray):
        intrinsics_path = scene_dirpath / 'CameraIntrinsics.csv'
        numpy.savetxt(intrinsics_path.as_posix(), intrinsics, delimiter=',')
        return

    @staticmethod
    def save_extrinsics(scene_dirpath: Path, extrinsics: numpy.ndarray):
        extrinsics_path = scene_dirpath / 'CameraExtrinsics.csv'
        numpy.savetxt(extrinsics_path.as_posix(), extrinsics, delimiter=',')
        return


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
    return new_configs


def demo1():
    configs = {
        'DataExtractor': this_filename,
        'split_name': 'test',
        'identifier': 2,
        'resolution': [576, 1024],
    }
    features = [DataFeatures.FRAME, DataFeatures.EXTRINSIC, DataFeatures.INTRINSIC]

    root_dirpath = Path(f"../")
    extracted_dirpath = root_dirpath / 'data/extracted_data'
    processed_dirpath = root_dirpath / 'data/processed_data'
    tmp_dirpath = root_dirpath / 'tmp'
    output_dirpath = processed_dirpath / f"{configs['split_name']}{configs['identifier']:02}"
    configs = save_configs(output_dirpath, configs)

    data_extractor = DataExtractor(extracted_dirpath, configs, tmp_dirpath, output_dirpath)
    data_extractor.extract_data(features, range(500))
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
