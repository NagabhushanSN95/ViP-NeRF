# Shree KRISHNAya Namaha
# Extracts pixelNeRF data (downsampled to 300x400) in my convention from the unzipped files
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import datetime
import shutil
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import List

import cv2
import numpy
import skimage.io
from tqdm import tqdm

this_filepath = Path(__file__)
this_filename = this_filepath.stem


# ------------- Enums for easier data passing ---------- #
class DataFeatures(Enum):
    FRAME = 'frame'
    INTRINSIC = 'intrinsic'
    EXTRINSIC = 'extrinsic'


class DataExtractor:
    def __init__(self, unzipped_dirpath: Path, extracted_dirpath: Path, features: List[DataFeatures]):
        self.unzipped_dirpath = unzipped_dirpath
        self.extracted_dirpath = extracted_dirpath
        self.features = features
        return

    def extract_data(self):
        for scene_dirpath in tqdm(sorted(self.unzipped_dirpath.iterdir())):
            if not scene_dirpath.is_dir():
                continue

            scene_num = int(scene_dirpath.stem[4:])
            num_frames = len(list((scene_dirpath / 'image').iterdir()))
            resolution = None

            if DataFeatures.FRAME in self.features:
                for frame_num in range(num_frames):
                    src_frame_path = scene_dirpath / f'image/{frame_num:06}.png'
                    tgt_frame_path = self.extracted_dirpath / f'{scene_num:05}/rgb/{frame_num:04}.png'
                    self.extract_frame(src_frame_path, tgt_frame_path)
                    if resolution is None:
                        resolution = skimage.io.imread(src_frame_path.as_posix()).shape[:2]

            permuter = numpy.eye(4)
            permuter[1:3, 1:3] *= -1

            cameras_path = scene_dirpath / 'cameras.npz'
            intrinsics, extrinsics = [], []
            with numpy.load(cameras_path.as_posix()) as camera_data:
                for frame_num in range(num_frames):
                    int_ext = camera_data[f'world_mat_{frame_num}']
                    intrinsic, rot, trans = cv2.decomposeProjectionMatrix(int_ext[:3])[:3]

                    intrinsic = intrinsic / intrinsic[2, 2]
                    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
                    intrinsic = numpy.eye(3)
                    intrinsic[0, 0] = fx
                    intrinsic[1, 1] = fy
                    intrinsic[0, 2] = resolution[1] / 2
                    intrinsic[1, 2] = resolution[0] / 2
                    intrinsics.append(intrinsic)

                    extrinsic = numpy.eye(4, dtype=numpy.float32)
                    extrinsic[:3, :3] = rot.transpose()
                    extrinsic[:3, 3] = (trans[:3] / trans[3])[:, 0]

                    scale_mat = camera_data.get(f'scale_mat_{frame_num}')
                    if scale_mat is not None:
                        norm_trans = scale_mat[:3, 3:]
                        norm_scale = numpy.diagonal(scale_mat[:3, :3])[..., None]
                        extrinsic[:3, 3:] -= norm_trans
                        extrinsic[:3, 3:] /= norm_scale

                    # extrinsic = permuter @ extrinsic @ permuter.T
                    extrinsic = numpy.linalg.inv(extrinsic)
                    extrinsics.append(extrinsic)
            intrinsics = numpy.stack(intrinsics)
            focal_length = numpy.sum(intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / (2 * num_frames)
            intrinsics[:, 0, 0] = focal_length
            intrinsics[:, 1, 1] = focal_length
            extrinsics = numpy.stack(extrinsics)

            if DataFeatures.INTRINSIC in self.features:
                tgt_intrinsics_path = self.extracted_dirpath / f'{scene_num:05}/CameraIntrinsics.csv'
                self.extract_intrinsics(tgt_intrinsics_path, intrinsics)

            if DataFeatures.EXTRINSIC in self.features:
                tgt_extrinsics_path = self.extracted_dirpath / f'{scene_num:05}/CameraExtrinsics.csv'
                self.extract_extrinsics(tgt_extrinsics_path, extrinsics)
        return

    @staticmethod
    def extract_frame(src_path: Path, tgt_path: Path):
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_path.as_posix(), tgt_path.as_posix())
        return

    @staticmethod
    def extract_intrinsics(tgt_intrinsics_path: Path, intrinsics: numpy.ndarray):
        tgt_intrinsics_path.parent.mkdir(parents=True, exist_ok=True)
        numpy.savetxt(tgt_intrinsics_path.as_posix(), intrinsics.reshape(intrinsics.shape[0], -1), delimiter=',')
        return

    @staticmethod
    def extract_extrinsics(tgt_extrinsics_path: Path, extrinsics: numpy.ndarray):
        tgt_extrinsics_path.parent.mkdir(parents=True, exist_ok=True)
        numpy.savetxt(tgt_extrinsics_path.as_posix(), extrinsics.reshape(extrinsics.shape[0], -1), delimiter=',')
        return

    @staticmethod
    def change_coordinate_system(extrinsic: numpy.ndarray, p: numpy.ndarray):
        r = extrinsic[:3, :3]
        t = extrinsic[:3, 3:]
        rc = p.T @ r @ p
        tc = p @ t
        changed_extrinsic = numpy.concatenate([numpy.concatenate([rc, tc], axis=1), extrinsic[3:]], axis=0)
        return changed_extrinsic


def demo1():
    features = [DataFeatures.FRAME, DataFeatures.EXTRINSIC, DataFeatures.INTRINSIC]

    root_dirpath = Path('../../../../')
    database_dirpath = root_dirpath / 'data/databases/DTU/data'
    unzipped_dirpath = database_dirpath / 'all/unzipped_data/PixelNeRF/rs_dtu_4/DTU'
    extracted_dirpath = database_dirpath / 'all/database_data'

    data_extractor = DataExtractor(unzipped_dirpath, extracted_dirpath, features)
    data_extractor.extract_data()
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
