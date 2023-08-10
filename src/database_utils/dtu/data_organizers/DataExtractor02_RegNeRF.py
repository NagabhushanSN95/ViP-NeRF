# Shree KRISHNAya Namaha
# Extracts RegNeRF data (object masks) in my convention from the unzipped files
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import datetime
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import List

import numpy
import skimage.io
import skimage.transform
from tqdm import tqdm

this_filepath = Path(__file__)
this_filename = this_filepath.stem


# ------------- Enums for easier data passing ---------- #
class DataFeatures(Enum):
    OBJECT_MASKS = 'object_masks'


class DataExtractor:
    def __init__(self, unzipped_dirpath: Path, extracted_dirpath: Path, features: List[DataFeatures]):
        self.unzipped_dirpath = unzipped_dirpath
        self.extracted_dirpath = extracted_dirpath
        self.features = features
        return

    def extract_data(self, downsampling_factor: int):
        for scene_dirpath in tqdm(sorted(self.unzipped_dirpath.iterdir())):
            if not scene_dirpath.is_dir():
                continue

            scene_num = int(scene_dirpath.stem[4:])
            num_frames = 64

            if DataFeatures.OBJECT_MASKS in self.features:
                for frame_num in range(num_frames):
                    src_mask_path = scene_dirpath / f'{frame_num:03}.png'
                    if not src_mask_path.exists():
                        src_mask_path = scene_dirpath / f'mask/{frame_num:03}.png'
                    tgt_mask_path = self.extracted_dirpath / f'{scene_num:05}/object_masks/{frame_num:04}.png'
                    if src_mask_path.exists():
                        self.extract_mask(src_mask_path, tgt_mask_path, downsampling_factor)
        return

    def extract_mask(self, src_path: Path, tgt_path: Path, downsampling_factor: int):
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        mask = self.read_mask(src_path)
        down_mask = skimage.transform.rescale(mask, scale=1 / downsampling_factor, preserve_range=True,
                                              anti_aliasing=False)
        self.save_mask(tgt_path, down_mask)
        return

    @staticmethod
    def read_mask(path: Path):
        mask_image = skimage.io.imread(path.as_posix())
        mask = numpy.mean(mask_image, axis=2) >= 128
        return mask

    @staticmethod
    def save_mask(path: Path, mask: numpy.ndarray):
        mask_image = mask.astype('uint8') * 255
        skimage.io.imsave(path.as_posix(), mask_image)
        return


def demo1():
    features = [DataFeatures.OBJECT_MASKS]

    root_dirpath = Path('../../../../')
    database_dirpath = root_dirpath / 'data/databases/DTU/data'
    unzipped_dirpath = database_dirpath / 'all/unzipped_data/RegNeRF/idrmasks'
    extracted_dirpath = database_dirpath / 'all/database_data'

    data_extractor = DataExtractor(unzipped_dirpath, extracted_dirpath, features)
    data_extractor.extract_data(downsampling_factor=4)
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
