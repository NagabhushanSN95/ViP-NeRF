# Shree KRISHNAya Namaha
# Computes visibility weights/masks for frames.
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import datetime
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy
import pandas
import simplejson
import skimage.io
from tqdm import tqdm

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class VisibilityWeightsComputer:
    def __init__(self, configs: dict):
        self.configs = configs
        return

    def compute_weights(self, frame1: numpy.ndarray, frame2: numpy.ndarray, extrinsic1, extrinsic2, intrinsic1,
                        intrinsic2, min_depth: float, max_depth: float):
        depth_planes = self.get_depth_planes(min_depth, max_depth, self.configs['num_depth_planes'])
        psv = self.create_psv(frame1.astype('float32'), frame2.astype('float32'), depth_planes, extrinsic1, extrinsic2, intrinsic1, intrinsic2)
        psv_error = psv - frame1[:, :, None, :]
        abs_error = numpy.mean((numpy.abs(psv_error)), axis=3)
        min_error = numpy.min(abs_error, axis=2)
        weights = numpy.exp(-min_error / self.configs['temperature'])
        return weights

    def get_depth_planes(self, min_depth, max_depth, num_depth_planes):
        depth_planes = 1 / numpy.linspace(1/min_depth, 1/max_depth, num_depth_planes)
        return depth_planes

    def create_psv(self, frame1, frame2, depth_planes, extrinsic1, extrinsic2, intrinsic1, intrinsic2):
        depth_psv = numpy.ones_like(frame1[:, :, 0])[:, :, None] * depth_planes[None, None, :]
        trans_coords = self.compute_transformed_coordinates(depth_psv, extrinsic1, extrinsic2, intrinsic1, intrinsic2)
        grid = self.create_grid(*frame1.shape[:2])[:, :, None, :]
        flow12 = trans_coords - grid
        psv = self.bilinear_interpolation(frame2, None, flow12, None, is_image=False)[0]
        return psv

    def compute_transformed_coordinates(self, depth1: numpy.ndarray, transformation1: numpy.ndarray,
                                   transformation2: numpy.ndarray, intrinsic1: numpy.ndarray,
                                   intrinsic2: Optional[numpy.ndarray]):
        """
        Computes transformed position for each pixel location
        """
        h, w, d = depth1.shape
        if intrinsic2 is None:
            intrinsic2 = numpy.copy(intrinsic1)
        transformation = numpy.matmul(transformation2, numpy.linalg.inv(transformation1))

        y1d = numpy.array(range(h))
        x1d = numpy.array(range(w))
        x2d, y2d = numpy.meshgrid(x1d, y1d)
        ones_2d = numpy.ones(shape=(h, w))
        ones_4d = numpy.repeat(ones_2d[:, :, None, None, None], repeats=d, axis=2)
        pos_vectors_homo = numpy.stack([x2d, y2d, ones_2d], axis=2)[:, :, None, :, None]

        intrinsic1_inv = numpy.linalg.inv(intrinsic1)
        intrinsic1_inv_4d = intrinsic1_inv[None, None, None]
        intrinsic2_4d = intrinsic2[None, None, None]
        depth_4d = depth1[:, :, :, None, None]
        trans_4d = transformation[None, None, None]  # (1, 1, 1, 4, 4)

        unnormalized_pos = numpy.matmul(intrinsic1_inv_4d, pos_vectors_homo)  # (h, w, 1, 3, 1)
        world_points = depth_4d * unnormalized_pos  # (h, w, d, 3, 1)
        world_points_homo = numpy.concatenate([world_points, ones_4d], axis=3)  # (h, w, d, 4, 1)
        trans_world_homo = numpy.matmul(trans_4d, world_points_homo)  # (h, w, d, 4, 1)
        trans_world = trans_world_homo[:, :, :, :3]  # (h, w, d, 3, 1)
        trans_norm_points = numpy.matmul(intrinsic2_4d, trans_world)  # (h, w, d, 3, 1)
        trans_coordinates = trans_norm_points[:, :, :, :2, 0] / trans_norm_points[:, :, :, 2:3, 0]  # (h, w, d, 2, 1)
        return trans_coordinates

    def bilinear_interpolation(self, frame2: numpy.ndarray, mask2: Optional[numpy.ndarray], flow12: numpy.ndarray,
                               flow12_mask: Optional[numpy.ndarray], is_image: bool = False) -> \
            Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Using bilinear interpolation
        :param frame2: (h, w, c)
        :param mask2: (h, w): True if known and False if unknown. Optional
        :param flow12: (h, w, 2)
        :param flow12_mask: (h, w): True if valid and False if invalid. Optional
        :param is_image: If true, the return array will be clipped to be in the range [0, 255] and type-casted to uint8
        :return: warped_frame1: (h, w, c)
                 mask1: (h, w): True if known and False if unknown
        """
        h, w, c = frame2.shape
        d = flow12.shape[2]
        if mask2 is None:
            mask2 = numpy.ones(shape=(h, w), dtype=bool)
        if flow12_mask is None:
            flow12_mask = numpy.ones(shape=(h, w, d), dtype=bool)
        grid = self.create_grid(h, w)[:, :, None, :]
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = numpy.floor(trans_pos_offset).astype('int')
        trans_pos_ceil = numpy.ceil(trans_pos_offset).astype('int')
        trans_pos_offset[:, :, :, 0] = numpy.clip(trans_pos_offset[:, :, :, 0], a_min=0, a_max=w + 1)
        trans_pos_offset[:, :, :, 1] = numpy.clip(trans_pos_offset[:, :, :, 1], a_min=0, a_max=h + 1)
        trans_pos_floor[:, :, :, 0] = numpy.clip(trans_pos_floor[:, :, :, 0], a_min=0, a_max=w + 1)
        trans_pos_floor[:, :, :, 1] = numpy.clip(trans_pos_floor[:, :, :, 1], a_min=0, a_max=h + 1)
        trans_pos_ceil[:, :, :, 0] = numpy.clip(trans_pos_ceil[:, :, :, 0], a_min=0, a_max=w + 1)
        trans_pos_ceil[:, :, :, 1] = numpy.clip(trans_pos_ceil[:, :, :, 1], a_min=0, a_max=h + 1)

        prox_weight_nw = (1 - (trans_pos_offset[:, :, :, 1] - trans_pos_floor[:, :, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, :, 0] - trans_pos_floor[:, :, :, 0]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, :, :, 1] - trans_pos_offset[:, :, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, :, 0] - trans_pos_floor[:, :, :, 0]))
        prox_weight_ne = (1 - (trans_pos_offset[:, :, :, 1] - trans_pos_floor[:, :, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, :, 0] - trans_pos_offset[:, :, :, 0]))
        prox_weight_se = (1 - (trans_pos_ceil[:, :, :, 1] - trans_pos_offset[:, :, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, :, 0] - trans_pos_offset[:, :, :, 0]))

        weight_nw = prox_weight_nw * flow12_mask
        weight_sw = prox_weight_sw * flow12_mask
        weight_ne = prox_weight_ne * flow12_mask
        weight_se = prox_weight_se * flow12_mask

        weight_nw_3d = weight_nw[:, :, :, None]
        weight_sw_3d = weight_sw[:, :, :, None]
        weight_ne_3d = weight_ne[:, :, :, None]
        weight_se_3d = weight_se[:, :, :, None]

        frame2_offset = numpy.pad(frame2, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        mask2_offset = numpy.pad(mask2, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=0)

        f2_nw = frame2_offset[trans_pos_floor[:, :, :, 1], trans_pos_floor[:, :, :, 0]]
        f2_sw = frame2_offset[trans_pos_ceil[:, :, :, 1], trans_pos_floor[:, :, :, 0]]
        f2_ne = frame2_offset[trans_pos_floor[:, :, :, 1], trans_pos_ceil[:, :, :, 0]]
        f2_se = frame2_offset[trans_pos_ceil[:, :, :, 1], trans_pos_ceil[:, :, :, 0]]

        m2_nw = mask2_offset[trans_pos_floor[:, :, :, 1], trans_pos_floor[:, :, :, 0]]
        m2_sw = mask2_offset[trans_pos_ceil[:, :, :, 1], trans_pos_floor[:, :, :, 0]]
        m2_ne = mask2_offset[trans_pos_floor[:, :, :, 1], trans_pos_ceil[:, :, :, 0]]
        m2_se = mask2_offset[trans_pos_ceil[:, :, :, 1], trans_pos_ceil[:, :, :, 0]]

        m2_nw_3d = m2_nw[:, :, :, None]
        m2_sw_3d = m2_sw[:, :, :, None]
        m2_ne_3d = m2_ne[:, :, :, None]
        m2_se_3d = m2_se[:, :, :, None]

        nr = weight_nw_3d * f2_nw * m2_nw_3d + weight_sw_3d * f2_sw * m2_sw_3d + \
             weight_ne_3d * f2_ne * m2_ne_3d + weight_se_3d * f2_se * m2_se_3d
        dr = weight_nw_3d * m2_nw_3d + weight_sw_3d * m2_sw_3d + weight_ne_3d * m2_ne_3d + weight_se_3d * m2_se_3d
        warped_frame1 = numpy.where(dr > 0, nr / dr, 0)
        mask1 = dr[:, :, :, 0] > 0

        if is_image:
            assert numpy.min(warped_frame1) >= 0
            assert numpy.max(warped_frame1) <= 256
            clipped_image = numpy.clip(warped_frame1, a_min=0, a_max=255)
            warped_frame1 = numpy.round(clipped_image).astype('uint8')
        return warped_frame1, mask1

    @staticmethod
    def create_grid(h, w):
        x_1d = numpy.arange(0, w)[None]
        y_1d = numpy.arange(0, h)[:, None]
        x_2d = numpy.repeat(x_1d, repeats=h, axis=0)
        y_2d = numpy.repeat(y_1d, repeats=w, axis=1)
        grid = numpy.stack([x_2d, y_2d], axis=2)
        return grid

    @staticmethod
    def read_image(path: Path) -> numpy.ndarray:
        if path.suffix in ['.jpg', '.png', '.bmp']:
            image = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            image = numpy.load(path.as_posix())
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return image

    @staticmethod
    def save_mask(path: Path, mask: numpy.ndarray, as_image: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        mask_image = mask.astype('uint8') * 255
        if path.suffix == '.png':
            skimage.io.imsave(path.as_posix(), mask_image)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), mask)
            if as_image:
                path1 = path.parent / f'{path.stem}.png'
                skimage.io.imsave(path1.as_posix(), mask_image)
        else:
            raise RuntimeError(f'Unknown format: {path.as_posix()}')
        return

    @staticmethod
    def save_weights(path: Path, weights: numpy.ndarray, as_png: bool = False):
        weights_image = numpy.round(weights * 255).astype('uint8')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in ['.png']:
            skimage.io.imsave(path.as_posix(), weights_image)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), weights)
            if as_png:
                path1 = path.parent / f'{path.stem}.png'
                skimage.io.imsave(path1.as_posix(), weights_image)
        else:
            raise RuntimeError(f'Unknown weights format: {path.as_posix()}')
        return


def save_configs(output_dirpath: Path, configs: dict):
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = simplejson.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs != old_configs:
            raise RuntimeError('Configs mismatch while resuming testing')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def start_generation(gen_configs: dict):
    root_dirpath = Path('../../../')
    database_dirpath = root_dirpath / 'data/databases' / gen_configs['database_dirpath']

    min_depth = 1
    max_depth = 100

    output_dirpath = database_dirpath / f"test/visibility_masks/VW{gen_configs['gen_num']:02}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    save_configs(output_dirpath, gen_configs)

    set_num = gen_configs['gen_set_num']
    video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_nums = numpy.unique(video_data['scene_num'].to_numpy())

    for scene_num in tqdm(scene_nums):
        frame_nums = video_data.loc[video_data['scene_num'] == scene_num]['pred_frame_num'].to_numpy()

        for frame1_num in frame_nums:
            for frame2_num in frame_nums:
                if frame2_num <= frame1_num:
                    continue

                mask1_output_path = output_dirpath / f'{scene_num:05}/visibility_masks/{frame1_num:04}_{frame2_num:04}.npy'
                mask2_output_path = output_dirpath / f'{scene_num:05}/visibility_masks/{frame2_num:04}_{frame1_num:04}.npy'
                weights1_output_path = output_dirpath / f'{scene_num:05}/visibility_weights/{frame1_num:04}_{frame2_num:04}.npy'
                weights2_output_path = output_dirpath / f'{scene_num:05}/visibility_weights/{frame2_num:04}_{frame1_num:04}.npy'
                if mask1_output_path.exists() and mask2_output_path.exists() and \
                        weights1_output_path.exists() and weights2_output_path.exists():
                    continue

                frame1_path = database_dirpath / f'test/database_data/{scene_num:05}/rgb/{frame1_num:04}.png'
                frame2_path = database_dirpath / f'test/database_data/{scene_num:05}/rgb/{frame2_num:04}.png'
                extrinsics_path = database_dirpath / f'test/database_data/{scene_num:05}/CameraExtrinsics.csv'
                intrinsics_path = database_dirpath / f'test/database_data/{scene_num:05}/CameraIntrinsics.csv'

                weights_computer = VisibilityWeightsComputer(gen_configs)

                frame1 = weights_computer.read_image(frame1_path)
                frame2 = weights_computer.read_image(frame2_path)
                extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))[frame_nums]
                intrinsics = numpy.loadtxt(intrinsics_path.as_posix(), delimiter=',').reshape((-1, 3, 3))[frame_nums]

                weights1 = weights_computer.compute_weights(frame1, frame2, extrinsics[0], extrinsics[1], intrinsics[0], intrinsics[1], min_depth, max_depth)
                weights2 = weights_computer.compute_weights(frame2, frame1, extrinsics[1], extrinsics[0], intrinsics[1], intrinsics[0], min_depth, max_depth)

                mask1 = weights1 > 0.5
                mask2 = weights2 > 0.5

                weights_computer.save_mask(mask1_output_path, mask1, as_image=True)
                weights_computer.save_mask(mask2_output_path, mask2, as_image=True)
                weights_computer.save_weights(weights1_output_path, weights1, as_png=True)
                weights_computer.save_weights(weights2_output_path, weights2, as_png=True)
    return


def demo1():
    gen_configs = {
        'generator': this_filename,
        'gen_num': 2,
        'gen_set_num': 2,
        'database_name': 'RealEstate10K',
        'database_dirpath': 'RealEstate10K/data',
        'num_depth_planes': 64,
        'temperature': 10,
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': this_filename,
        'gen_num': 3,
        'gen_set_num': 3,
        'database_name': 'RealEstate10K',
        'database_dirpath': 'RealEstate10K/data',
        'num_depth_planes': 64,
        'temperature': 10,
    }
    start_generation(gen_configs)

    gen_configs = {
        'generator': this_filename,
        'gen_num': 4,
        'gen_set_num': 4,
        'database_name': 'RealEstate10K',
        'database_dirpath': 'RealEstate10K/data',
        'num_depth_planes': 64,
        'temperature': 10,
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
