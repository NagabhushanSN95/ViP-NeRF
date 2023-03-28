# Shree KRISHNAYa Namaha
# Loads RealEstate Data for NeRF, Colmap sparse depth, dense depth, visibility prior from RealEstate-10K dataset.
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

from pathlib import Path
from typing import Optional

import numpy
import pandas
import skimage.io

from data_loaders.DataLoaderParent import DataLoaderParent


class RealEstateDataLoader(DataLoaderParent):
    def __init__(self, configs: dict, data_dirpath: Path, mode: Optional[str]):
        super(RealEstateDataLoader, self).__init__()
        self.configs = configs
        self.data_dirpath = data_dirpath
        self.mode = mode
        self.scene_num = int(configs['data_loader']['scene_id'])

        self.sparse_depth_needed = 'sparse_depth' in self.configs['data_loader']
        self.dense_depth_needed = ('dense_depth' in self.configs['data_loader'])
        self.visibility_prior_needed = ('visibility_prior' in self.configs['data_loader'])
        return

    def load_data(self):
        frame_nums = self.get_frame_nums()
        data_dict = {
            'frame_nums': frame_nums,
        }

        data_dict['nerf_data'] = self.load_nerf_data(data_dict)

        if self.sparse_depth_needed and self.mode == 'train':
            data_dict['sparse_depth_data'] = self.load_sparse_depth_data(data_dict)

        if self.dense_depth_needed and self.mode == 'train':
            data_dict['dense_depth_data'] = self.load_dense_depth_data(data_dict)

        if self.visibility_prior_needed and self.mode == 'train':
            data_dict['visibility_prior_data'] = self.load_visibility_prior_data(data_dict)

        return data_dict

    def get_frame_nums(self):
        set_num = self.configs['data_loader']['train_set_num']
        video_datapath = self.data_dirpath / f'train_test_sets/set{set_num:02}/{self.mode.capitalize()}VideosData.csv'
        video_data = pandas.read_csv(video_datapath)
        frame_nums = video_data.loc[video_data['scene_num'] == self.scene_num]['pred_frame_num'].to_numpy()
        return frame_nums

    def load_nerf_data(self, data_dict):
        frame_nums = data_dict['frame_nums']
        images_dirpath = self.data_dirpath / f'test/database_data/{self.scene_num:05}/rgb'
        if not images_dirpath.exists():
            print(f'{images_dirpath.as_posix()} does not exist, returning.')
            return
        images_paths = [images_dirpath / f'{frame_num:04}.png' for frame_num in frame_nums]
        images = [self.read_image(image_path) for image_path in images_paths]
        images = numpy.stack(images)

        bounds = numpy.array([1, 100]).astype('float32')

        extrinsics_path = self.data_dirpath / f'test/database_data/{self.scene_num:05}/CameraExtrinsics.csv'
        extrinsic_matrices = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))
        extrinsics = extrinsic_matrices[frame_nums]

        intrinsics_path = self.data_dirpath / f'test/database_data/{self.scene_num:05}/CameraIntrinsics.csv'
        intrinsic_matrices = numpy.loadtxt(intrinsics_path.as_posix(), delimiter=',').reshape((-1, 3, 3))
        intrinsics = intrinsic_matrices[frame_nums]

        h, w = images.shape[1:3]
        return_dict = {
            'images': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'resolution': (h, w),
            'bounds': bounds,
        }
        return return_dict

    def load_sparse_depth_data(self, data_dict: dict):
        sparse_depth_data = {}
        depth_dirname = self.configs['data_loader']['sparse_depth']['dirname']
        for frame_num in data_dict['frame_nums']:
            depth_path = self.data_dirpath / f'test/estimated_depths/{depth_dirname}/{self.scene_num:05}/estimated_depths/{frame_num:04}.csv'
            depth_data = pandas.read_csv(depth_path)
            sparse_depth_data[frame_num] = depth_data
        return sparse_depth_data

    def load_dense_depth_data(self, data_dict: dict):
        dense_depth_data = {}

        h, w = data_dict['nerf_data']['resolution']
        depths, depth_weights = [], []
        depth_dirname = self.configs['data_loader']['dense_depth']['dirname']
        weights_suffix = ''
        if 'weights_suffix' in self.configs['data_loader']['dense_depth']:
            weights_suffix = self.configs['data_loader']['dense_depth']['weights_suffix']
        for frame_num in data_dict['frame_nums']:
            depth_path = self.data_dirpath / f'test/estimated_depths/{depth_dirname}/{self.scene_num:05}/estimated_depths/{frame_num:04}.npy'
            print(f'Loading depth: {depth_path.as_posix()}')

            depth = numpy.load(depth_path.as_posix())
            depths.append(depth)

            weights_path = self.data_dirpath / f'test/estimated_depths/{depth_dirname}/{self.scene_num:05}/Weights{weights_suffix}/{frame_num:04}.npy'
            if weights_path.exists():
                depth_weight = numpy.load(weights_path.as_posix())[:, :]
            else:
                print(f'Dense Depth Weights {weights_path.as_posix()} not found!. Loading unit weights.')
                depth_weight = numpy.ones(shape=(h, w))
            depth_weights.append(depth_weight)
        depths = numpy.stack(depths, axis=0)
        depth_weights = numpy.stack(depth_weights, axis=0)

        dense_depth_data['depth_values'] = depths
        dense_depth_data['depth_weights'] = depth_weights

        return dense_depth_data

    def load_visibility_prior_data(self, data_dict):
        visibility_prior_data = {}

        if self.configs['data_loader']['visibility_prior']['load_masks']:
            masks = []
            masks_dirname = self.configs['data_loader']['visibility_prior']['masks_dirname']
            frame1_nums = data_dict['frame_nums']
            for frame1_num in frame1_nums:
                frame2_nums = [x for x in frame1_nums if x != frame1_num]
                frame1_masks = []
                for frame2_num in frame2_nums:
                    mask_path = self.data_dirpath / f'test/visibility_masks/{masks_dirname}/{self.scene_num:05}/visibility_masks/{frame1_num:04}_{frame2_num:04}.png'
                    print(f'Loading visibility prior mask: {mask_path.as_posix()}')
                    mask = self.read_mask(mask_path)
                    frame1_masks.append(mask)
                masks.append(frame1_masks)
            masks = numpy.array(masks)  # (n, n-1, h, w)
            visibility_prior_data['masks'] = masks

        if self.configs['data_loader']['visibility_prior']['load_weights']:
            weights = []
            weights_dirname = self.configs['data_loader']['visibility_prior']['weights_dirname']
            frame1_nums = data_dict['frame_nums']
            for frame1_num in frame1_nums:
                frame2_nums = [x for x in frame1_nums if x != frame1_num]
                frame1_weights = []
                for frame2_num in frame2_nums:
                    weight_path = self.data_dirpath / f'test/visibility_masks/{weights_dirname}/{self.scene_num:05}/visibility_weights/{frame1_num:04}_{frame2_num:04}.npy'
                    print(f'Loading visibility prior weight: {weight_path.as_posix()}')
                    weight = self.read_npy_file(weight_path)
                    frame1_weights.append(weight)
                weights.append(frame1_weights)
            weights = numpy.array(weights)  # (n, n-1, h, w)
            visibility_prior_data['weights'] = weights
        return visibility_prior_data

    @staticmethod
    def read_image(path: Path, mmap_mode: str = None):
        if path.suffix in ['.png']:
            image = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            image = numpy.load(path.as_posix(), mmap_mode=mmap_mode)
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return image

    @staticmethod
    def read_mask(path: Path, mmap_mode: str = None):
        if path.suffix in ['.png']:
            mask = skimage.io.imread(path.as_posix()) == 255
        elif path.suffix == '.npy':
            mask = numpy.load(path.as_posix(), mmap_mode=mmap_mode)
        else:
            raise RuntimeError(f'Unknown mask format: {path.as_posix()}')
        return mask

    @staticmethod
    def read_npy_file(path: Path, mmap_mode: str = None):
        if path.suffix == '.npy':
            data = numpy.load(path.as_posix(), mmap_mode=mmap_mode)
        else:
            raise RuntimeError(f'Unknown data format: {path.as_posix()}')
        return data
