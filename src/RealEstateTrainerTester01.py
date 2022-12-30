# Shree KRISHNAya Namaha
# Runs both training and testing on RealEstate dataset.
# Author: Nagabhushan S N
# Last Modified: 30/12/2022

import datetime
import os
import time
import traceback
from pathlib import Path

import numpy
import pandas
import skimage.io
import skvideo.io

import Tester01 as Tester
import Trainer01 as Trainer

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def read_image(path: Path):
    image = skimage.io.imread(path.as_posix())
    return image


def save_video(path: Path, video: numpy.ndarray):
    if path.exists():
        return
    try:
        skvideo.io.vwrite(path.as_posix(), video,
                          inputdict={'-r': str(15)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p'}, verbosity=1)
    except (NameError , OSError):
        pass
    return


def start_training(train_configs: dict):
    root_dirpath = Path('../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / train_configs['database_dirpath']

    # Setup output dirpath
    output_dirpath = root_dirpath / f'Runs/Training/Train{train_configs["train_num"]:04}'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    scene_nums = train_configs['data_loader'].get('scene_nums', None)
    Trainer.save_configs(output_dirpath, train_configs)
    train_configs['data_loader']['scene_nums'] = scene_nums

    if train_configs['data_loader']['scene_nums'] is None:
        set_num = train_configs['data_loader']['train_set_num']
        video_datapath = database_dirpath / f'TrainTestSets/Set{set_num:02}/TrainVideosData.csv'
        video_data = pandas.read_csv(video_datapath)
        scene_nums = numpy.unique(video_data['scene_num'].to_numpy())
    scene_ids = [f'{scene_num:05}' for scene_num in scene_nums]
    train_configs['data_loader']['scene_ids'] = scene_ids
    Trainer.start_training(train_configs)
    return


def start_testing(test_configs: dict):
    root_dirpath = Path('../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'Databases' / test_configs['database_dirpath']

    output_dirpath = root_dirpath / f"Runs/Testing/Test{test_configs['test_num']:04}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    Tester.save_configs(output_dirpath, test_configs)

    set_num = test_configs['test_set_num']
    train_video_datapath = database_dirpath / f'TrainTestSets/Set{set_num:02}/TrainVideosData.csv'
    test_video_datapath = database_dirpath / f'TrainTestSets/Set{set_num:02}/TestVideosData.csv'
    train_video_data = pandas.read_csv(train_video_datapath)
    test_video_data = pandas.read_csv(test_video_datapath)
    scene_nums = test_configs.get('scene_nums', test_video_data['scene_num'].to_numpy())
    scene_nums = numpy.unique(scene_nums)
    scenes_data = {}
    for scene_num in scene_nums:
        scene_id = f'{scene_num:05}'
        scenes_data[scene_id] = {
            'output_dirname': scene_id,
            'frames_data': {}
        }

        extrinsics_path = database_dirpath / f'test/DatabaseData/{scene_id}/CameraExtrinsics.csv'
        extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))
        # Intrinsics and frames required to compute plane sweep volume for conv visibility prediction
        intrinsics_path = database_dirpath / f'test/DatabaseData/{scene_id}/CameraIntrinsics.csv'
        intrinsics = numpy.loadtxt(intrinsics_path.as_posix(), delimiter=',').reshape((-1, 3, 3))
        frames_dirpath = database_dirpath / f'test/DatabaseData/{scene_id}/rgb'

        test_frame_nums = test_video_data.loc[test_video_data['scene_num'] == scene_num]['pred_frame_num'].to_list()
        train_frame_nums = train_video_data.loc[train_video_data['scene_num'] == scene_num]['pred_frame_num'].to_list()
        frame_nums = numpy.unique(sorted(test_frame_nums + train_frame_nums))
        for frame_num in frame_nums:
            frame_path = frames_dirpath / f'{frame_num:04}.png'
            frame = read_image(frame_path)
            scenes_data[scene_id]['frames_data'][frame_num] = {
                'extrinsic': extrinsics[frame_num],
                'intrinsic': intrinsics[frame_num],
                'is_train_frame': frame_num in train_frame_nums,
                'frame': frame,
            }
    Tester.start_testing(test_configs, scenes_data, save_depth=True, save_depth_var=True, save_visibility=True)

    # Run QA
    qa_filepath = Path('../../../../QA/00_Common/src/AllMetrics02_RealEstate.py')
    cmd = f'python {qa_filepath.absolute().as_posix()} ' \
          f'--demo_function_name demo2 ' \
          f'--pred_videos_dirpath {output_dirpath.absolute().as_posix()} ' \
          f'--database_dirpath {database_dirpath.absolute().as_posix()} ' \
          f'--frames_datapath {test_video_datapath.absolute().as_posix()} ' \
          f'--pred_folder_name PredictedFrames'
    os.system(cmd)
    return


def start_testing_videos(test_configs: dict):
    root_dirpath = Path('../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'Databases' / test_configs['database_dirpath']

    output_dirpath = root_dirpath / f"Runs/Testing/Test{test_configs['test_num']:04}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    Tester.save_configs(output_dirpath, test_configs)

    set_num = test_configs['test_set_num']
    video_datapath = database_dirpath / f'TrainTestSets/Set{set_num:02}/TestVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_nums = test_configs.get('scene_nums', video_data['scene_num'].to_numpy())
    scene_nums = numpy.unique(scene_nums)

    videos_data = [1, 2, 3, ]
    for video_num in videos_data:
        video_frame_nums_path = database_dirpath / f'TrainTestSets/Set{set_num:02}/VideoPoses{video_num:02}/VideoFrameNums.csv'
        if video_frame_nums_path.exists():
            video_frame_nums = numpy.loadtxt(video_frame_nums_path.as_posix(), delimiter=',').astype(int)
        else:
            video_frame_nums = None
        for scene_num in scene_nums:
            scenes_data = {}
            scene_id = f'{scene_num:05}'
            scenes_data[scene_id] = {
                'output_dirname': scene_id,
                'frames_data': {}
            }

            extrinsics_path = database_dirpath / f'TrainTestSets/Set{set_num:02}/VideoPoses{video_num:02}/{scene_id}.csv'
            if not extrinsics_path.exists():
                continue
            extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))

            frame_nums = numpy.arange(extrinsics.shape[0] - 1)
            for frame_num in frame_nums:
                scenes_data[scene_id]['frames_data'][frame_num] = {
                    'extrinsic': extrinsics[frame_num + 1]
                }
            output_dir_suffix = f'_Video{video_num:02}'
            output_dirpath = Tester.start_testing(test_configs, scenes_data, output_dir_suffix)
            scene_output_dirpath = output_dirpath / f'{scene_id}{output_dir_suffix}'
            if not scene_output_dirpath.exists():
                continue
            pred_frames = [read_image(scene_output_dirpath / f'PredictedFrames/{frame_num:04}.png') for frame_num in frame_nums]
            video_frames = numpy.stack(pred_frames)
            if video_frame_nums is not None:
                video_frames = video_frames[video_frame_nums]
            video_output_path = scene_output_dirpath / 'PredictedVideo.mp4'
            save_video(video_output_path, video_frames)
    return


def start_testing_static_videos(test_configs: dict):
    """
    This is for view_dirs visualization
    :param test_configs:
    :return:
    """
    root_dirpath = Path('../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'Databases' / test_configs['database_dirpath']

    output_dirpath = root_dirpath / f"Runs/Testing/Test{test_configs['test_num']:04}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    Tester.save_configs(output_dirpath, test_configs)

    set_num = test_configs['test_set_num']
    video_datapath = database_dirpath / f'TrainTestSets/Set{set_num:02}/TestVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_nums = test_configs.get('scene_nums', video_data['scene_num'].to_numpy())
    scene_nums = numpy.unique(scene_nums)

    videos_data = [1, 2, 3, ]
    for video_num in videos_data:
        video_frame_nums_path = database_dirpath / f'TrainTestSets/Set{set_num:02}/VideoPoses{video_num:02}/VideoFrameNums.csv'
        if video_frame_nums_path.exists():
            video_frame_nums = numpy.loadtxt(video_frame_nums_path.as_posix(), delimiter=',').astype(int)
        else:
            video_frame_nums = None
        for scene_num in scene_nums:
            scenes_data = {}
            scene_id = f'{scene_num:05}'
            scenes_data[scene_id] = {
                'output_dirname': scene_id,
                'frames_data': {}
            }

            extrinsics_path = database_dirpath / f'TrainTestSets/Set{set_num:02}/VideoPoses{video_num:02}/{scene_id}.csv'
            if not extrinsics_path.exists():
                continue
            extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))

            frame_nums = numpy.arange(extrinsics.shape[0] - 1)
            for frame_num in frame_nums:
                scenes_data[scene_id]['frames_data'][frame_num] = {
                    'extrinsic': extrinsics[0],
                    'extrinsic_viewcam': extrinsics[frame_num + 1]
                }
            output_dir_suffix = f'_Video{video_num:02}_StaticCamera'
            output_dirpath = Tester.start_testing(test_configs, scenes_data, output_dir_suffix)
            scene_output_dirpath = output_dirpath / f'{scene_id}{output_dir_suffix}'
            if not scene_output_dirpath.exists():
                continue
            pred_frames = [read_image(scene_output_dirpath / f'PredictedFrames/{frame_num:04}.png') for frame_num in frame_nums]
            video_frames = numpy.stack(pred_frames)
            if video_frame_nums is not None:
                video_frames = video_frames[video_frame_nums]
            video_output_path = scene_output_dirpath / 'StaticCameraVideo.mp4'
            save_video(video_output_path, video_frames)
    return


def demo1a():
    train_num = 1
    test_num = 1
    scene_nums = [0, 1, 3, 4, 6]

    for scene_num in scene_nums:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'database': 'RealEstate10K',
            'database_dirpath': 'Databases/RealEstate10K/Data',
            'data_loader': {
                'data_loader_name': 'RealEstateDataLoader01',
                'data_preprocessor_name': 'DataPreprocessor01',
                'train_set_num': 2,
                'scene_nums': [scene_num],
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 2048,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'visibility_consistency' : {
                    'load_masks': True,
                    'load_weights': False,
                    'masks_dirname': 'VSR006_VW02',
                },
                'sparse_depth': {
                    'dirname': 'DEL003_DE02',
                    'num_rays': 2048,
                },
            },
            'model': {
                'name': 'VipNeRF01',
                'use_coarse_mlp': True,
                'use_fine_mlp': True,
                'num_samples_coarse': 64,
                'num_samples_fine': 128,
                'chunk': 4*1024,
                'lindisp': False,
                'points_positional_encoding_degree': 10,
                'views_positional_encoding_degree': 4,
                'netchunk': 16*1024,
                'netdepth_coarse': 8,
                'netdepth_fine': 8,
                'netwidth_coarse': 256,
                'netwidth_fine': 256,
                'perturb': True,
                'raw_noise_std': 1.0,
                'use_view_dirs': True,
                'view_dependent_rgb': True,
                'white_bkgd': False,
                'predict_visibility': True,
            },
            'losses': [
                {
                    'name': 'MSE08',
                    'weight': 1,
                },
                {
                    'name': 'VisibilityLoss09',
                    'weight': 0.1,
                },
                {
                    'name': 'VisibilityConsistencyLoss44',
                    'iter_weights': {
                        '0': 0, '30000': 0.001,
                    },
                },
                {
                    "name": "SparseDepthMSE07",
                    "weight": 0.1,
                },
            ],
            'optimizer': {
                'lr_decayer_name': 'NeRFLearningRateDecayer01',
                'lr_initial': 5e-4,
                'lr_decay': 250,
                'beta1': 0.9,
                'beta2': 0.999,
            },
            'resume_training': True,
            'num_iterations': 50000,
            'validation_interval': 10000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 10000,
            'mixed_precision_training': False,
            # 'seed': numpy.random.randint(1000),
            'seed': 0,
            'device': 'gpu0',
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 2,
            'train_num': train_num,
            'model_name': 'Model_Iter050000.tar',
            'database_name': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/Data',
            'scene_nums': [scene_num],
            'device': 'gpu0',
        }
        start_training(train_configs)
        start_testing(test_configs)
        start_testing_videos(test_configs)
        start_testing_static_videos(test_configs)
    return


def demo1b():
    train_num = 2
    test_num = 2
    scene_nums = [0, 1, 3, 4, 6]

    for scene_num in scene_nums:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'database': 'RealEstate10K',
            'database_dirpath': 'Databases/RealEstate10K/Data',
            'data_loader': {
                'data_loader_name': 'RealEstateDataLoader01',
                'data_preprocessor_name': 'DataPreprocessor01',
                'train_set_num': 3,
                'scene_nums': [scene_num],
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 2048,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'visibility_consistency' : {
                    'load_masks': True,
                    'load_weights': False,
                    'masks_dirname': 'VSR006_VW03',
                },
                'sparse_depth': {
                    'dirname': 'DEL003_DE03',
                    'num_rays': 2048,
                },
            },
            'model': {
                'name': 'VipNeRF01',
                'use_coarse_mlp': True,
                'use_fine_mlp': True,
                'num_samples_coarse': 64,
                'num_samples_fine': 128,
                'chunk': 4*1024,
                'lindisp': False,
                'points_positional_encoding_degree': 10,
                'views_positional_encoding_degree': 4,
                'netchunk': 16*1024,
                'netdepth_coarse': 8,
                'netdepth_fine': 8,
                'netwidth_coarse': 256,
                'netwidth_fine': 256,
                'perturb': True,
                'raw_noise_std': 1.0,
                'use_view_dirs': True,
                'view_dependent_rgb': True,
                'white_bkgd': False,
                'predict_visibility': True,
            },
            'losses': [
                {
                    'name': 'MSE08',
                    'weight': 1,
                },
                {
                    'name': 'VisibilityLoss09',
                    'weight': 0.1,
                },
                {
                    'name': 'VisibilityConsistencyLoss44',
                    'iter_weights': {
                        '0': 0, '30000': 0.001,
                    },
                },
                {
                    "name": "SparseDepthMSE07",
                    "weight": 0.1,
                },
            ],
            'optimizer': {
                'lr_decayer_name': 'NeRFLearningRateDecayer01',
                'lr_initial': 5e-4,
                'lr_decay': 250,
                'beta1': 0.9,
                'beta2': 0.999,
            },
            'resume_training': True,
            'num_iterations': 50000,
            'validation_interval': 10000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 10000,
            'mixed_precision_training': False,
            # 'seed': numpy.random.randint(1000),
            'seed': 0,
            'device': 'gpu0',
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 3,
            'train_num': train_num,
            'model_name': 'Model_Iter050000.tar',
            'database_name': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/Data',
            'scene_nums': [scene_num],
            'device': 'gpu0',
        }
        start_training(train_configs)
        start_testing(test_configs)
        start_testing_videos(test_configs)
        start_testing_static_videos(test_configs)
    return


def demo1c():
    train_num = 3
    test_num = 3
    scene_nums = [0, 1, 3, 4, 6]

    for scene_num in scene_nums:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'database': 'RealEstate10K',
            'database_dirpath': 'Databases/RealEstate10K/Data',
            'data_loader': {
                'data_loader_name': 'RealEstateDataLoader01',
                'data_preprocessor_name': 'DataPreprocessor01',
                'train_set_num': 4,
                'scene_nums': [scene_num],
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 2048,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'visibility_consistency' : {
                    'load_masks': True,
                    'load_weights': False,
                    'masks_dirname': 'VSR006_VW04',
                },
                'sparse_depth': {
                    'dirname': 'DEL003_DE04',
                    'num_rays': 2048,
                },
            },
            'model': {
                'name': 'VipNeRF01',
                'use_coarse_mlp': True,
                'use_fine_mlp': True,
                'num_samples_coarse': 64,
                'num_samples_fine': 128,
                'chunk': 4*1024,
                'lindisp': False,
                'points_positional_encoding_degree': 10,
                'views_positional_encoding_degree': 4,
                'netchunk': 16*1024,
                'netdepth_coarse': 8,
                'netdepth_fine': 8,
                'netwidth_coarse': 256,
                'netwidth_fine': 256,
                'perturb': True,
                'raw_noise_std': 1.0,
                'use_view_dirs': True,
                'view_dependent_rgb': True,
                'white_bkgd': False,
                'predict_visibility': True,
            },
            'losses': [
                {
                    'name': 'MSE08',
                    'weight': 1,
                },
                {
                    'name': 'VisibilityLoss09',
                    'weight': 0.1,
                },
                {
                    'name': 'VisibilityConsistencyLoss44',
                    'iter_weights': {
                        '0': 0, '30000': 0.001,
                    },
                },
                {
                    "name": "SparseDepthMSE07",
                    "weight": 0.1,
                },
            ],
            'optimizer': {
                'lr_decayer_name': 'NeRFLearningRateDecayer01',
                'lr_initial': 5e-4,
                'lr_decay': 250,
                'beta1': 0.9,
                'beta2': 0.999,
            },
            'resume_training': True,
            'num_iterations': 50000,
            'validation_interval': 10000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 10000,
            'mixed_precision_training': False,
            # 'seed': numpy.random.randint(1000),
            'seed': 0,
            'device': 'gpu0',
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 4,
            'train_num': train_num,
            'model_name': 'Model_Iter050000.tar',
            'database_name': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/Data',
            'scene_nums': [scene_num],
            'device': 'gpu0',
        }
        start_training(train_configs)
        start_testing(test_configs)
        start_testing_videos(test_configs)
        start_testing_static_videos(test_configs)
    return


def demo1d():
    train_num = 1+3
    test_num = 1+3
    scene_nums = [0, 1, 3, 4, 6]

    for scene_num in scene_nums:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'database': 'RealEstate10K',
            'database_dirpath': 'Databases/RealEstate10K/Data',
            'data_loader': {
                'data_loader_name': 'RealEstateDataLoader01',
                'data_preprocessor_name': 'DataPreprocessor01',
                'train_set_num': 2,
                'scene_nums': [scene_num],
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 1024,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'visibility_consistency' : {
                    'load_masks': True,
                    'load_weights': False,
                    'masks_dirname': 'VSR006_VW02',
                },
            },
            'model': {
                'name': 'VipNeRF01',
                'use_coarse_mlp': True,
                'use_fine_mlp': True,
                'num_samples_coarse': 64,
                'num_samples_fine': 128,
                'chunk': 4*1024,
                'lindisp': False,
                'points_positional_encoding_degree': 10,
                'views_positional_encoding_degree': 4,
                'netchunk': 16*1024,
                'netdepth_coarse': 8,
                'netdepth_fine': 8,
                'netwidth_coarse': 256,
                'netwidth_fine': 256,
                'perturb': True,
                'raw_noise_std': 1.0,
                'use_view_dirs': True,
                'view_dependent_rgb': True,
                'white_bkgd': False,
                'predict_visibility': True,
            },
            'losses': [
                {
                    'name': 'MSE08',
                    'weight': 1,
                },
                {
                    'name': 'VisibilityLoss09',
                    'weight': 0.1,
                },
                {
                    'name': 'VisibilityConsistencyLoss44',
                    'iter_weights': {
                        '0': 0, '20000': 0.001,
                    },
                },
            ],
            'optimizer': {
                'lr_decayer_name': 'NeRFLearningRateDecayer01',
                'lr_initial': 5e-4,
                'lr_decay': 250,
                'beta1': 0.9,
                'beta2': 0.999,
            },
            'resume_training': True,
            'num_iterations': 50000,
            'validation_interval': 10000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 10000,
            'mixed_precision_training': False,
            'seed': numpy.random.randint(1000),
            'device': 'gpu0',
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 2,
            'train_num': train_num,
            'model_name': 'Model_Iter050000.tar',
            'database_name': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/Data',
            'scene_nums': [scene_num],
            'device': 'gpu0',
        }
        start_training(train_configs)
        start_testing(test_configs)
        start_testing_videos(test_configs)
        start_testing_static_videos(test_configs)
    return


def demo1e():
    train_num = 2+3
    test_num = 2+3
    scene_nums = [0, 1, 3, 4, 6]

    for scene_num in scene_nums:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'database': 'RealEstate10K',
            'database_dirpath': 'Databases/RealEstate10K/Data',
            'data_loader': {
                'data_loader_name': 'RealEstateDataLoader01',
                'data_preprocessor_name': 'DataPreprocessor01',
                'train_set_num': 3,
                'scene_nums': [scene_num],
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 1024,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'visibility_consistency' : {
                    'load_masks': True,
                    'load_weights': False,
                    'masks_dirname': 'VSR006_VW03',
                },
            },
            'model': {
                'name': 'VipNeRF01',
                'use_coarse_mlp': True,
                'use_fine_mlp': True,
                'num_samples_coarse': 64,
                'num_samples_fine': 128,
                'chunk': 4*1024,
                'lindisp': False,
                'points_positional_encoding_degree': 10,
                'views_positional_encoding_degree': 4,
                'netchunk': 16*1024,
                'netdepth_coarse': 8,
                'netdepth_fine': 8,
                'netwidth_coarse': 256,
                'netwidth_fine': 256,
                'perturb': True,
                'raw_noise_std': 1.0,
                'use_view_dirs': True,
                'view_dependent_rgb': True,
                'white_bkgd': False,
                'predict_visibility': True,
            },
            'losses': [
                {
                    'name': 'MSE08',
                    'weight': 1,
                },
                {
                    'name': 'VisibilityLoss09',
                    'weight': 0.1,
                },
                {
                    'name': 'VisibilityConsistencyLoss44',
                    'iter_weights': {
                        '0': 0, '20000': 0.001,
                    },
                },
            ],
            'optimizer': {
                'lr_decayer_name': 'NeRFLearningRateDecayer01',
                'lr_initial': 5e-4,
                'lr_decay': 250,
                'beta1': 0.9,
                'beta2': 0.999,
            },
            'resume_training': True,
            'num_iterations': 50000,
            'validation_interval': 10000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 10000,
            'mixed_precision_training': False,
            'seed': numpy.random.randint(1000),
            'device': 'gpu0',
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 3,
            'train_num': train_num,
            'model_name': 'Model_Iter050000.tar',
            'database_name': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/Data',
            'scene_nums': [scene_num],
            'device': 'gpu0',
        }
        start_training(train_configs)
        start_testing(test_configs)
        start_testing_videos(test_configs)
        start_testing_static_videos(test_configs)
    return


def demo1f():
    train_num = 3+3
    test_num = 3+3
    scene_nums = [0, 1, 3, 4, 6]

    for scene_num in scene_nums:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'database': 'RealEstate10K',
            'database_dirpath': 'Databases/RealEstate10K/Data',
            'data_loader': {
                'data_loader_name': 'RealEstateDataLoader01',
                'data_preprocessor_name': 'DataPreprocessor01',
                'train_set_num': 4,
                'scene_nums': [scene_num],
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 1024,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'visibility_consistency' : {
                    'load_masks': True,
                    'load_weights': False,
                    'masks_dirname': 'VSR006_VW04',
                },
            },
            'model': {
                'name': 'VipNeRF01',
                'use_coarse_mlp': True,
                'use_fine_mlp': True,
                'num_samples_coarse': 64,
                'num_samples_fine': 128,
                'chunk': 4*1024,
                'lindisp': False,
                'points_positional_encoding_degree': 10,
                'views_positional_encoding_degree': 4,
                'netchunk': 16*1024,
                'netdepth_coarse': 8,
                'netdepth_fine': 8,
                'netwidth_coarse': 256,
                'netwidth_fine': 256,
                'perturb': True,
                'raw_noise_std': 1.0,
                'use_view_dirs': True,
                'view_dependent_rgb': True,
                'white_bkgd': False,
                'predict_visibility': True,
            },
            'losses': [
                {
                    'name': 'MSE08',
                    'weight': 1,
                },
                {
                    'name': 'VisibilityLoss09',
                    'weight': 0.1,
                },
                {
                    'name': 'VisibilityConsistencyLoss44',
                    'iter_weights': {
                        '0': 0, '20000': 0.001,
                    },
                },
            ],
            'optimizer': {
                'lr_decayer_name': 'NeRFLearningRateDecayer01',
                'lr_initial': 5e-4,
                'lr_decay': 250,
                'beta1': 0.9,
                'beta2': 0.999,
            },
            'resume_training': True,
            'num_iterations': 50000,
            'validation_interval': 10000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 10000,
            'mixed_precision_training': False,
            'seed': numpy.random.randint(1000),
            'device': 'gpu0',
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 4,
            'train_num': train_num,
            'model_name': 'Model_Iter050000.tar',
            'database_name': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/Data',
            'scene_nums': [scene_num],
            'device': 'gpu0',
        }
        start_training(train_configs)
        start_testing(test_configs)
        start_testing_videos(test_configs)
        start_testing_static_videos(test_configs)
    return


def demo1g():
    train_num = 4+3
    test_num = 4+3
    scene_nums = [0, 1, 3, 4, 6]

    for scene_num in scene_nums:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'database': 'RealEstate10K',
            'database_dirpath': 'Databases/RealEstate10K/Data',
            'data_loader': {
                'data_loader_name': 'RealEstateDataLoader01',
                'data_preprocessor_name': 'DataPreprocessor01',
                'train_set_num': 2,
                'scene_nums': [scene_num],
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 2048,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'sparse_depth': {
                    'dirname': 'DEL003_DE02',
                    'num_rays': 2048,
                },
                'dense_depth': {
                    'dirname': 'VSR006_DE02',
                },
            },
            'model': {
                'name': 'VipNeRF01',
                'use_coarse_mlp': True,
                'use_fine_mlp': True,
                'num_samples_coarse': 64,
                'num_samples_fine': 128,
                'chunk': 4*1024,
                'lindisp': False,
                'points_positional_encoding_degree': 10,
                'views_positional_encoding_degree': 4,
                'netchunk': 16*1024,
                'netdepth_coarse': 8,
                'netdepth_fine': 8,
                'netwidth_coarse': 256,
                'netwidth_fine': 256,
                'perturb': True,
                'raw_noise_std': 1.0,
                'use_view_dirs': True,
                'view_dependent_rgb': True,
                'white_bkgd': False,
                'predict_visibility': False,
            },
            'losses': [
                {
                    'name': 'MSE08',
                    'weight': 1,
                },
                {
                    "name": "SparseDepthMSE07",
                    "weight": 0.1,
                },
                {
                    "name": "DenseDepthMSE04",
                    "weight": 0.1,
                },
            ],
            'optimizer': {
                'lr_decayer_name': 'NeRFLearningRateDecayer01',
                'lr_initial': 5e-4,
                'lr_decay': 250,
                'beta1': 0.9,
                'beta2': 0.999,
            },
            'resume_training': True,
            'num_iterations': 50000,
            'validation_interval': 10000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 10000,
            'mixed_precision_training': False,
            'seed': numpy.random.randint(1000),
            'device': 'gpu0',
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 2,
            'train_num': train_num,
            'model_name': 'Model_Iter050000.tar',
            'database_name': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/Data',
            'scene_nums': [scene_num],
            'device': 'gpu0',
        }
        start_training(train_configs)
        start_testing(test_configs)
        start_testing_videos(test_configs)
        start_testing_static_videos(test_configs)
    return


def demo1h():
    train_num = 5+3
    test_num = 5+3
    scene_nums = [0, 1, 3, 4, 6]

    for scene_num in scene_nums:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'database': 'RealEstate10K',
            'database_dirpath': 'Databases/RealEstate10K/Data',
            'data_loader': {
                'data_loader_name': 'RealEstateDataLoader01',
                'data_preprocessor_name': 'DataPreprocessor01',
                'train_set_num': 1,
                'scene_nums': [scene_num],
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 1024,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
            },
            'model': {
                'name': 'VipNeRF01',
                'use_coarse_mlp': True,
                'use_fine_mlp': True,
                'num_samples_coarse': 64,
                'num_samples_fine': 128,
                'chunk': 4*1024,
                'lindisp': False,
                'points_positional_encoding_degree': 10,
                'views_positional_encoding_degree': 4,
                'netchunk': 16*1024,
                'netdepth_coarse': 8,
                'netdepth_fine': 8,
                'netwidth_coarse': 256,
                'netwidth_fine': 256,
                'perturb': True,
                'raw_noise_std': 1.0,
                'use_view_dirs': True,
                'view_dependent_rgb': True,
                'white_bkgd': False,
                'predict_visibility': True,
            },
            'losses': [
                {
                    'name': 'MSE08',
                    'weight': 1,
                },
                {
                    'name': 'VisibilityLoss09',
                    'weight': 0.1,
                },
            ],
            'optimizer': {
                'lr_decayer_name': 'NeRFLearningRateDecayer01',
                'lr_initial': 5e-4,
                'lr_decay': 250,
                'beta1': 0.9,
                'beta2': 0.999,
            },
            'resume_training': True,
            'num_iterations': 300000,
            'validation_interval': 10000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 50000,
            'mixed_precision_training': False,
            'seed': numpy.random.randint(1000),
            'device': 'gpu0',
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 1,
            'train_num': train_num,
            'model_name': 'Model_Iter300000.tar',
            'database_name': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/Data',
            'scene_nums': [scene_num],
            'device': 'gpu0',
        }
        start_training(train_configs)
        start_testing(test_configs)
        start_testing_videos(test_configs)
        start_testing_static_videos(test_configs)
    return


def demo2():
    configs = {
        'trainer': f'{this_filename}/{Trainer.this_filename}',
        'train_num': 2,
        'resume_training': True,
    }
    start_training(configs)
    return


def demo3():
    """
    Saves plots mid training
    :return:
    """
    train_num = 1
    scene_num = 0
    loss_plots_dirpath = Path(f'../Runs/Training/Train{train_num:04}/{scene_num:05}/Logs')
    Trainer.save_plots(loss_plots_dirpath)
    import sys
    sys.exit(0)


def demo4():
    for train_num in [2, ]:
        test_num = train_num
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 2,
            'train_num': train_num,
            'model_name': 'Model_Iter050000.tar',
            'database_name': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/Data',
            'device': 'gpu0',
        }
        start_testing(test_configs)
        start_testing_videos(test_configs)
        start_testing_static_videos(test_configs)
    return


def main():
    demo1a()
    demo1b()
    demo1c()
    demo1d()
    demo1e()
    demo1f()
    demo1g()
    demo1h()
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
