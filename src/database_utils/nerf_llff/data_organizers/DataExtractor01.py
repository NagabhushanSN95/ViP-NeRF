# Shree KRISHNAya Namaha
# Extracts camera intrinsics and extrisics
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import datetime
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from zipfile import ZipFile

import numpy
import pandas
import skimage.io
from tqdm import tqdm

import prior_generators.sparse_depth.llff.poses.colmap_read_model as read_model

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def read_colmap_data(data_dirpath):
    camerasfile = os.path.join(data_dirpath, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)

    camera_intrinsics = {}
    for key in camdata.keys():
        cam = camdata[key]
        h, w, f = cam.height, cam.width, cam.params[0]
        intrinsics = numpy.eye(3)
        intrinsics[0, 0] = f
        intrinsics[1, 1] = f
        intrinsics[0, 2] = w / 2
        intrinsics[1, 2] = h / 2
        camera_intrinsics[key] = intrinsics

    bounds_file = os.path.join(data_dirpath, 'poses_bounds.npy')
    bounds = numpy.load(bounds_file)[:, 15:17]

    images_filepath = os.path.join(data_dirpath, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(images_filepath)

    frames_data = []
    bottom = numpy.array([0, 0, 0, 1.]).reshape([1, 4])
    for i, k in enumerate(imdata):
        im = imdata[k]
        name = im.name[:-4]
        intrinsic = camera_intrinsics[im.camera_id]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        extrinsic = numpy.concatenate([numpy.concatenate([R, t], 1), bottom], 0)
        frames_data.append([name, intrinsic.ravel(), extrinsic.ravel(), bounds[i]])
    return frames_data


def extract_scene_data(scene_dirpath):
    scene_name = scene_dirpath.name
    camera_data = read_colmap_data(scene_dirpath)

    names_mapping, intrinsics, extrinsics, bounds = [], [], [], []
    for frame_num in tqdm(range(len(camera_data)), desc=scene_name):
        old_frame_name, intrinsic, extrinsic, bound = camera_data[frame_num]
        names_mapping.append([old_frame_name, frame_num])
        intrinsics.append(intrinsic)
        extrinsics.append(extrinsic)
        bounds.append(bound)

        old_filepath = next(scene_dirpath.glob(f'images/{old_frame_name}.*'))
        new_filepath = scene_dirpath / f'rgb/{frame_num:04}.png'
        if not old_filepath.exists():
            print(f'{old_filepath.as_posix()} does not exist!')
            sys.exit(1)
        new_filepath.parent.mkdir(parents=True, exist_ok=True)
        # shutil.copy(old_filepath, new_filepath)
        image = skimage.io.imread(old_filepath.as_posix())
        skimage.io.imsave(new_filepath.as_posix(), image)

        old_filepath = sorted(filter(lambda path: path.is_file(), (scene_dirpath / 'images_4').iterdir()))[frame_num]
        new_filepath = scene_dirpath / f'rgb_down4/{frame_num:04}.png'
        if not old_filepath.exists():
            print(f'{old_filepath.as_posix()} does not exist!')
            sys.exit(1)
        new_filepath.parent.mkdir(parents=True, exist_ok=True)
        # shutil.copy(old_filepath, new_filepath)
        image = skimage.io.imread(old_filepath.as_posix())
        skimage.io.imsave(new_filepath.as_posix(), image)

        # old_filepath = sorted((scene_dirpath / 'images_8').iterdir())[frame_num]
        old_filepath = sorted(filter(lambda path: path.is_file(), (scene_dirpath / 'images_8').iterdir()))[frame_num]
        new_filepath = scene_dirpath / f'rgb_down8/{frame_num:04}.png'
        if not old_filepath.exists():
            print(f'{old_filepath.as_posix()} does not exist!')
            sys.exit(1)
        new_filepath.parent.mkdir(parents=True, exist_ok=True)
        # shutil.copy(old_filepath, new_filepath)
        image = skimage.io.imread(old_filepath.as_posix())
        skimage.io.imsave(new_filepath.as_posix(), image)
    
    names_mapping_data = pandas.DataFrame(names_mapping, columns=['OldFrameName', 'NewFrameNum'])
    names_mapping_path = scene_dirpath / 'FrameNamesMapping.csv'
    names_mapping_data.to_csv(names_mapping_path, index=False)
    
    intrinsics_array = numpy.stack(intrinsics).reshape(-1, 9)
    intrinsics_path = scene_dirpath / 'CameraIntrinsics.csv'
    numpy.savetxt(intrinsics_path, intrinsics_array, delimiter=',')

    intrinsics_array1 = intrinsics_array.copy()
    intrinsics_array1[:, 0] /= 4
    intrinsics_array1[:, 4] /= 4
    intrinsics_array1[:, 2] /= 4
    intrinsics_array1[:, 5] /= 4
    intrinsics_path = scene_dirpath / 'CameraIntrinsics_down4.csv'
    numpy.savetxt(intrinsics_path, intrinsics_array1, delimiter=',')

    intrinsics_array1 = intrinsics_array.copy()
    intrinsics_array1[:, 0] /= 8
    intrinsics_array1[:, 4] /= 8
    intrinsics_array1[:, 2] /= 8
    intrinsics_array1[:, 5] /= 8
    intrinsics_path = scene_dirpath / 'CameraIntrinsics_down8.csv'
    numpy.savetxt(intrinsics_path, intrinsics_array1, delimiter=',')

    extrinsics_array = numpy.stack(extrinsics).reshape(-1, 16)
    extrinsics_path = scene_dirpath / 'CameraExtrinsics.csv'
    numpy.savetxt(extrinsics_path, extrinsics_array, delimiter=',')

    bounds_array = numpy.stack(bounds)
    bounds_path = scene_dirpath / 'DepthBounds.csv'
    numpy.savetxt(bounds_path, bounds_array, delimiter=',')
    return


def unzip_data(zip_filepath: Path, database_data_dirpath: Path):
    print('Unzipping data...')
    database_data_dirpath.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_filepath, 'r') as zip_file:
        zip_file.extractall(database_data_dirpath.parent)
    shutil.move(database_data_dirpath.parent / 'nerf_llff_data', database_data_dirpath)
    print('Unzipping done.')
    return


def extract_data(database_data_dirpath: Path):
    print('Extracting data...')
    for scene_dirpath in sorted(database_data_dirpath.iterdir()):
        extract_scene_data(scene_dirpath)
    print('Extracting done.')
    return


def main():
    root_dirpath = Path('../../../../')
    database_dirpath = root_dirpath / 'data/databases/NeRF_LLFF/data/'
    zip_filepath = database_dirpath / f'nerf_llff_data.zip'
    database_data_dirpath = database_dirpath / 'all/database_data'

    unzip_data(zip_filepath, database_data_dirpath)
    extract_data(database_data_dirpath)
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
