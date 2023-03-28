# Shree KRISHNAya Namaha
# SSIM measure between predicted frames and ground truth frames
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import argparse
import datetime
import json
import time
import traceback
from pathlib import Path

import numpy
import pandas
import simplejson
import skimage.io
import skimage.transform
from skimage.metrics import structural_similarity
from tqdm import tqdm

this_filepath = Path(__file__)
this_filename = Path(__file__).stem
this_metric_name = this_filename[:-4]


class SSIM:
    def __init__(self, frames_data: pandas.DataFrame, verbose_log: bool = True) -> None:
        super().__init__()
        self.frames_data = frames_data
        self.verbose_log = verbose_log
        return

    @staticmethod
    def compute_frame_ssim(gt_frame: numpy.ndarray, eval_frame: numpy.ndarray):
        gt_frame = gt_frame
        eval_frame = eval_frame
        ssim = structural_similarity(gt_frame, eval_frame, multichannel=True, gaussian_weights=True, sigma=1.5,
                                            use_sample_covariance=False)
        return ssim

    def compute_avg_ssim(self, old_data: pandas.DataFrame, database_dirpath: Path, pred_videos_dirpath: Path,
                         pred_folder_name: str, downsampling_factor: int):
        """

        :param old_data:
        :param database_dirpath: Should be path to databases/OursBlender/Data
        :param pred_videos_dirpath:
        :param pred_folder_name:
        :param downsampling_factor:
        :return:
        """
        qa_scores = []
        for i, frame_data in tqdm(self.frames_data.iterrows(), total=self.frames_data.shape[0], leave=self.verbose_log):
            scene_num, pred_frame_num = frame_data
            if old_data is not None and old_data.loc[
                (old_data['scene_num'] == scene_num) & (old_data['pred_frame_num'] == pred_frame_num)
            ].size > 0:
                continue
            gt_frame_path = database_dirpath / f'all/database_data/{scene_num:05}/rgb/{pred_frame_num:04}.png'
            pred_frame_path = pred_videos_dirpath / f'{scene_num:05}/{pred_folder_name}/{pred_frame_num:04}.png'
            if not pred_frame_path.exists():
                continue
            gt_frame = self.read_image(gt_frame_path)
            pred_frame = self.read_image(pred_frame_path)
            if downsampling_factor > 1:
                gt_frame = self.downsample_image(gt_frame, downsampling_factor)
            qa_score = self.compute_frame_ssim(gt_frame, pred_frame)
            qa_scores.append([scene_num, pred_frame_num, qa_score])
        qa_scores_data = pandas.DataFrame(qa_scores, columns=['scene_num', 'pred_frame_num', this_metric_name])

        merged_data = self.update_qa_frame_data(old_data, qa_scores_data)
        merged_data = merged_data.round({this_metric_name: 4, })

        avg_ssim = numpy.mean(merged_data[this_metric_name])
        if isinstance(avg_ssim, numpy.ndarray):
            avg_ssim = avg_ssim.item()
        avg_ssim = numpy.round(avg_ssim, 4)
        return avg_ssim, merged_data

    @staticmethod
    def update_qa_frame_data(old_data: pandas.DataFrame, new_data: pandas.DataFrame):
        if (old_data is not None) and (new_data.size > 0):
            old_data = old_data.copy()
            new_data = new_data.copy()
            old_data.set_index(['scene_num', 'pred_frame_num'], inplace=True)
            new_data.set_index(['scene_num', 'pred_frame_num'], inplace=True)
            merged_data = old_data.combine_first(new_data).reset_index()
        elif old_data is not None:
            merged_data = old_data
        else:
            merged_data = new_data
        return merged_data

    @staticmethod
    def read_image(path: Path):
        image = skimage.io.imread(path.as_posix())
        return image

    @staticmethod
    def downsample_image(image: numpy.ndarray, downsampling_factor: int):
        downsampled_image = skimage.transform.rescale(image, scale=1 / downsampling_factor, preserve_range=True,
                                                      multichannel=True, anti_aliasing=True)
        downsampled_image = numpy.round(downsampled_image).astype('uint8')
        return downsampled_image


# noinspection PyUnusedLocal
def start_qa(pred_videos_dirpath: Path, database_dirpath: Path, frames_datapath: Path, pred_folder_name: str,
             downsampling_factor: int):
    if not pred_videos_dirpath.exists():
        print(f'Skipping QA of folder: {pred_videos_dirpath.stem}. Reason: pred_videos_dirpath does not exist')
        return

    if not database_dirpath.exists():
        print(f'Skipping QA of folder: {pred_videos_dirpath.stem}. Reason: database_dirpath does not exist')
        return

    qa_scores_filepath = pred_videos_dirpath / 'QA_Scores.json'
    ssim_data_path = pred_videos_dirpath / f'QA_Scores/{pred_folder_name}/{this_metric_name}_FrameWise.csv'
    if qa_scores_filepath.exists():
        with open(qa_scores_filepath.as_posix(), 'r') as qa_scores_file:
            qa_scores = json.load(qa_scores_file)
    else:
        qa_scores = {}

    if pred_folder_name in qa_scores:
        if this_metric_name in qa_scores[pred_folder_name]:
            avg_ssim = qa_scores[pred_folder_name][this_metric_name]
            print(f'Average {this_metric_name}: {pred_videos_dirpath.as_posix()} - {pred_folder_name}: {avg_ssim}')
            print('Running QA again.')
    else:
        qa_scores[pred_folder_name] = {}

    if ssim_data_path.exists():
        ssim_data = pandas.read_csv(ssim_data_path)
    else:
        ssim_data = None

    frames_data = pandas.read_csv(frames_datapath)[['scene_num', 'pred_frame_num']]

    mse_computer = SSIM(frames_data)
    avg_ssim, ssim_data = mse_computer.compute_avg_ssim(ssim_data, database_dirpath, pred_videos_dirpath,
                                                        pred_folder_name, downsampling_factor)
    qa_scores[pred_folder_name][this_metric_name] = avg_ssim
    print(f'Average {this_metric_name}: {pred_videos_dirpath.as_posix()} - {pred_folder_name}: {avg_ssim}')
    with open(qa_scores_filepath.as_posix(), 'w') as qa_scores_file:
        simplejson.dump(qa_scores, qa_scores_file, indent=4)
    ssim_data_path.parent.mkdir(parents=True, exist_ok=True)
    ssim_data.to_csv(ssim_data_path, index=False)
    return avg_ssim


def demo1():
    root_dirpath = Path('../../../../')
    pred_videos_dirpath = root_dirpath / 'runs/testing/test0041'
    database_dirpath = root_dirpath / 'data/databases/DTU/data'
    frames_data_path = database_dirpath / 'train_test_sets/set02/TestVideosData.csv'
    pred_folder_name = 'predicted_frames'
    downsampling_factor = 1
    avg_ssim = start_qa(pred_videos_dirpath, database_dirpath, frames_data_path, pred_folder_name, downsampling_factor)
    return avg_ssim


def demo2(args: dict):
    pred_videos_dirpath = args['pred_videos_dirpath']
    if pred_videos_dirpath is None:
        raise RuntimeError(f'Please provide pred_videos_dirpath')
    pred_videos_dirpath = Path(pred_videos_dirpath)

    database_dirpath = args['database_dirpath']
    if database_dirpath is None:
        raise RuntimeError(f'Please provide database_dirpath')
    database_dirpath = Path(database_dirpath)

    frames_datapath = args['frames_datapath']
    if frames_datapath is None:
        raise RuntimeError(f'Please provide frames_datapath')
    frames_datapath = Path(frames_datapath)

    pred_folder_name = args['pred_folder_name']
    downsampling_factor = args['downsampling_factor']

    avg_ssim = start_qa(pred_videos_dirpath, database_dirpath, frames_datapath, pred_folder_name, downsampling_factor)
    return avg_ssim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_function_name', default='demo1')
    parser.add_argument('--pred_videos_dirpath')
    parser.add_argument('--database_dirpath')
    parser.add_argument('--frames_datapath')
    parser.add_argument('--pred_folder_name', default='predicted_frames')
    parser.add_argument('--downsampling_factor', type=int, default=1)
    args = parser.parse_args()

    args_dict = {
        'demo_function_name': args.demo_function_name,
        'pred_videos_dirpath': args.pred_videos_dirpath,
        'database_dirpath': args.database_dirpath,
        'frames_datapath': args.frames_datapath,
        'pred_folder_name': args.pred_folder_name,
        'downsampling_factor': args.downsampling_factor,
    }
    return args_dict


def main(args: dict):
    if args['demo_function_name'] == 'demo1':
        avg_ssim = demo1()
    elif args['demo_function_name'] == 'demo2':
        avg_ssim = demo2(args)
    else:
        raise RuntimeError(f'Unknown demo function: {args["demo_function_name"]}')
    return avg_ssim


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    args = parse_args()
    try:
        output_score = main(args)
        run_result = f'Program completed successfully!\nAverage {this_metric_name}: {output_score}'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = "Error: " + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
