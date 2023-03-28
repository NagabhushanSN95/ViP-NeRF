# Shree KRISHNAya Namaha
# Groups QA scores scene-wise
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import datetime
import time
import traceback
from pathlib import Path

import pandas

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def get_grouped_qa_scores(qa_data: pandas.DataFrame):
    final_column_names = [x for x in qa_data.columns if x != 'pred_frame_num']
    group_column_names = list(qa_data)[:-2]
    grouped_qa_data = qa_data.groupby(by=group_column_names).mean().reset_index()[final_column_names]
    grouped_qa_data = grouped_qa_data.round({final_column_names[-1]: 4, })
    return grouped_qa_data


def group_qa_scores(testing_dirpath: Path, test_nums: list):
    for test_num in test_nums:
        qa_dirpath = testing_dirpath / f'test{test_num:04}/QA_Scores'
        for pred_dirpath in sorted(qa_dirpath.iterdir()):
            for qa_filepath in sorted(pred_dirpath.glob('*_FrameWise.csv')):
                qa_data = pandas.read_csv(qa_filepath)
                grouped_qa_data = get_grouped_qa_scores(qa_data)
                grouped_qa_filepath = qa_filepath.parent / f'{qa_filepath.stem[:-9]}SceneWise.csv'
                grouped_qa_data.to_csv(grouped_qa_filepath, index=False)
    return


def demo1():
    # testing_dirpath = Path('../../../ViewSynthesis/Literature/015_DDP_NeRF/runs/testing')
    # test_nums = [1,2,3,4,5,11,12,13,14,42,43,44]
    testing_dirpath = Path('../../../ViewSynthesis/Research/006_NeRF/runs/testing')
    test_nums = [1255]
    group_qa_scores(testing_dirpath, test_nums)
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
