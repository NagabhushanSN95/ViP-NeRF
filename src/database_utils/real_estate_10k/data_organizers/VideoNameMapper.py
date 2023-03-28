# Shree KRISHNAya Namaha
# 
# Author: Nagabhushan S N
# Last Modified: 29/03/2023
import datetime
import shutil
import time
import traceback
from pathlib import Path

import pandas
from tqdm import tqdm

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def extract_scene_data(unzipped_dirpath: Path, extracted_dirpath: Path, split_name: str):
    mapping = []
    src_dirpath = unzipped_dirpath / split_name
    tgt_dirpath = extracted_dirpath / split_name
    for scene_num, src_data_path in enumerate(tqdm(sorted(src_dirpath.iterdir()))):
        tgt_data_path = tgt_dirpath / f'{scene_num:05}/CameraData{src_data_path.suffix}'
        tgt_data_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_data_path, tgt_data_path)
        mapping.append([src_data_path.stem, tgt_data_path.stem])
    mapping_data = pandas.DataFrame(mapping, columns=['OriginalVideoName', 'NewVideoName'])
    mapping_data_path = extracted_dirpath / f'{split_name}ScenesNameMapping.csv'
    mapping_data.to_csv(mapping_data_path, index=False)
    return


def demo1():
    root_dirpath = Path('../../../../')
    unzipped_dirpath = root_dirpath / 'data/unzipped_data'
    extracted_dirpath = root_dirpath / 'data/extracted_data'

    # extract_scene_data(unzipped_dirpath, extracted_dirpath, split_name='train')
    extract_scene_data(unzipped_dirpath, extracted_dirpath, split_name='test')
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
