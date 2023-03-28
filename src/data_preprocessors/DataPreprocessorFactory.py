# Shree KRISHNAya Namaha
# A Factory method that returns a Data Loader
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import importlib.util
import inspect
from typing import Optional

from data_preprocessors.DataPreprocessorParent01 import DataPreprocessorParent


def get_data_preprocessor(configs: dict, mode: str, *, raw_data_dict: Optional[dict] = None,
                          model_configs: Optional[dict] = None) -> DataPreprocessorParent:
    filename = configs['data_loader']['data_preprocessor_name']
    classname = filename[:-2]
    data_preprocessor = None
    module = importlib.import_module(f'data_preprocessors.{filename}')
    candidate_classes = inspect.getmembers(module, inspect.isclass)
    for candidate_class in candidate_classes:
        if candidate_class[0] == classname:
            data_preprocessor = candidate_class[1](configs, mode, raw_data_dict, model_configs)
            break
    if data_preprocessor is None:
        raise RuntimeError(f'Unknown data preprocessor: {filename}')
    return data_preprocessor
