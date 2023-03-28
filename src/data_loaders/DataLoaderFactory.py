# Shree KRISHNAya Namaha
# A Factory method that returns a Data Loader
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import importlib.util
import inspect
from pathlib import Path
from typing import Optional

from data_loaders.DataLoaderParent import DataLoaderParent


def get_data_loader(configs: dict, data_dirpath: Path, mode: Optional[str]) -> DataLoaderParent:
    filename = configs['data_loader']['data_loader_name']
    classname = filename[:-2]
    data_loader = None
    module = importlib.import_module(f'data_loaders.{filename}')
    candidate_classes = inspect.getmembers(module, inspect.isclass)
    for candidate_class in candidate_classes:
        if candidate_class[0] == classname:
            data_loader = candidate_class[1](configs, data_dirpath, mode)
            break
    if data_loader is None:
        raise RuntimeError(f'Unknown data loader: {filename}')
    return data_loader
