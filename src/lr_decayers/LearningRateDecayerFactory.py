# Shree KRISHNAya Namaha
# A Factory method that returns a Learning Rate Decayer
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import importlib.util
import inspect

from lr_decayers.LearningRateDecayerParent import LearningRateDecayerParent


def get_lr_decayer(configs: dict) -> LearningRateDecayerParent:
    filename = configs['optimizer']['lr_decayer_name']
    classname = filename[:-2]
    data_preprocessor = None
    module = importlib.import_module(f'lr_decayers.{filename}')
    candidate_classes = inspect.getmembers(module, inspect.isclass)
    for candidate_class in candidate_classes:
        if candidate_class[0] == classname:
            data_preprocessor = candidate_class[1](configs)
            break
    if data_preprocessor is None:
        raise RuntimeError(f'Unknown lr decayer: {filename}')
    return data_preprocessor
