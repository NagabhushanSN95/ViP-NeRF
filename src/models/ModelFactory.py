# Shree KRISHNAya Namaha
# A Factory method that returns a Model
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import importlib.util
import inspect


def get_model(configs: dict, model_configs: dict = None):
    filename = configs['model']['name']
    classname = f'{filename[:-2]}'
    model = None
    module = importlib.import_module(f'models.{filename}')
    candidate_classes = inspect.getmembers(module, inspect.isclass)
    for candidate_class in candidate_classes:
        if candidate_class[0] == classname:
            model = candidate_class[1](configs, model_configs)
            break
    if model is None:
        raise RuntimeError(f'Unknown model: {filename}')
    return model
