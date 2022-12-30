# Shree KRISHNAya Namaha
# Common Utility Functions
# Author: Nagabhushan S N
# Last Modified: 15/02/2022

from pathlib import Path
from typing import Union

from matplotlib import pyplot

import torch

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def get_device(device: str):
    """
    Returns torch device object
    :param device: cpu/gpu0/gpu1
    :return:
    """
    if device == 'cpu':
        device = torch.device('cpu')
    elif device.startswith('gpu') and torch.cuda.is_available():
        gpu_num = int(device[3:])
        device = torch.device(f'cuda:{gpu_num}')
    else:
        device = torch.device('cpu')
    return device


def move_to_device(tensor_data: Union[torch.Tensor, list, dict], device):
    if isinstance(tensor_data, torch.Tensor):
        moved_tensor_data = tensor_data.to(device, non_blocking=True)
    elif isinstance(tensor_data, list):
        moved_tensor_data = []
        for tensor_elem in tensor_data:
            moved_tensor_data.append(move_to_device(tensor_elem, device))
    elif isinstance(tensor_data, dict):
        moved_tensor_data = {}
        for key in tensor_data:
            moved_tensor_data[key] = move_to_device(tensor_data[key], device)
    else:
        moved_tensor_data = tensor_data
    return moved_tensor_data
