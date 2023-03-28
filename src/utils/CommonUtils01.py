# Shree KRISHNAya Namaha
# Common Utility Functions
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

from pathlib import Path
from typing import Union

import torch

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def get_device(device):
    """
    Returns torch device object
    :param device: None//0/[0,],[0,1]. If multiple gpus are specified, first one is chosen
    :return:
    """
    if (device is None) or (device == '') or (not torch.cuda.is_available()):
        device = torch.device('cpu')
    else:
        device0 = device[0] if isinstance(device, list) else device
        device = torch.device(f'cuda:{device0}')
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
