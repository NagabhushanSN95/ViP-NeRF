# Shree KRISHNAya Namaha
# Computes all specified losses by calling appropriate functions
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import importlib.util
import inspect

import torch


class LossComputer:
    def __init__(self, configs: dict):
        self.losses = {}
        for loss_configs in configs['losses']:
            loss_name = loss_configs['name']
            self.losses[loss_name] = self.get_loss_object(loss_name, configs, loss_configs)
        return

    @staticmethod
    def get_loss_object(loss_name, configs: dict, loss_configs: dict):
        loss_obj = None
        module = importlib.import_module(f'loss_functions.{loss_name}')
        candidate_classes = inspect.getmembers(module, inspect.isclass)
        for candidate_class in candidate_classes:
            if candidate_class[0] == loss_name[:-2]:
                loss_obj = candidate_class[1](configs, loss_configs)
                break
        if loss_obj is None:
            raise RuntimeError(f'Unknown Loss Function: {loss_name}')
        return loss_obj

    def compute_losses(self, input_dict, output_dict, return_loss_maps: bool = False):
        if 'common_data' in input_dict.keys():
            # unpack common data
            for key in input_dict['common_data'].keys():
                if isinstance(input_dict['common_data'][key], torch.Tensor):
                    input_dict['common_data'][key] = input_dict['common_data'][key][0]

        loss_values = {}
        total_loss = 0
        iter_num = input_dict['iter_num']
        for loss_name in self.losses.keys():
            loss_obj = self.losses[loss_name]
            loss_weight = self.get_loss_weight(loss_obj, iter_num)
            loss_dict = loss_obj.compute_loss(input_dict, output_dict, return_loss_maps=return_loss_maps)
            # loss_dict may be None for visibility consistency loss for validation images during validation.
            if loss_dict is not None:
                loss_values[loss_name] = loss_dict
                total_loss += loss_weight * loss_dict['loss_value']
        loss_values['TotalLoss'] = total_loss
        return loss_values

    @staticmethod
    def get_loss_weight(loss_obj, iter_num):
        loss_weight = None
        if 'weight' in loss_obj.loss_configs:
            loss_weight = loss_obj.loss_configs['weight']
        elif 'iter_weights' in loss_obj.loss_configs:
            loss_iter_weights = loss_obj.loss_configs['iter_weights']
            loss_iter_keys = [int(key) for key in loss_iter_weights.keys()]
            loss_iter_keys = sorted(loss_iter_keys)[::-1]
            for loss_iter_key in loss_iter_keys:
                if iter_num >= int(loss_iter_key):
                    loss_weight = loss_iter_weights[str(loss_iter_key)]
                    break
        if loss_weight is None:
            raise RuntimeError(f'loss_weight is None for {loss_obj.__module__} at iter {iter_num}')
        return loss_weight
