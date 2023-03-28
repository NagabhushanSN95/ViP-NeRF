# Shree KRISHNAya Namaha
# MSE loss function. Excludes sparse depth pixels.
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

from pathlib import Path

import torch

from loss_functions import LossUtils01
from loss_functions.LossFunctionParent01 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class MSE(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_mlp_needed = 'coarse_mlp' in self.configs['model']
        self.fine_mlp_needed = 'fine_mlp' in self.configs['model']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, return_loss_maps: bool = False):
        total_loss = 0
        loss_maps = {}

        indices_mask = input_dict['indices_mask_nerf']
        target_rgb = input_dict['target_rgb']

        if self.coarse_mlp_needed:
            pred_rgb_coarse = output_dict['rgb_coarse']
            mse_coarse = self.compute_mse(pred_rgb_coarse, target_rgb, indices_mask, return_loss_maps)
            total_loss += mse_coarse['loss_value']
            if return_loss_maps:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, mse_coarse['loss_maps'], suffix='coarse')

        if self.fine_mlp_needed:
            pred_rgb_fine = output_dict['rgb_fine']
            mse_fine = self.compute_mse(pred_rgb_fine, target_rgb, indices_mask, return_loss_maps)
            total_loss += mse_fine['loss_value']
            if return_loss_maps:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, mse_fine['loss_maps'], suffix='fine')

        loss_dict = {
            'loss_value': total_loss,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = loss_maps
        return loss_dict

    @staticmethod
    def compute_mse(pred_value, true_value, indices_mask, return_loss_maps: bool):
        pred_value = pred_value[indices_mask]
        true_value = true_value[indices_mask]
        error = pred_value - true_value
        mse = torch.mean(torch.square(error), dim=1)
        mean_mse = torch.mean(mse) if pred_value.numel() > 0 else 0
        loss_dict = {
            'loss_value': mean_mse,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = {
                this_filename: mse
            }
        return loss_dict
