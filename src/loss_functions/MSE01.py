# Shree KRISHNAya Namaha
# MSE loss function
# Extended from MSE07.py. Excludes sparse depth pixels when computing MSE loss - in line with DS-NeRF.
# Author: Nagabhushan S N
# Last Modified: 04/12/2022

from pathlib import Path

from matplotlib import pyplot

import torch
import torch.nn.functional as F

from loss_functions import LossUtils01
from loss_functions.LossFunctionParent01 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class MSE(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_mlp_needed = self.configs['model']['use_coarse_mlp']
        self.fine_mlp_needed = self.configs['model']['use_fine_mlp']
        self.num_rays = self.configs['data_loader']['num_rays']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, training: bool = True):
        target_rgb = input_dict['target_rgb'][:self.num_rays]
        total_loss = 0
        loss_maps = {}

        if self.coarse_mlp_needed:
            pred_rgb_coarse = output_dict['rgb_coarse'][:self.num_rays]
            mse_coarse = self.compute_mse(pred_rgb_coarse, target_rgb, training)
            total_loss += mse_coarse['loss_value']
            if not training:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, mse_coarse['loss_maps'], suffix='coarse')

        if self.fine_mlp_needed:
            pred_rgb_fine = output_dict['rgb_fine'][:self.num_rays]
            mse_fine = self.compute_mse(pred_rgb_fine, target_rgb, training)
            total_loss += mse_fine['loss_value']
            if not training:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, mse_fine['loss_maps'], suffix='fine')

        loss_dict = {
            'loss_value': total_loss,
        }
        if not training:
            loss_dict['loss_maps'] = loss_maps
        return loss_dict

    @staticmethod
    def compute_mse(pred_value, true_value, training: bool):
        error = pred_value - true_value
        mse = torch.mean(torch.square(error), dim=1)
        mean_mse = torch.mean(mse)
        loss_dict = {
            'loss_value': mean_mse,
        }
        if not training:
            loss_dict['loss_maps'] = {
                this_filename: mse
            }
        return loss_dict
