# Shree KRISHNAya Namaha
# MSE loss function on dense depth
# Extended from DenseDepthMSE03.py. Loss imposed in non-ndc space, similar to SparseDepthMSE08.py.
# Author: Nagabhushan S N
# Last Modified: 14/12/2022

from pathlib import Path

from matplotlib import pyplot

import torch
import torch.nn.functional as F

from loss_functions import LossUtils01
from loss_functions.LossFunctionParent01 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class DenseDepthMSE(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_mlp_needed = self.configs['model']['use_coarse_mlp']
        self.fine_mlp_needed = self.configs['model']['use_fine_mlp']
        self.num_rays = self.configs['data_loader']['num_rays']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, training: bool = True):
        gt_depth = input_dict['dense_depth_values'][:self.num_rays, 0]
        total_loss = 0
        loss_maps = {}

        if self.coarse_mlp_needed:
            pred_depth_coarse = output_dict['depth_coarse'][:self.num_rays]
            depth_mse_coarse = self.compute_mse(pred_depth_coarse, gt_depth, training)
            total_loss += depth_mse_coarse['loss_value']
            if not training:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, depth_mse_coarse['loss_maps'], suffix='coarse')

        if self.fine_mlp_needed:
            pred_depth_fine = output_dict['depth_fine'][:self.num_rays]
            depth_mse_fine = self.compute_mse(pred_depth_fine, gt_depth, training)
            total_loss += depth_mse_fine['loss_value']
            if not training:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, depth_mse_fine['loss_maps'], suffix='fine')

        loss_dict = {
            'loss_value': total_loss,
        }
        if not training:
            loss_dict['loss_maps'] = loss_maps
        return loss_dict

    @staticmethod
    def compute_mse(pred_depth, gt_depth, training: bool):
        error = pred_depth - gt_depth
        ray_se = torch.square(error)
        depth_mse = torch.mean(ray_se)
        loss_dict = {
            'loss_value': depth_mse,
        }
        if not training:
            loss_dict['loss_maps'] = {
                this_filename: ray_se
            }
        return loss_dict
