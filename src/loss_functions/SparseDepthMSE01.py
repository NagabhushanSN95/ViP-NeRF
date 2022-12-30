# Shree KRISHNAya Namaha
# MSE loss function on sparse depth
# Author: Nagabhushan S N
# Last Modified: 08/12/2022

from pathlib import Path

from matplotlib import pyplot

import torch
import torch.nn.functional as F

from loss_functions import LossUtils01
from loss_functions.LossFunctionParent02 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class SparseDepthMSE(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_mlp_needed = self.configs['model']['use_coarse_mlp']
        self.fine_mlp_needed = self.configs['model']['use_fine_mlp']
        self.ndc = self.configs['data_loader']['ndc']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, training: bool = True):
        total_loss = 0
        loss_maps = {}

        if not training:
            return {'loss_value': torch.Tensor([0]).to(input_dict['rays_o']).mean()}

        gt_depth = input_dict['sparse_depth_values'][:, 0]
        indices_mask = input_dict['indices_mask_sparse_depth']

        if not self.fine_mlp_needed:
            pred_depth_coarse = output_dict['depth_coarse']
            loss_coarse = self.compute_depth_loss(pred_depth_coarse, gt_depth, indices_mask, training)
            total_loss += loss_coarse['loss_value']
            if not training:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_coarse['loss_maps'], suffix='coarse')
        else:
            pred_depth_fine = output_dict['depth_fine']
            loss_fine = self.compute_depth_loss(pred_depth_fine, gt_depth, indices_mask, training)
            total_loss += loss_fine['loss_value']
            if not training:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_fine['loss_maps'], suffix='fine')

        loss_dict = {
            'loss_value': total_loss,
        }
        if not training:
            loss_dict['loss_maps'] = loss_maps
        return loss_dict

    @staticmethod
    def compute_depth_loss(pred_depth, true_depth, indices_mask, training: bool):
        pred_depth = pred_depth[indices_mask]
        true_depth = true_depth[indices_mask]
        error = pred_depth - true_depth
        squared_error = torch.square(error)
        depth_loss = torch.mean(squared_error)
        loss_dict = {
            'loss_value': depth_loss,
        }
        if not training:
            loss_dict['loss_maps'] = {
                # No loss maps
            }
        return loss_dict
