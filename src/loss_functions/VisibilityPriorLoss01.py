# Shree KRISHNAya Namaha
# In a ray, points corresponding to visible pixels should have high visibility in the other view
# Author: Nagabhushan S N
# Last Modified: 30/12/2022

from pathlib import Path

from matplotlib import pyplot

import torch
import torch.nn.functional as F

from loss_functions import LossUtils01
from loss_functions.LossFunctionParent01 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class VisibilityConsistencyLoss(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_mlp_needed = self.configs['model']['use_coarse_mlp']
        self.fine_mlp_needed = self.configs['model']['use_fine_mlp']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, training: bool = True):
        total_loss = 0
        loss_maps = {}

        if (self.coarse_mlp_needed and ('raw_visibility2_coarse' not in output_dict)) or \
            (self.fine_mlp_needed and ('raw_visibility2_fine' not in output_dict)):
            return None

        if 'visibility_prior_masks' in input_dict:
            weights = input_dict['visibility_prior_masks']
        elif 'visibility_prior_weights' in input_dict:
            weights = input_dict['visibility_prior_weights']
        else:
            num_frames = input_dict['num_frames']
            num_rays = input_dict['rays_o'].shape[0]
            weights = torch.ones((num_rays, num_frames-1)).to(input_dict['rays_o'])

        if self.coarse_mlp_needed:
            weights_coarse = output_dict['weights_coarse']

            for i in range(len(output_dict['raw_visibility2_coarse'])):
                pred_vis_other_coarse = output_dict['raw_visibility2_coarse'][i][..., 0]
                loss_coarse = self.compute_consistency_loss(weights_coarse, pred_vis_other_coarse, weights[:, i], training)
                total_loss += loss_coarse['loss_value']
                if not training:
                    loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_coarse['loss_maps'], suffix='coarse')

        if self.fine_mlp_needed:
            weights_fine = output_dict['weights_fine']

            for i in range(len(output_dict['raw_visibility2_fine'])):
                pred_vis_other_fine = output_dict['raw_visibility2_fine'][i][..., 0]
                loss_fine = self.compute_consistency_loss(weights_fine, pred_vis_other_fine, weights[:, i], training)
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
    def compute_consistency_loss(weights1, vis2, mask, training: bool):
        vis_loss_point = 1 - vis2
        vis_loss_ray = torch.sum(weights1 * vis_loss_point, dim=1)  # (num_rays,)
        total_loss_ray = mask * vis_loss_ray
        mean_loss = torch.mean(total_loss_ray)

        loss_dict = {
            'loss_value': mean_loss,
        }
        if not training:
            loss_dict['loss_maps'] = {
                f'{this_filename}_visible': vis_loss_ray,
            }
        return loss_dict
