# Shree KRISHNAya Namaha
# In a ray, points corresponding to visible pixels should have high visibility in the other view
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

from pathlib import Path

import torch

from loss_functions import LossUtils01
from loss_functions.LossFunctionParent01 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class VisibilityPriorLoss(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_mlp_needed = 'coarse_mlp' in self.configs['model']
        self.fine_mlp_needed = 'fine_mlp' in self.configs['model']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, return_loss_maps: bool = False):
        total_loss = 0
        loss_maps = {}

        if (self.coarse_mlp_needed and ('raw_visibility2_coarse' not in output_dict)) or \
            (self.fine_mlp_needed and ('raw_visibility2_fine' not in output_dict)):
            return None

        indices_mask = input_dict['indices_mask_nerf']
        if 'visibility_prior_masks' in input_dict:
            prior_weights = input_dict['visibility_prior_masks']
        elif 'visibility_prior_weights' in input_dict:
            prior_weights = input_dict['visibility_prior_weights']
        else:
            num_frames = input_dict['num_frames']
            num_rays = input_dict['rays_o'].shape[0]
            prior_weights = torch.ones((num_rays, num_frames-1)).to(input_dict['rays_o'])

        if self.coarse_mlp_needed:
            pred_vis_other_coarse = output_dict['visibility2_coarse']
            loss_coarse = self.compute_consistency_loss(pred_vis_other_coarse, prior_weights, indices_mask, return_loss_maps)
            total_loss += loss_coarse['loss_value']
            if return_loss_maps:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_coarse['loss_maps'], suffix='coarse')

        if self.fine_mlp_needed:
            pred_vis_other_fine = output_dict['visibility2_fine']
            loss_fine = self.compute_consistency_loss(pred_vis_other_fine, prior_weights, indices_mask, return_loss_maps)
            total_loss += loss_fine['loss_value']
            if return_loss_maps:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_fine['loss_maps'], suffix='fine')

        loss_dict = {
            'loss_value': total_loss,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = loss_maps
        return loss_dict

    @staticmethod
    def compute_consistency_loss(vis2, mask, indices_mask, return_loss_maps: bool):
        """

        :param vis2: (num_rays, num_frames-1)
        :param mask: (num_rays, num_frames-1)
        :param indices_mask: (num_rays, )
        :param return_loss_maps:
        :return:
        """
        vis2 = vis2[indices_mask]
        mask = mask[indices_mask]

        vis_loss_ray_views = 1 - vis2  # (nr, nf - 1)
        masked_loss_ray_views = mask * vis_loss_ray_views
        vis_loss_ray = torch.sum(masked_loss_ray_views, dim=1)  # (nr, )
        mean_loss = torch.mean(vis_loss_ray) if vis2.numel() > 0 else 0

        loss_dict = {
            'loss_value': mean_loss,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = {
                this_filename: vis_loss_ray,
            }
        return loss_dict
