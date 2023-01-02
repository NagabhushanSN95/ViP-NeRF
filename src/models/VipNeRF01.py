# Shree KRISHNAya Namaha
# NeRF that supports predicting visibility.
# Author: Nagabhushan S N
# Last Modified: 02/01/2023

import numpy
import torch
import torch.nn.functional as F

from utils import CommonUtils01 as CommonUtils


class VipNeRF(torch.nn.Module):
    def __init__(self, configs: dict, model_configs: dict):
        super().__init__()
        self.configs = configs
        self.model_configs = model_configs
        self.device = CommonUtils.get_device(self.configs['device'])
        self.ndc = self.configs['data_loader']['ndc']
        self.coarse_mlp_needed = self.configs['model']['use_coarse_mlp']
        self.fine_mlp_needed = self.configs['model']['use_fine_mlp']
        self.predict_visibility = self.configs['model']['predict_visibility']

        self.pts_pos_enc_fn = None
        self.views_pos_enc_fn = None
        self.coarse_model = None
        self.fine_model = None
        self.build_nerf()
        return

    def build_nerf(self):
        self.pts_pos_enc_fn, pts_input_dim = self.get_positional_encoder(self.configs['model']['points_positional_encoding_degree'])

        views_input_dim = 0
        if self.configs['model']['use_view_dirs']:
            self.views_pos_enc_fn, views_input_dim = self.get_positional_encoder(self.configs['model']['views_positional_encoding_degree'])

        if self.coarse_mlp_needed:
            self.coarse_model = MLP(self.configs, pts_input_dim, views_input_dim).to(self.device)

        if self.fine_mlp_needed:
            self.fine_model = MLP(self.configs, pts_input_dim, views_input_dim).to(self.device)
        return

    def forward(self, input_batch: dict, retraw: bool = False, sec_views_vis: bool = False):
        render_output_dict = self.render(input_batch, retraw=retraw or self.training, sec_views_vis=sec_views_vis or self.training)
        return render_output_dict

    def render(self, input_dict: dict, retraw: bool, sec_views_vis: bool):
        all_ret = self.batchify_rays(input_dict, retraw, sec_views_vis)
        return all_ret

    def batchify_rays(self, input_dict: dict, retraw, sec_views_vis):
        """
        Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        num_rays = input_dict['rays_o'].shape[0]
        chunk = self.configs['model']['chunk']
        for i in range(0, num_rays, chunk):
            render_rays_dict = {}
            for key in input_dict:
                if isinstance(input_dict[key], torch.Tensor) and (input_dict[key].shape[0] == num_rays):
                    render_rays_dict[key] = input_dict[key][i:i+chunk]
                elif isinstance(input_dict[key], numpy.ndarray) and (input_dict[key].shape[0] == num_rays):  # indices
                    render_rays_dict[key] = input_dict[key][i:i+chunk]
                elif isinstance(input_dict[key], list) and (input_dict[key][0].shape[0] == num_rays):
                    render_rays_dict[key] = []
                    for i1 in range(len(input_dict[key])):
                        render_rays_dict[key].append(input_dict[key][i1][i:i+chunk])
                else:
                    render_rays_dict[key] = input_dict[key]

            # ret = self.render_rays(rays_flat[i:i+chunk], **kwargs)
            ret = self.render_rays(render_rays_dict, retraw, sec_views_vis)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = self.merge_mini_batch_data(all_ret)
        return all_ret

    def render_rays(self, input_dict: dict, retraw, sec_views_vis):
        rays_o = input_dict['rays_o']
        rays_d = input_dict['rays_d']
        if self.ndc:
            rays_o_ndc = input_dict['rays_o_ndc']
            rays_d_ndc = input_dict['rays_d_ndc']
        if self.configs['model']['use_view_dirs']:
            # provide ray directions as input
            view_dirs = input_dict['view_dirs']

        if self.predict_visibility and sec_views_vis:
            if 'rays_o2_list' in input_dict:
                rays_o2 = input_dict['rays_o2_list']
            else:
                poses = input_dict['poses']
                pixel_id = input_dict['pixel_id']
                image_id = pixel_id[:, 0].long()
                num_frames = input_dict['num_frames']
                rays_o2 = []
                for i in range(num_frames-1):
                    other_image_id = i + (i >= image_id).long()
                    poses2i = poses[other_image_id]  # (n, 3, 4)
                    rays_o2i = poses2i[:, :, 3]  # (n, 3)
                    rays_o2.append(rays_o2i)

        return_dict = {}
        if self.coarse_mlp_needed:
            z_vals_coarse = self.get_z_vals_coarse(input_dict)

            if not self.ndc:
                pts_coarse = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_coarse[...,:,None] # [num_rays, num_samples, 3]
            else:
                pts_coarse = rays_o_ndc[..., None, :] + rays_d_ndc[..., None, :] * z_vals_coarse[..., :, None]  # [num_rays, num_samples, 3]
            network_input_coarse = {
                'pts': pts_coarse,
            }
            if self.configs['model']['use_view_dirs']:
                network_input_coarse['view_dirs'] = view_dirs

            if self.predict_visibility and sec_views_vis:
                view_dirs2_list = []
                for i in range(len(rays_o2)):
                    view_dirs2 = self.compute_other_view_dirs(z_vals_coarse, rays_o, rays_d, rays_o2[i])
                    view_dirs2_list.append(view_dirs2)
                network_input_coarse['view_dirs2'] = view_dirs2_list

            network_output_coarse = self.run_network(network_input_coarse, self.coarse_model)
            if not self.ndc:
                outputs_coarse = self.volume_rendering(network_output_coarse, z_vals=z_vals_coarse, rays_d=rays_d,
                                                       sec_views_vis=sec_views_vis)
            else:
                outputs_coarse = self.volume_rendering(network_output_coarse, z_vals_ndc=z_vals_coarse,
                                                       rays_d_ndc=rays_d_ndc, rays_o=rays_o, rays_d=rays_d,
                                                       sec_views_vis=sec_views_vis)
            weights_coarse = outputs_coarse['weights']

            return_dict['z_vals_coarse'] = z_vals_coarse
            for key in outputs_coarse:
                return_dict[f'{key}_coarse'] = outputs_coarse[key]
            if retraw:
                for key in network_output_coarse.keys():
                    return_dict[f'raw_{key}_coarse'] = network_output_coarse[key]

        if self.fine_mlp_needed:
            z_vals_fine = self.get_z_vals_fine(z_vals_coarse, weights_coarse)
            if not self.ndc:
                pts_fine = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_fine[...,:,None]  # [num_rays, num_samples, 3]
            else:
                pts_fine = rays_o_ndc[..., None, :] + rays_d_ndc[..., None, :] * z_vals_fine[..., :, None]  # [num_rays, num_samples, 3]

            network_input_fine = {
                'pts': pts_fine,
            }
            if self.configs['model']['use_view_dirs']:
                network_input_fine['view_dirs'] = view_dirs

            if self.predict_visibility and sec_views_vis:
                view_dirs2_list = []
                for i in range(len(rays_o2)):
                    view_dirs2 = self.compute_other_view_dirs(z_vals_fine, rays_o, rays_d, rays_o2[i])
                    view_dirs2_list.append(view_dirs2)
                network_input_fine['view_dirs2'] = view_dirs2_list

            network_output_fine = self.run_network(network_input_fine, self.fine_model)
            if not self.ndc:
                outputs_fine = self.volume_rendering(network_output_fine, z_vals=z_vals_fine, rays_d=rays_d,
                                                     sec_views_vis=sec_views_vis)
            else:
                outputs_fine = self.volume_rendering(network_output_fine, z_vals_ndc=z_vals_fine, rays_d_ndc=rays_d_ndc,
                                                     rays_o=rays_o, rays_d=rays_d, sec_views_vis=sec_views_vis)
            weights_fine = outputs_fine['weights']

            return_dict['z_vals_fine'] = z_vals_fine
            for key in outputs_fine:
                return_dict[f'{key}_fine'] = outputs_fine[key]
            if retraw:
                for key in network_output_fine.keys():
                    return_dict[f'raw_{key}_fine'] = network_output_fine[key]

        if not retraw:
            del return_dict['z_vals_coarse'], return_dict['visibility_coarse'], return_dict['weights_coarse']
            del return_dict['z_vals_fine'], return_dict['visibility_fine'], return_dict['weights_fine']
        return return_dict

    def get_z_vals_coarse(self, input_dict: dict):
        num_rays = input_dict['rays_o'].shape[0]
        if not self.ndc:
            near, far = input_dict['near'], input_dict['far']
        else:
            near, far = input_dict['near_ndc'], input_dict['far_ndc']

        perturb = self.configs['model']['perturb']
        if not self.training:
            perturb = False
        lindisp = self.configs['model']['lindisp']

        num_samples_coarse = self.configs['model']['num_samples_coarse']
        t_vals = torch.linspace(0., 1., steps=num_samples_coarse)
        if not lindisp:
            z_vals_coarse = near * (1.-t_vals) + far * t_vals
        else:
            z_vals_coarse = 1./(1. / near * (1.-t_vals) + 1. / far * t_vals)

        z_vals_coarse = z_vals_coarse.expand([num_rays, num_samples_coarse])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
            upper = torch.cat([mids, z_vals_coarse[..., -1:]], -1)
            lower = torch.cat([z_vals_coarse[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals_coarse.shape)

            z_vals_coarse = lower + (upper - lower) * t_rand
        return z_vals_coarse

    def get_z_vals_fine(self, z_vals_coarse, weights_coarse):
        num_samples_fine = self.configs['model']['num_samples_fine']
        perturb = self.configs['model']['perturb']
        if not self.training:
            perturb = False

        z_vals_mid = .5 * (z_vals_coarse[...,1:] + z_vals_coarse[...,:-1])
        z_samples = self.sample_pdf(z_vals_mid, weights_coarse[...,1:-1], num_samples_fine, det=(not perturb))
        z_samples = z_samples.detach()

        z_vals_fine, _ = torch.sort(torch.cat([z_vals_coarse, z_samples], -1), -1)
        return z_vals_fine

    def compute_other_view_dirs(self, z_vals, rays_o, rays_d, rays_o2):
        if self.ndc:
            near = 1  # TODO: do not hard-code
            tn = -(near + rays_o[..., 2]) / rays_d[..., 2]
            z_vals = (((rays_o[..., None, 2] + tn[..., None] * rays_d[..., None, 2]) / (1 - z_vals + 1e-6)) - rays_o[..., None, 2]) / rays_d[..., None, 2]
        pts = rays_o[..., None, :] + z_vals[..., None] * rays_d[..., None, :]
        view_dirs_other = (pts - rays_o2[..., None, :])
        view_dirs_other = view_dirs_other / torch.norm(view_dirs_other, dim=-1, keepdim=True)
        return view_dirs_other

    # Hierarchical sampling (section 5.2)
    @staticmethod
    def sample_pdf(bins, weights, N_samples, det=False):
        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0])/denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

        return samples

    def run_network(self, input_dict, nerf_mlp):
        """
        Prepares inputs and applies network 'nerf_mlp'.
        """
        pts_flat = torch.reshape(input_dict['pts'], [-1, input_dict['pts'].shape[-1]])
        encoded_pts = self.pts_pos_enc_fn(pts_flat)
        network_input_dict = {
            'pts': encoded_pts,
        }

        if self.configs['model']['use_view_dirs']:
            viewdirs = input_dict['view_dirs']
            if viewdirs.ndim == 2:
                viewdirs = viewdirs[:,None].expand(input_dict['pts'].shape)
            viewdirs_flat = torch.reshape(viewdirs, [-1, viewdirs.shape[-1]])
            encoded_view_dirs = self.views_pos_enc_fn(viewdirs_flat)
            network_input_dict['view_dirs'] = encoded_view_dirs

            if self.predict_visibility and ('view_dirs2' in input_dict):
                view_dirs2 = input_dict['view_dirs2']
                encoded_view_dirs2 = [self.views_pos_enc_fn(x) for x in view_dirs2]
                view_dirs2_flat = [torch.reshape(x, [-1, x.shape[-1]]) for x in encoded_view_dirs2]
                network_input_dict['view_dirs2'] = view_dirs2_flat

        network_output_dict = self.batchify(nerf_mlp)(network_input_dict)

        for k, v in network_output_dict.items():
            if isinstance(v, torch.Tensor):
                network_output_dict[k] = torch.reshape(v, list(input_dict['pts'].shape[:-1]) + [v.shape[-1]])
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                for i in range(len(v)):
                    network_output_dict[k][i] = torch.reshape(v[i], list(input_dict['pts'].shape[:-1]) + [v[i].shape[-1]])
            else:
                raise NotImplementedError
        return network_output_dict

    def batchify(self, nerf_mlp):
        """Constructs a version of 'nerf_mlp' that applies to smaller batches.
        """
        chunk = self.configs['model']['netchunk']
        if chunk is None:
            return nerf_mlp

        def ret(input_dict: dict):
            num_pts = input_dict['pts'].shape[0]
            network_output_chunks = {}
            for i in range(0, num_pts, chunk):
                network_input_chunk = {}
                for key in input_dict:
                    if isinstance(input_dict[key], torch.Tensor):
                        network_input_chunk[key] = input_dict[key][i:i+chunk]
                    elif isinstance(input_dict[key], list) and isinstance(input_dict[key][0], torch.Tensor):
                        network_input_chunk[key] = [input_dict[key][j][i:i+chunk] for j in range(len(input_dict[key]))]
                    else:
                        raise RuntimeError(key)

                network_output_chunk = nerf_mlp(network_input_chunk)

                for k in network_output_chunk.keys():
                    if k not in network_output_chunks:
                        network_output_chunks[k] = []
                    if isinstance(network_output_chunk[k], torch.Tensor):
                        network_output_chunks[k].append(network_output_chunk[k])
                    elif isinstance(network_output_chunk[k], list) and isinstance(network_output_chunk[k][0], torch.Tensor):
                        if len(network_output_chunks[k]) == 0:
                            for j in range(len(network_output_chunk[k])):
                                network_output_chunks[k].append([])
                        for j in range(len(network_output_chunk[k])):
                            network_output_chunks[k][j].append(network_output_chunk[k][j])
                    else:
                        raise RuntimeError

            for k in network_output_chunks:
                if isinstance(network_output_chunks[k][0], torch.Tensor):
                    network_output_chunks[k] = torch.cat(network_output_chunks[k], dim=0)
                elif isinstance(network_output_chunks[k][0], list) and isinstance(network_output_chunks[k][0][0], torch.Tensor):
                    for j in range(len(network_output_chunks[k])):
                        network_output_chunks[k][j] = torch.cat(network_output_chunks[k][j], dim=0)
                else:
                    raise NotImplementedError
            return network_output_chunks
        return ret

    def volume_rendering(self, network_output_dict, z_vals=None, rays_o=None, rays_d=None, z_vals_ndc=None,
                         rays_d_ndc=None, sec_views_vis=False):
        if not self.ndc:
            inf_depth = 1e10
            z_vals1 = torch.cat([z_vals, torch.Tensor([inf_depth]).expand(z_vals[...,:1].shape)], -1)
            z_dists = z_vals1[...,1:] - z_vals1[...,:-1]  # [N_rays, N_samples]
            delta = z_dists * torch.norm(rays_d[...,None,:], dim=-1)
        else:
            inf_depth = 1
            z_vals1 = torch.cat([z_vals_ndc, torch.Tensor([inf_depth]).expand(z_vals_ndc[...,:1].shape)], -1)
            z_dists = z_vals1[...,1:] - z_vals1[...,:-1]  # [N_rays, N_samples]
            delta = z_dists * torch.norm(rays_d_ndc[...,None,:], dim=-1)

        rgb = network_output_dict['rgb']  # [N_rays, N_samples, 3]
        sigma = network_output_dict['sigma'][..., 0]  # [N_rays, N_samples]

        alpha = 1. - torch.exp(-sigma * delta)  # [N_rays, N_samples]
        visibility = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        weights = alpha * visibility
        rgb_map = torch.sum(weights[...,None] * rgb, dim=-2)  # [N_rays, 3]

        acc_map = torch.sum(weights, dim=-1)
        if not self.ndc:
            depth_map = torch.sum(weights * z_vals, dim=-1) / (acc_map + 1e-6)
            depth_var_map = torch.sum(weights * torch.square(z_vals - depth_map[..., None]), dim=-1)
        else:
            depth_map_ndc = torch.sum(weights * z_vals_ndc, dim=-1) / (acc_map + 1e-6)
            depth_var_map_ndc = torch.sum(weights * torch.square(z_vals_ndc - depth_map_ndc[..., None]), dim=-1)
            z_vals = self.convert_depth_from_ndc(z_vals_ndc, rays_o, rays_d)
            depth_map = torch.sum(weights * z_vals, dim=-1) / (acc_map + 1e-6)
            depth_var_map = torch.sum(weights * torch.square(z_vals - depth_map[..., None]), dim=-1)

        if self.configs['model']['white_bkgd']:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return_dict = {
            'rgb': rgb_map,
            'acc': acc_map,
            'alpha': alpha,
            'visibility': visibility,
            'weights': weights,
            'depth': depth_map,
            'depth_var': depth_var_map,
        }

        if self.ndc:
            return_dict['depth_ndc'] = depth_map_ndc
            return_dict['depth_var_ndc'] = depth_var_map_ndc

        if self.predict_visibility and sec_views_vis:
            vis2_point3d = network_output_dict['visibility2']
            vis2_pixel = [torch.sum(weights * vis[..., 0], dim=-1) / (acc_map + 1e-6) for vis in vis2_point3d]
            return_dict['visibility2'] = vis2_pixel
        return return_dict

    @staticmethod
    def convert_depth_from_ndc(z_vals_ndc, rays_o, rays_d):
        """
        Converts depth in ndc to actual values
        From ndc write up, t' is z_vals_ndc and t is z_vals.
        t' = 1 - oz / (oz + t * dz)
        t = (oz / dz) * (1 / (1-t') - 1)
        But due to the final trick, oz is shifted. So, the actual oz = oz + tn * dz
        Overall t_act = t + tn = ((oz + tn * dz) / dz) * (1 / (1 - t') - 1) + tn
        """
        near = 1  # TODO: do not hard-code
        oz = rays_o[..., 2:3]
        dz = rays_d[..., 2:3]
        tn = -(near + oz) / dz
        constant = torch.where(z_vals_ndc == 1., 1e-3, 0.)
        # depth = (((oz + tn * dz) / (1 - z_vals_ndc + constant)) - oz) / dz
        depth = (oz + tn * dz) / dz * (1 / (1 - z_vals_ndc + constant) - 1) + tn
        return depth

    @staticmethod
    def get_positional_encoder(degree):
        pos_enc_kwargs = {
            'include_input': True,
            'input_dims': 3,
            'max_freq_log2': degree - 1,
            'num_freqs': degree,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }

        pos_enc = PositionalEncoder(**pos_enc_kwargs)
        pos_enc_fn = pos_enc.encode
        return pos_enc_fn, pos_enc.out_dim

    @staticmethod
    def append_to_dict_element(data_dict: dict, key: str, new_element):
        """
        Appends the `new_element` to the list identified by `key` in the dictionary `data_dict`
        """
        if key not in data_dict:
            data_dict[key] = []
        data_dict[key].append(new_element)
        return

    @staticmethod
    def merge_mini_batch_data(data_chunks: dict):
        merged_data = {}
        for key in data_chunks:
            if isinstance(data_chunks[key][0], torch.Tensor):
                merged_data[key] = torch.cat(data_chunks[key], dim=0)
            elif isinstance(data_chunks[key][0], list):
                merged_data[key] = []
                for i in range(len(data_chunks[key][0])):
                    merged_data[key].append(torch.cat([data_chunks[key][j][i] for j in range(len(data_chunks[key]))], dim=0))
            else:
                raise NotImplementedError
        return merged_data


class PositionalEncoder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.out_dim = None
        self.pos_enc_fns = []
        self.create_pos_enc_fns()
        return

    def create_pos_enc_fns(self):
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            self.pos_enc_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                self.pos_enc_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.out_dim = out_dim
        return

    def encode(self, inputs):
        return torch.cat([fn(inputs) for fn in self.pos_enc_fns], -1)


class MLP(torch.nn.Module):
    def __init__(self, configs, pts_input_dim=3, views_input_dim=3, coarse=True):
        """
        """
        super(MLP, self).__init__()
        self.configs = configs
        if coarse:
            self.D = self.configs['model']['netdepth_coarse']
            self.W = self.configs['model']['netwidth_coarse']
        else:
            self.D = self.configs['model']['netdepth_fine']
            self.W = self.configs['model']['netwidth_fine']
        self.pts_input_dim = pts_input_dim
        self.views_input_dim = views_input_dim
        self.skips = [4]
        self.view_dep_rgb = self.configs['model']['view_dependent_rgb']
        self.predict_visibility = self.configs['model']['predict_visibility']
        self.view_dep_outputs = self.view_dep_rgb or self.predict_visibility
        self.raw_noise_std = self.configs['model']['raw_noise_std']

        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(self.pts_input_dim, self.W)] +
            [torch.nn.Linear(self.W, self.W) if i not in self.skips else torch.nn.Linear(self.W + self.pts_input_dim, self.W) for i in range(self.D - 1)]
        )
        if self.view_dep_outputs:
            self.views_linears = torch.nn.ModuleList([torch.nn.Linear(self.views_input_dim + self.W, self.W // 2)])

        pts_output_dim = 1  # sigma
        views_output_dim = 0
        if not self.view_dep_rgb:
            pts_output_dim += 3
        else:
            views_output_dim += 3
        if self.predict_visibility:
            views_output_dim += 1

        self.pts_output_linear = torch.nn.Linear(self.W, pts_output_dim)
        if self.view_dep_outputs:
            self.feature_linear = torch.nn.Linear(self.W, self.W)
            self.views_output_linear = torch.nn.Linear(self.W // 2, views_output_dim)
        return

    def forward(self, input_batch):
        input_pts = input_batch['pts']
        input_views = input_batch['view_dirs']
        output_batch = {}

        pts_outputs = self.get_view_independent_outputs(input_pts)
        output_batch.update(pts_outputs)
        if not self.view_dep_rgb:
            rgb = pts_outputs['rgb_view_independent']

        if self.view_dep_outputs:
            view_outputs = self.get_view_dependent_outputs(pts_outputs, input_views)
            output_batch.update(view_outputs)
            if self.view_dep_rgb:
                rgb = view_outputs['rgb_view_dependent']

            if 'view_dirs2' in input_batch.keys():
                output_batch['visibility2'] = []
                for view_dirs2 in input_batch['view_dirs2']:
                    view_outputs2 = self.get_view_dependent_outputs(pts_outputs, view_dirs2)
                    output_batch['visibility2'].append(view_outputs2['visibility'])
        output_batch['rgb'] = rgb

        if 'feature' in output_batch:
            del output_batch['feature']
        return output_batch

    def get_view_independent_outputs(self, input_pts):
        output_dict = {}
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        pts_output = self.pts_output_linear(h)
        ch_i = 0  # Denotes number of channels which have already been taken output

        sigma = pts_output[..., ch_i:ch_i + 1]
        if self.training and (self.raw_noise_std > 0.):
            noise = torch.randn(sigma.shape) * self.raw_noise_std
            sigma = sigma + noise
        sigma = F.relu(sigma)
        output_dict['sigma'] = sigma
        ch_i += 1

        if not self.view_dep_rgb:
            rgb = pts_output[..., ch_i:ch_i+3]
            rgb = torch.sigmoid(rgb)
            output_dict['rgb_view_independent'] = rgb
            ch_i += 3

        if self.view_dep_outputs:
            feature = self.feature_linear(h)
            output_dict['feature'] = feature
        return output_dict

    def get_view_dependent_outputs(self, pts_outputs, input_views):
        output_dict = {}

        feature = pts_outputs['feature']
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        view_outputs = self.views_output_linear(h)
        ch_i = 0  # Denotes number of channels which have already been taken output

        if self.view_dep_rgb:
            rgb = view_outputs[..., ch_i:ch_i+3]
            rgb = torch.sigmoid(rgb)
            output_dict['rgb_view_dependent'] = rgb
            ch_i += 3

        if self.predict_visibility:
            visibility = view_outputs[..., ch_i:ch_i+1]
            visibility = torch.sigmoid(visibility)
            output_dict['visibility'] = visibility
            ch_i += 1
        return output_dict
