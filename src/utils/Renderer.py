

import torch
from src.common import get_rays, sample_pdf, normalize_3d_coordinate
from src.common import (get_samples, random_select, matrix_to_cam_pose, cam_pose_to_matrix, get_edge_samples)

class Renderer(object):
    """
    Renderer class for rendering depth and color.
    Args:
        cfg (dict): configuration.
        ray_batch_size (int): batch size for sampling rays.
    """
    def __init__(self, cfg, plgslam, ray_batch_size=10000):
        self.cfg = plgslam.cfg
        self.ray_batch_size = ray_batch_size
        self.device = plgslam.device
        self.perturb = cfg['rendering']['perturb']
        self.n_stratified = cfg['rendering']['n_stratified']
        self.n_importance = cfg['rendering']['n_importance']

        self.scale = cfg['scale']
        self.bound = plgslam.bound.to(plgslam.device, non_blocking=True)
        self.cur_rf_id = plgslam.shared_cur_rf_id

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = plgslam.H, plgslam.W, plgslam.fx, plgslam.fy, plgslam.cx, plgslam.cy
        self.embedpos_fn = plgslam.embedpos_fn

    def perturbation(self, z_vals):
        """
        Add perturbation to sampled depth values on the rays.
        Args:
            z_vals (tensor): sampled depth values on the rays.
        Returns:
            z_vals (tensor): perturbed depth values on the rays.
        """
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)

        return lower + (upper - lower) * t_rand

    def render_batch_ray(self, all_planes, decoders, rays_d, rays_o, device, truncation, gt_depth=None):

        """
        Render depth and color for a batch of rays.
        Args:
            all_planes (Tuple): all feature planes.
            all_planes_global(Tuple): all global feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
            rays_d (tensor): ray directions.
            rays_o (tensor): ray origins.
            device (torch.device): device to run on.
            truncation (float): truncation threshold.
            gt_depth (tensor): ground truth depth.
        Returns:
            depth_map (tensor): depth map.
            color_map (tensor): color map.
            volume_densities (tensor): volume densities for sampled points.
            z_vals (tensor): sampled depth values on the rays.

        """
        n_stratified = self.n_stratified
        n_importance = self.n_importance
        n_rays = rays_o.shape[0]

        z_vals = torch.empty([n_rays, n_stratified + n_importance], device=device)
        near = 0.0
        t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=device)
        t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=device)

        ### pixels with gt depth:
        # print("gt_depth: ", gt_depth.shape)
        gt_depth = gt_depth.reshape(-1, 1)
        # print("gt_depth_reshape: ", gt_depth.shape)
        gt_mask = (gt_depth > 0).squeeze()
        # print("gt_mask: ", gt_mask.shape)
        gt_nonezero = gt_depth[gt_mask]

        ## Sampling points around the gt depth (surface)
        gt_depth_surface = gt_nonezero.expand(-1, n_importance)
        z_vals_surface = gt_depth_surface - (1.5 * truncation)  + (3 * truncation * t_vals_surface)

        gt_depth_free = gt_nonezero.expand(-1, n_stratified)
        z_vals_free = near + 1.2 * gt_depth_free * t_vals_uni

        z_vals_nonzero, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)
        if self.perturb:
            z_vals_nonzero = self.perturbation(z_vals_nonzero)
        z_vals[gt_mask] = z_vals_nonzero.float()
        #z_vals = z_vals.float()

        ### pixels without gt depth (importance sampling):
        if not gt_mask.all():
            with torch.no_grad():
                rays_o_uni = rays_o[~gt_mask].detach()
                rays_d_uni = rays_d[~gt_mask].detach()
                det_rays_o = rays_o_uni.unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d_uni.unsqueeze(-1)  # (N, 3, 1)
                #t = (self.bound[self.cur_rf_id[0]].unsqueeze(0) - det_rays_o) / det_rays_d  # (N, 3, 2)
                t = (self.bound.unsqueeze(0) - det_rays_o)/det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01

                z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
                if self.perturb:
                    z_vals_uni = self.perturbation(z_vals_uni)
                pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(-1)  # [n_rays, n_stratified, 3]
                inputs_flat = torch.reshape(pts_uni, [-1, pts_uni.shape[-1]])
                embed_pos = self.embedpos_fn(inputs_flat)
            ##############################
                raw_uni = decoders(pts_uni, embed_pos, all_planes)
                sdf_uni = raw_uni[..., -1]
                #sdf_uni = decoders.get_raw_sdf(pts_uni_nor, embed_pos, all_planes)
                sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
                alpha_uni = self.sdf2alpha(sdf_uni, decoders.beta)
                weights_uni = alpha_uni * torch.cumprod(torch.cat([torch.ones((alpha_uni.shape[0], 1), device=device)
                                                        , (1. - alpha_uni + 1e-10)], -1), -1)[:, :-1]
                '''
                weights_uni = torch.sigmoid(sdf_uni / self.cfg['training']['trunc']) * torch.sigmoid(
                    -sdf_uni / self.cfg['training']['trunc'])

                signs = sdf_uni[:, 1:] * sdf_uni[:, :-1]
                mask = torch.where(signs < 0.0, torch.ones_like(signs),
                                   torch.zeros_like(signs))
                inds = torch.argmax(mask, axis=1)
                inds = inds[..., None]
                z_min = torch.gather(z_vals_uni, 1, inds)
                mask = torch.where(z_vals_uni < z_min + self.cfg['data']['sc_factor'] * self.cfg['training']['trunc'],
                                   torch.ones_like(z_vals_uni), torch.zeros_like(z_vals_uni))

                weights_uni = weights_uni * mask
                weights_uni = weights_uni / (torch.sum(weights_uni, axis=-1, keepdims=True) + 1e-8)
'''
                z_vals_uni_mid = .5 * (z_vals_uni[..., 1:] + z_vals_uni[..., :-1])
                z_samples_uni = sample_pdf(z_vals_uni_mid, weights_uni[..., 1:-1], n_importance, det=False, device=device)
                z_vals_uni, ind = torch.sort(torch.cat([z_vals_uni, z_samples_uni], -1), -1)
                z_vals[~gt_mask] = z_vals_uni

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [n_rays, n_stratified+n_importance, 3]
        # mask_outbbox = ~(torch.max((torch.abs(world2rf - pts_uni))) > max_drift).any(
        #         dim=-1
        #         )
        # pts = pts[mask_outbbox]
        inputs_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        embed_pos = self.embedpos_fn(inputs_flat)
        #raw = decoders(pts, embed_pos, all_planes)  #(4000,40,4) rgb+sdf
        raw = decoders(pts.to(torch.float32), embed_pos, all_planes)
        alpha = self.sdf2alpha(raw[..., -1], decoders.beta) # Need to modify
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device)
                                                , (1. - alpha + 1e-10)], -1), -1)[:, :-1]
        '''
        sdf = raw[..., -1]
        weights = torch.sigmoid(sdf / self.cfg['training']['trunc']) * torch.sigmoid(-sdf / self.cfg['training']['trunc'])

        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs),
                           torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds)
        mask = torch.where(z_vals < z_min + self.cfg['data']['sc_factor'] * self.cfg['training']['trunc'],
                           torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        weights = weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)
'''
        rendered_rgb = torch.sum(weights[..., None] * raw[..., :3], -2)
        rendered_depth = torch.sum(weights * z_vals, -1)

        return rendered_depth, rendered_rgb, raw[..., -1], z_vals

    def sdf2alpha(self, sdf, beta=10):
        """

        """
        return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))

    def render_img(self, all_planes, decoders, c2w, truncation, device, gt_depth=None):
        """
        Renders out depth and color images.
        Args:
            all_planes (Tuple): feature planes
            all_planes_global(Tuple): all global feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
            c2w (tensor, 4*4): camera pose.
            truncation (float): truncation distance.
            device (torch.device): device to run on.
            gt_depth (tensor, H*W): ground truth depth image.
        Returns:
            rendered_depth (tensor, H*W): rendered depth image.
            rendered_rgb (tensor, H*W*3): rendered color image.

        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                if gt_depth is None:
                    ret = self.render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    ret = self.render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=gt_depth_batch)

                depth, color, _, _ = ret
                # print("blended_depth_render_img: ", depth.shape)
                # print("blended_color_render_img: ", color.shape)
                depth_list.append(depth.double())
                color_list.append(color)

            depth = torch.cat(depth_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(H, W)
            color = color.reshape(H, W, 3)

            return depth, color
        

    def render_img_multi_rf(self, all_planes_list, decoders_list, world2rf, c2w, truncation, gt_depth, device):
        """
        Renders an image using all reference frames, blending their contributions.

        Args:
            all_planes_list (list): List of all feature planes for each RF.
            decoders_list (list): List of decoders for each RF.
            world2rf (torch.Tensor): Tensor of world-to-RF transformations.
            c2w (torch.Tensor): Camera-to-world matrix for the current frame.
            gt_depth (torch.Tensor): Ground truth depth for rendering.

        Returns:
            tuple: Rendered depth and color images.
        """
        with torch.no_grad():
            # Generate rays
            H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

            rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)

            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            # Split rays into batches to manage memory
            depth_renders = []
            color_renders = []
            gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_o.shape[0], self.ray_batch_size):
                rays_o_batch = rays_o[i:i + self.ray_batch_size]
                rays_d_batch = rays_d[i:i + self.ray_batch_size]
                gt_depth_batch = gt_depth[i:i + self.ray_batch_size]

                # Render for each RF and blend
                batch_depths = []
                batch_colors = []
                for rf_idx, (all_planes, decoders) in enumerate(zip(all_planes_list, decoders_list)):
                    depth, color, _, _ = self.render_batch_ray(
                        all_planes, decoders, rays_d_batch, rays_o_batch, self.device,
                        truncation, gt_depth=gt_depth_batch
                    )
                    batch_depths.append(depth)
                    batch_colors.append(color)

                # Blend results
                points = rays_o_batch + rays_d_batch * gt_depth_batch.unsqueeze(-1)
                weights = self.compute_rf_weights(points, world2rf)
                # print("weight: ", weights.shape)
                depth_stack = torch.stack(batch_depths).transpose(0, 1)  # (batch_size, num_rf)
                color_stack = torch.stack(batch_colors).transpose(0, 1)  # (batch_size, num_rf, 3)
                weights_exp = weights.unsqueeze(-1)  # (batch_size, num_rf, 1)
                blended_depth = torch.sum(depth_stack * weights, dim=1)      # (batch_size,)
                blended_color = torch.sum(color_stack * weights_exp, dim=1)  # (batch_size, 3)
                # print("blended_depth_render_img_multi_rf: ", blended_depth.shape)
                # print("blended_color_render_img_multi_rf: ", blended_color.shape)
                depth_renders.append(blended_depth.double())
                color_renders.append(blended_color)

            # Concatenate and reshape
            depth_render = torch.cat(depth_renders, dim=0).reshape(H, W)
            color_render = torch.cat(color_renders, dim=0).reshape(H, W, 3)

            return depth_render, color_render

    def compute_rf_weights(self, points, world2rf, power=2):
        """
        Computes blending weights for each RF based on distance to points.

        Args:
            points (torch.Tensor): Points to compute weights for, shape (N, 3).
            world2rf (torch.Tensor): World-to-RF transformations, shape (num_rf, 3).
            power (float): Power for inverse distance weighting.

        Returns:
            torch.Tensor: Weights for each RF, shape (N, num_rf).
        """
        points_expanded = points.unsqueeze(1)  # [N, 1, 3]
        world2rf_expanded = world2rf.unsqueeze(0)  # [1, num_rf, 3]
        distances = torch.sum((points_expanded - world2rf_expanded) ** 2, dim=2)  # [N, num_rf]
        weights = 1 / (distances ** power + 1e-8)  # Avoid division by zero
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)  # Normalize
        return weights

