import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common import cam_pose_to_matrix
from utils.loss_utils import l1_loss, ssim, psnr
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_msssim import ssim

# 计算两幅图像之间的多尺度结构相似性（MS-SSIM）指标
def eval_ssim(image_es, image_gt):
    return ssim(
        image_es.unsqueeze(0),
        image_gt.unsqueeze(0),
        data_range=1.0,
        size_average=True,
    )

# 计算LPIPS感知相似性损失
loss_fn_alex = LearnedPerceptualImagePatchSimilarity(
    net_type="alex", normalize=True
).to(device='cpu')

depth_error_max = 0.08  #深度误差最大值
transmission_max = 0.2  # 透射率最大值
# 颜色权重和深度权重的最大值：通常用于生成伪色彩图像
color_hit_weight_max = 1    
depth_hit_weight_max = 1

class Frame_Visualizer(object):
    """
    Visualizes itermediate results, render out depth and color images.
    It can be called per iteration, which is good for debuging (to see how each tracking/mapping iteration performs).
    Args:
        freq (int): frequency of visualization.
        inside_freq (int): frequency of visualization inside each iteration.
        vis_dir (str): directory to save the visualization results.
        renderer (Renderer): renderer.
        truncation (float): truncation distance.
        verbose (bool): whether to print out the visualization results.
        device (str): device.
    """

    def __init__(self, freq, n_imgs, inside_freq, vis_dir, renderer, truncation, verbose, device='cuda:0'):
        self.freq = freq
        self.n_imgs = n_imgs
        self.device = device
        self.vis_dir = vis_dir
        self.verbose = verbose
        self.renderer = renderer
        self.inside_freq = inside_freq
        self.truncation = truncation
        self.min_depth=0.5
        self.max_depth=3.0

        os.makedirs(f'{vis_dir}', exist_ok=True)

    def save_imgs(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, all_planes, decoders):
        """
        Visualization of depth and color images and save to file.
        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in 
                camera to world matrix or quaternion and translation tensor.
            all_planes (Tuple): feature planes.
            all_planes_global (Tuple): global feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
        """
        with torch.no_grad():
            if (idx % self.freq == 0) and ((iter + 1) % self.inside_freq == 0):
                gt_depth_np = gt_depth.squeeze(0).cpu().numpy()
                gt_color_np = gt_color.squeeze(0).cpu().numpy()

                if c2w_or_camera_tensor.shape[-1] > 4: ## 6od
                    c2w = cam_pose_to_matrix(c2w_or_camera_tensor.clone().detach()).squeeze()
                else:
                    c2w = c2w_or_camera_tensor.squeeze().detach()

                depth, color = self.renderer.render_img(all_planes, decoders, c2w, self.truncation,
                                                        self.device, gt_depth=gt_depth)
                depth_np = depth.detach().cpu().numpy()
                color_np = color.detach().cpu().numpy()
                depth_residual = np.abs(gt_depth_np - depth_np)
                depth_residual[gt_depth_np == 0.0] = 0.0
                color_residual = np.abs(gt_color_np - color_np)
                color_residual[gt_depth_np == 0.0] = 0.0

                fig, axs = plt.subplots(2, 3, figsize=(12, 8))
                fig.tight_layout()
                max_depth = np.max(gt_depth_np)

                axs[0, 0].imshow(gt_depth_np, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 0].set_title('Input Depth')
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                axs[0, 1].imshow(depth_np, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 1].set_title('Generated Depth')
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                axs[0, 2].imshow(depth_residual, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 2].set_title('Depth Residual')
                axs[0, 2].set_xticks([])
                axs[0, 2].set_yticks([])
                gt_color_np = np.clip(gt_color_np, 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)
                axs[1, 0].imshow(gt_color_np, cmap="plasma")
                axs[1, 0].set_title('Input RGB')
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                axs[1, 1].imshow(color_np, cmap="plasma")
                axs[1, 1].set_title('Generated RGB')
                axs[1, 1].set_xticks([])
                axs[1, 1].set_yticks([])
                axs[1, 2].imshow(color_residual, cmap="plasma")
                axs[1, 2].set_title('RGB Residual')
                axs[1, 2].set_xticks([])
                axs[1, 2].set_yticks([])
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(f'{self.vis_dir}/{idx:05d}_{(iter + 1):04d}.jpg', bbox_inches='tight', pad_inches=0.2, dpi=600)
                plt.cla()
                plt.clf()

                # if self.verbose:
                print(f'Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{(iter + 1):04d}.jpg')
                # psnr_value = psnr(gt_color.squeeze(0).cpu(), color.detach().cpu()).mean()
                # ssim_value = eval_ssim(color.detach().cpu().double(), gt_color.squeeze(0).cpu().double()).mean()
                # lpips_value = loss_fn_alex(
                #     torch.clamp(gt_color.squeeze(0).cpu().unsqueeze(0).permute(0, 3, 1, 2).to(dtype=torch.float32) , 0.0, 1.0),
                #     torch.clamp(color.detach().cpu().unsqueeze(0).permute(0, 3, 1, 2).to(dtype=torch.float32) , 0.0, 1.0),
                # ).item()

                # color_loss = l1_loss(gt_color.squeeze(0).cpu(), color.detach().cpu())

                # # 过滤掉无效的深度值
                # valid_range_mask = (gt_depth > self.min_depth) & (gt_depth < self.max_depth)
                # gt_depth[~valid_range_mask] = 0
                # invalid_depth_mask = (gt_depth == 0)
                # valid_depth_mask = ~invalid_depth_mask
                # depth_loss = l1_loss(depth[valid_depth_mask].detach().cpu(), gt_depth[valid_depth_mask].squeeze(0).cpu())
                # log_info = "color loss={:.3f}, depth loss={:.3f}cm, psnr={:.3f}, ssim={:.3f}, lpips={:.3f}".format(
                #     color_loss, depth_loss * 100, psnr_value, ssim_value, lpips_value
                # )
                # print(log_info)


            if (idx % (self.n_imgs - 1) == 0) and ((iter + 1) % self.inside_freq == 0):
                gt_depth_np = gt_depth.squeeze(0).cpu().numpy()
                gt_color_np = gt_color.squeeze(0).cpu().numpy()

                if c2w_or_camera_tensor.shape[-1] > 4: ## 6od
                    c2w = cam_pose_to_matrix(c2w_or_camera_tensor.clone().detach()).squeeze()
                else:
                    c2w = c2w_or_camera_tensor.squeeze().detach()

                depth, color = self.renderer.render_img(all_planes, decoders, c2w, self.truncation,
                                                        self.device, gt_depth=gt_depth)
                depth_np = depth.detach().cpu().numpy()
                color_np = color.detach().cpu().numpy()
                depth_residual = np.abs(gt_depth_np - depth_np)
                depth_residual[gt_depth_np == 0.0] = 0.0
                color_residual = np.abs(gt_color_np - color_np)
                color_residual[gt_depth_np == 0.0] = 0.0

                fig, axs = plt.subplots(2, 3, figsize=(12, 8))
                fig.tight_layout()
                max_depth = np.max(gt_depth_np)

                axs[0, 0].imshow(gt_depth_np, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 0].set_title('Input Depth')
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                axs[0, 1].imshow(depth_np, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 1].set_title('Generated Depth')
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                axs[0, 2].imshow(depth_residual, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 2].set_title('Depth Residual')
                axs[0, 2].set_xticks([])
                axs[0, 2].set_yticks([])
                gt_color_np = np.clip(gt_color_np, 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)
                axs[1, 0].imshow(gt_color_np, cmap="plasma")
                axs[1, 0].set_title('Input RGB')
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                axs[1, 1].imshow(color_np, cmap="plasma")
                axs[1, 1].set_title('Generated RGB')
                axs[1, 1].set_xticks([])
                axs[1, 1].set_yticks([])
                axs[1, 2].imshow(color_residual, cmap="plasma")
                axs[1, 2].set_title('RGB Residual')
                axs[1, 2].set_xticks([])
                axs[1, 2].set_yticks([])
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(f'{self.vis_dir}/{idx:05d}_{(iter + 1):04d}.jpg', bbox_inches='tight', pad_inches=0.2, dpi=600)
                plt.cla()
                plt.clf()

                # if self.verbose:
                print(f'Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{(iter + 1):04d}.jpg')
                # psnr_value = psnr(gt_color.squeeze(0).cpu(), color.detach().cpu()).mean()
                # ssim_value = eval_ssim(color.detach().cpu().double(), gt_color.squeeze(0).cpu().double()).mean()
                # lpips_value = loss_fn_alex(
                #     torch.clamp(gt_color.squeeze(0).cpu().unsqueeze(0).permute(0, 3, 1, 2).to(dtype=torch.float32) , 0.0, 1.0),
                #     torch.clamp(color.detach().cpu().unsqueeze(0).permute(0, 3, 1, 2).to(dtype=torch.float32) , 0.0, 1.0),
                # ).item()

                # color_loss = l1_loss(gt_color.squeeze(0).cpu(), color.detach().cpu())

                # # 过滤掉无效的深度值
                # valid_range_mask = (gt_depth > self.min_depth) & (gt_depth < self.max_depth)
                # gt_depth[~valid_range_mask] = 0
                # invalid_depth_mask = (gt_depth == 0)
                # valid_depth_mask = ~invalid_depth_mask
                # depth_loss = l1_loss(depth[valid_depth_mask].detach().cpu(), gt_depth[valid_depth_mask].squeeze(0).cpu())
                # log_info = "color loss={:.3f}, depth loss={:.3f}cm, psnr={:.3f}, ssim={:.3f}, lpips={:.3f}".format(
                #     color_loss, depth_loss * 100, psnr_value, ssim_value, lpips_value
                # )
                # print(log_info)

        # return psnr_value, ssim_value, lpips_value, color_loss, depth_loss
