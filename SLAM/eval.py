import os
from typing import List

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import torchvision

from scene.cameras import Camera
from SLAM.utils import *
from utils.loss_utils import l1_loss, ssim, psnr
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import trimesh
from pytorch_msssim import ms_ssim
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm

# 计算两幅图像之间的多尺度结构相似性（MS-SSIM）指标
def eval_ssim(image_es, image_gt):
    return ms_ssim(
        image_es.unsqueeze(0),
        image_gt.unsqueeze(0),
        data_range=1.0,
        size_average=True,
    )

# 计算LPIPS感知相似性损失
loss_fn_alex = LearnedPerceptualImagePatchSimilarity(
    net_type="alex", normalize=True
).cuda()

depth_error_max = 0.08  #深度误差最大值
transmission_max = 0.2  # 透射率最大值
# 颜色权重和深度权重的最大值：通常用于生成伪色彩图像
color_hit_weight_max = 1    
depth_hit_weight_max = 1

#评估渲染输出与真实数据（Frame）之间的差异，包括多种指标（PNSR、SSIM、LPIPS、L1等）
def eval_picture(
    render_output,
    frame: Camera,
    save_path,
    min_depth,
    max_depth,
    save_picture
):
    move_to_gpu(frame)
    image, depth, normal, index = (
        render_output["render"],
        render_output["depth"],
        render_output["normal"],
        render_output["depth_index_map"],
    )
    color_hit_weight, depth_hit_weight, T_map = (
        render_output["color_hit_weight"],
        render_output["depth_hit_weight"],
        render_output["T_map"],
    )
    # check color map
    gt_image = frame.original_image
    image_error = (gt_image - image).abs()
    # check others
    psnr_value = psnr(gt_image, image).mean()
    ssim_value = eval_ssim(image, gt_image).mean()
    lpips_value = loss_fn_alex(
        torch.clamp(gt_image.unsqueeze(0), 0.0, 1.0),
        torch.clamp(image.unsqueeze(0), 0.0, 1.0),
    ).item()

    color_loss = l1_loss(gt_image, image)

    # 保存图像对比结果
    if save_picture:
        image_concat = torch.concat([image, gt_image, image_error], dim=-1)
        torchvision.utils.save_image(
            image_concat,
            os.path.join(save_path, "color_compare.jpg"),
        )
    
    # check depth map
    gt_depth = 255.0 * frame.original_depth
    # 过滤掉无效的深度值
    valid_range_mask = (gt_depth > min_depth) & (gt_depth < max_depth)
    gt_depth[~valid_range_mask] = 0
    
    depth_error = (gt_depth - depth).abs()
    invalid_depth_mask = (index == -1) | (gt_depth == 0)
    depth_error[invalid_depth_mask] = 0

    valid_depth_mask = ~invalid_depth_mask
    pixel_num = depth.shape[1] * depth.shape[2]
    valid_pixel_ratio = valid_depth_mask.sum() / pixel_num
    depth_loss = l1_loss(depth[valid_depth_mask], gt_depth[valid_depth_mask])
    
    # 保存深度图对比结果
    if save_picture:
        min_depth = gt_depth[gt_depth > 0].min()
        max_depth = gt_depth[gt_depth > 0].max()
        colored_depth_render = color_value(
            depth, depth == 0, min_depth, max_depth, cv2.COLORMAP_INFERNO
        )
        colored_depth_gt = color_value(
            gt_depth, gt_depth == 0, min_depth, max_depth, cv2.COLORMAP_INFERNO
        )
        colored_depth_error = color_value(
            depth_error, invalid_depth_mask, 0.0, depth_error_max
        )

        colored_depth_error = color_value(
            depth_error, invalid_depth_mask, 0, 0, cv2.COLORMAP_INFERNO
        )
        colored_depth_error[:, (depth == 0)[0]] = 0
        depth_concat = torch.concat(
            [colored_depth_render, colored_depth_gt, colored_depth_error], dim=-1
        )
        torchvision.utils.save_image(
            depth_concat,
            os.path.join(save_path, "depth_compare.jpg"),
        )

    # 保存权重图
    if save_picture:
        color_weight_color = color_value(
            color_hit_weight, None, 0, color_hit_weight_max, cv2.COLORMAP_JET
        )
        depth_weight_color = color_value(
            depth_hit_weight, None, 0, depth_hit_weight_max, cv2.COLORMAP_JET
        )
        T_color = color_value(T_map, None, 0, transmission_max, cv2.COLORMAP_JET)
        torchvision.utils.save_image(
            torch.concat([color_weight_color, depth_weight_color, T_color], dim=-1),
            os.path.join(save_path, "weight_compare.png"),
        )
    
    normal_loss = torch.tensor(0)
    # save log
    log_info = "valid pixel ratio={:.2%}, color loss={:.3f}, depth loss={:.3f}cm, normal loss={:.3f}, psnr={:.3f}".format(
        valid_pixel_ratio, color_loss, depth_loss * 100, normal_loss, psnr_value
    )
    print(log_info)
    losses = {
        "valid_pixel_ratio": valid_pixel_ratio.item(),
        "depth_loss": depth_loss.item(),
        "normal_loss": normal_loss.item(),
        "psnr": psnr_value.item(),
        "ssim": ssim_value.item(),
        "lpips": lpips_value,
    }
    move_to_cpu(frame)

    return losses

# 用于计算点云重建任务的完成率
def completion_ratio(gt_points, rec_points, dist_th=0.03):
    gen_points_kd_tree = KDTree(rec_points)     #KDTree用于从查找最近邻点
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio

# 用于计算点云重建任务的准确率
def accuracy_ratio(gt_points, rec_points, dist_th=0.03):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc_ratio = np.mean((distances < dist_th).astype(np.float32))
    return acc_ratio

# 评估重建点云到真实点云的平均距离（准确性）
def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc

# 评估真实点云到重建点云的平均距离（完成度）
def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp

# 评估3D点云重建任务的质量。计算多种指标，包括准确率、完成度、召回率、F1分数等
def eval_pcd(
    rec_meshfile, gt_meshfile, dist_thres=[0.03], transform=np.eye(4),
    sample_nums = 1000000
):
    """
    3D reconstruction metric.

    """
    mesh_gt = trimesh.load(gt_meshfile, process=False)
    bbox = np.zeros([2, 3])     #存放两个坐标点（包围盒的最小点和最大点）
    bbox[0] = mesh_gt.vertices.min(axis=0) - 0.05
    bbox[1] = mesh_gt.vertices.max(axis=0) + 0.05
    rec_pc = o3d.io.read_point_cloud(rec_meshfile)
    rec_pc.transform(transform)             #应用变换矩阵   
    points = np.asarray(rec_pc.points)      #NumPy数组，表示点云的坐标
    P = points.shape[0]                     #点云的总点数
    points = points[np.random.choice(P, min(P, sample_nums), replace=False), :]  #随机采样点云
    rec_pc_tri = trimesh.PointCloud(vertices=points)        #采样后的点云

    gt_pc = trimesh.sample.sample_surface(mesh_gt, sample_nums)#真实点云采样
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])          #采样后重建点云
    print("compute acc")
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    print("compute comp")
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    Ps = {}
    Rs = {}
    Fs = {}
    for thre in tqdm(dist_thres):
        P = accuracy_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, dist_th=thre) * 100
        R = (
            completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, dist_th=thre)
            * 100
        )
        F1 = 2 * P * R / (P + R)
        Ps["P (< {})".format(thre)] = P
        Rs["R (< {})".format(thre)] = R
        Fs["F1 (< {})".format(thre)] = F1
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    results = {
        "accuracy": accuracy_rec,
        "completion": completion_rec,
    }
    results.update(Ps)
    results.update(Rs)
    results.update(Fs)
    return results


# 评估单帧数据的质量。结合eval_picture和eval_pcd函数，计算渲染图像和点云的质量指标
def eval_frame(
    mapping,
    cam,
    dir_name,
    run_picture=True,
    run_pcd=False,
    min_depth=0.5,
    max_depth=3.0,
    pcd_path=None,
    gt_mesh_path=None,
    dist_threshs=[0.03],
    sample_nums=1000000,
    pcd_transform=np.eye(4),
    save_picture=False,
):
    with torch.no_grad():
        # save render
        frame_name = "frame_{:04d}".format(mapping.time)
        render_save_path = os.path.join(dir_name, frame_name)
        losses = {}
        if run_picture:
            os.makedirs(render_save_path, exist_ok=True)
            with torch.no_grad():
                render_output = mapping.renderer.render(
                    cam, mapping.global_params
                )

            pic_loss = eval_picture(
                render_output,
                cam,
                render_save_path,
                min_depth,
                max_depth,
                save_picture,
            )
            losses.update(pic_loss)


        if run_pcd and pcd_path is not None and gt_mesh_path is not None:
            os.makedirs(render_save_path, exist_ok=True)
            pcd_losses = eval_pcd(
                pcd_path,
                gt_mesh_path,
                dist_threshs,
                pcd_transform,
                sample_nums
            )
            losses.update(pcd_losses)
        return losses
