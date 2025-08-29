#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# 相机信息的管理：

# 1.定义了 Camera 类，用于存储和管理相机的内参（如焦距、主点坐标）和外参（如旋转矩阵、平移向量）。
# 支持加载图像、深度图等数据，并将其存储为相机的属性。
# 几何变换的计算：

# 2.提供了从世界坐标系到相机坐标系的变换（world_view_transform）。
# 提供了相机的投影矩阵（projection_matrix）和完整的投影变换（full_proj_transform）。
# 相机的更新和操作：

# 3.支持动态更新相机的位姿（updatePose 和 update 方法）。
# 提供了从相机坐标系到世界坐标系的变换（get_c2w 和 get_w2c 方法）。
# 相机内参的计算：

# 4.提供了计算相机内参矩阵的方法（get_intrinsic）。
# 支持根据视场角（FoV）计算焦距（get_focal_length）。
# 像素坐标的投影：

# 5.提供了将 3D 点投影到 2D 图像平面的方法（get_uv）。
# 设备管理：

# 6.支持将相机数据移动到 GPU 或 CPU（move_to_cpu_clone 方法）。

import numpy as np
import torch
from torch import nn
from SLAM.utils import downscale_img

from utils.graphics_utils import fov2focal, getProjectionMatrix, getWorld2View2
from utils.general_utils import devF


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        depth,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        pose_gt=np.eye(4),
        cx=-1,
        cy=-1,
        timestamp=0,
        depth_scale=1.0,
        preload=True,
        data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.preload = preload
        self.timestamp = timestamp
        self.depth_scale = depth_scale
        self.last_world_view_transform = torch.eye(4, device="cuda")
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        if not self.preload:
            self.data_device = torch.device("cpu")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if depth is not None:
            self.original_depth = depth.to(self.data_device)
        else:
            self.original_depth = torch.ones(1, self.image_height, self.image_width).to(
                self.data_device
            )

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            self.original_depth *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones(
                (1, self.image_height, self.image_width), device=self.data_device
            )
            self.original_depth *= torch.ones(
                (1, self.image_height, self.image_width), device=self.data_device
            )
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        # 世界到相机坐标系的变换
        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        # 相机的投影矩阵，3D->2D
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        # 完整的投影变换：包含world_view_transform和projection_matrix
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        # 相机中心在世界坐标系中的位置
        if torch.linalg.det(self.world_view_transform) == 0:
            self.world_view_transform = self.last_world_view_transform
            # raise ValueError("world_view_transform is singular at frame {}".format(self.uid))
        else:
            self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.last_world_view_transform = self.world_view_transform
        # for evaluation, unchange
        self.pose_gt = pose_gt
        self.cx = cx
        self.cy = cy

        self.world_view_transform.share_memory_()
        self.full_proj_transform.share_memory_()

    # 将相机的位姿更新为相机坐标系到世界坐标系的变换
    # pose_c2w: 相机坐标系到世界坐标系的变换矩阵
    def updatePose(self, pose_c2w):
        pose_w2c = np.linalg.inv(pose_c2w)
        self.update(pose_w2c[:3, :3].transpose(), pose_w2c[:3, 3])

    # 更新相机的位姿
    def update(self, R, T):
        self.R = R
        self.T = T
        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, self.trans, self.scale))
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    def get_w2c(self):
        return self.world_view_transform.transpose(0, 1)

    @property
    def get_c2w(self):
        return self.world_view_transform.transpose(0, 1).inverse()

    # TODO: only work for Repulica dataset, need to add load local depth intrinsic for ScanNet
    @property
    def get_intrinsic(self):
        w, h = self.image_width, self.image_height
        fx, fy = fov2focal(self.FoVx, w), fov2focal(self.FoVy, h)
        cx = self.cx if self.cx > 0 else w / 2
        cy = self.cy if self.cy > 0 else h / 2
        intrinstic = devF(torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
        return intrinstic

    def get_focal_length(self):
        w, h = self.image_width, self.image_height
        fx, fy = fov2focal(self.FoVx, w), fov2focal(self.FoVy, h)
        return (fx + fy) / 2.0

    # 将3D点投影到2D图像平面，返回像素坐标
    def get_uv(self, xyz_w):
        intrinsic = self.get_intrinsic
        w2c = self.get_w2c()
        xyz_c = xyz_w @ w2c[:3, :3].T + w2c[:3, 3]
        uv = xyz_c @ intrinsic.T
        uv = uv[:, :2] / uv[:, 2:]
        uv = uv.long()
        return uv

    def move_to_cpu_clone(self):
        new_cam = Camera(
            colmap_id=self.colmap_id,
            R=self.R,
            T=self.T,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
            image=self.original_image.detach(),
            depth=self.original_depth.detach(),
            gt_alpha_mask=None,
            image_name=self.image_name,
            uid=self.uid,
            data_device=self.data_device,
            pose_gt=self.pose_gt,
            cx=self.cx,
            cy=self.cy,
            timestamp=self.timestamp,
            preload=self.preload,
            depth_scale=self.depth_scale,
        )
        new_cam.original_depth = new_cam.original_depth.to("cpu")
        new_cam.original_image = new_cam.original_image.to("cpu")
        return new_cam


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.cx = -1
        self.cy = -1
