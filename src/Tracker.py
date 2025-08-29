# This file is a part of ESLAM.
#
# ESLAM is a NeRF-based SLAM system. It utilizes Neural Radiance Fields (NeRF)
# to perform Simultaneous Localization and Mapping (SLAM) in real-time.
# This software is the implementation of the paper "ESLAM: Efficient Dense SLAM
# System Based on Hybrid Representation of Signed Distance Fields" by
# Mohammad Mahdi Johari, Camilla Carta, and Francois Fleuret.
#
# Copyright 2023 ams-OSRAM AG
#
# Author: Mohammad Mahdi Johari <mohammad.johari@idiap.ch>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/Tracker.py
# which is covered by the following copyright and permission notice:
    #
    # Copyright 2022 Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, Marc Pollefeys
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
from PIL import Image
import torch
import copy
import os
import time
import numpy as np
import random
from colorama import Fore, Style
from torch.utils.data import DataLoader
from tqdm import tqdm
import struct
import collections
from types import SimpleNamespace
from src.common import (matrix_to_cam_pose, cam_pose_to_matrix, get_samples)
from src.utils.datasets import get_dataset
from src.utils.Frame_Visualizer import Frame_Visualizer
from src.keyframe import KeyFrameDatabase
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle
# import pycolmap
#import src.dataloading.common as common
#from src.dataloading.dataset import DataField
from utils.graphics_utils import (
    focal2fov,
    fov2focal,
    getK,
    getWorld2View,
    getWorld2View2,
)
from scene.dataset_readers import  CameraInfo
import matplotlib.pyplot as plt
from SLAM.gp import *

import torch.multiprocessing as mp
# from SLAM.render import Renderer
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from SLAM.icp import IcpTracker
from threading import Thread
from utils.camera_utils import loadCam
# from scene import Dataset

np.set_printoptions(suppress=True, precision=6)
torch.set_printoptions(4, sci_mode=False)
# RTG:
def convert_poses(trajs):
    poses = []
    stamps = []
    for traj in trajs:
        stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = traj
        pose_ = np.eye(4)
        pose_[:3, :3] = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
        pose_[:3, 3] = np.array([t0, t1, t2])
        poses.append(pose_)
        stamps.append(stamp)
    return poses, stamps


class Tracker_rtg(object):
    def __init__(self, args):

        if isinstance(args, dict):
            args = SimpleNamespace(**args)

        self.use_gt_pose = args.use_gt_pose
        self.mode = args.mode
        self.K = None
        self.type = args.type

        self.min_depth = args.min_depth
        self.max_depth = args.max_depth
        self.depth_filter = args.depth_filter
        self.verbose = args.verbose

        self.icp_tracker = IcpTracker(args)

        self.status = defaultdict(bool)     #默认值False
        self.pose_gt = []
        self.pose_es = []
        self.timestampes = []
        self.finish = mp.Condition()

        self.icp_success_count = 0
        self.orb_only = args.orb_only
        self.use_orb_backend = args.use_orb_backend
        self.orb_vocab_path = args.orb_vocab_path
        self.orb_settings_path = args.orb_settings_path
        self.orb_backend = None
        self.orb_useicp = args.orb_useicp

        self.invalid_confidence_thresh = args.invalid_confidence_thresh

        if self.mode == "single process":
            self.initialize_orb()

    # 根据frame_id获取位姿（从估计的位姿中获取指定帧的位姿）
    def get_new_poses_byid(self, frame_ids):
        if self.use_orb_backend and not self.use_gt_pose:
            new_poses = convert_poses(self.orb_backend.get_trajectory_points())
            frame_poses = [new_poses[frame_id] for frame_id in frame_ids]
        else:
            frame_poses = [self.pose_es[frame_id] for frame_id in frame_ids]
        return frame_poses

    # 获取最新的位姿
    def get_new_poses(self):
        if self.use_orb_backend and not self.use_gt_pose:
            new_poses, _ = convert_poses(self.orb_backend.get_trajectory_points())
        else:
            new_poses = None
        return new_poses

    # 保存无效的跟踪数据（如果估计值与真实值误差超过阈值（0.15），就保存相关数据）
    def save_invalid_traing(self, path):
        if np.linalg.norm(self.pose_es[-1][:3, 3] - self.pose_gt[-1][:3, 3]) > 0.15:
            if self.track_mode == "icp":
                frame_id = len(self.pose_es)
                torch.save(
                    self.icp_tracker.vertex_pyramid_t1,
                    os.path.join(path, "vertex_pyramid_t1_{}.pt".format(frame_id)),
                )
                torch.save(
                    self.icp_tracker.vertex_pyramid_t0,
                    os.path.join(path, "vertex_pyramid_t0_{}.pt".format(frame_id)),
                )
                torch.save(
                    self.icp_tracker.normal_pyramid_t1,
                    os.path.join(path, "normal_pyramid_t1_{}.pt".format(frame_id)),
                )
                torch.save(
                    self.icp_tracker.normal_pyramid_t0,
                    os.path.join(path, "normal_pyramid_t0_{}.pt".format(frame_id)),
                )

    # 对输入帧的深度图和法线图进行预处理
    def map_preprocess(self, frame, frame_id):
        depth_map, color_map = (
            frame.original_depth.permute(1, 2, 0) * 255,
            frame.original_image.permute(1, 2, 0),
        )  # [H, W, C], the image is scaled by 255 in function "PILtoTorch"
        depth_map_orb = (
            frame.original_depth.permute(1, 2, 0).cpu().numpy()
            * 255
            * frame.depth_scale
        ).astype(np.uint16)
        intrinsic = frame.get_intrinsic
        # depth filter
        if self.depth_filter:
            depth_map_filter = bilateralFilter_torch(depth_map, 5, 2, 2)
        else:
            depth_map_filter = depth_map

        valid_range_mask = (depth_map_filter > self.min_depth) & (depth_map_filter < self.max_depth)
        depth_map_filter[~valid_range_mask] = 0.0
        # update depth map
        frame.original_depth = depth_map_filter.permute(2, 0, 1) / 255.0
        # compute geometry info
        vertex_map_c = compute_vertex_map(depth_map_filter, intrinsic)
        normal_map_c = compute_normal_map(vertex_map_c)
        confidence_map = compute_confidence_map(normal_map_c, intrinsic)

        # confidence_threshold tum: 0.5, others: 0.2
        invalid_confidence_mask = ((normal_map_c == 0).all(dim=-1)) | (
            confidence_map < self.invalid_confidence_thresh
        )[..., 0]

        depth_map_filter[invalid_confidence_mask] = 0
        normal_map_c[invalid_confidence_mask] = 0
        vertex_map_c[invalid_confidence_mask] = 0
        confidence_map[invalid_confidence_mask] = 0

        color_map_orb = (
            (frame.original_image * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        )

        self.update_curr_status(
            frame,
            frame_id,
            depth_map,
            depth_map_filter,
            vertex_map_c,
            normal_map_c,
            color_map,
            color_map_orb,
            depth_map_orb,
            intrinsic,
        )

        frame_map = {}
        frame_map["depth_map"] = depth_map_filter
        frame_map["color_map"] = color_map
        frame_map["normal_map_c"] = normal_map_c
        frame_map["vertex_map_c"] = vertex_map_c
        frame_map["confidence_map"] = confidence_map
        frame_map["invalid_confidence_mask"] = invalid_confidence_mask
        frame_map["time"] = frame_id

        return frame_map

    # 将当前帧的状态（深度图、法线图、顶点图）存储到curr_frame中
    # 该函数在每次跟踪时调用
    def update_curr_status(
        self,
        frame,
        frame_id,
        depth_t1,
        depth_t1_filter,
        vertex_t1,
        normal_t1,
        color_t1,
        color_orb,
        depth_orb,
        K,
    ):
        if self.K is None:
            self.K = K
        self.curr_frame = {
            "K": frame.get_intrinsic,
            "normal_map": normal_t1,
            "depth_map": depth_t1,
            "depth_map_filter": depth_t1_filter,
            "vertex_map": vertex_t1,
            "frame_id": frame_id,
            "pose_gt": frame.get_c2w.cpu().numpy(), # 1
            "color_map": color_t1,
            "timestamp": frame.timestamp, # 1
            "color_map_orb": color_orb, # 1
            "depth_map_orb": depth_orb, # 1
        }
        self.icp_tracker.update_curr_status(depth_t1_filter, self.K)
        
    # 更新上一帧的状态
    def update_last_status_v2(
        self, frame, render_depth, frame_depth, render_normal, frame_normal
    ):
        intrinsic = frame.get_intrinsic
        normal_mask = (
            1 - F.cosine_similarity(render_normal, frame_normal, dim=-1)
        ) < self.icp_sample_normal_threshold
        depth_filling_mask = (
            (
                torch.abs(render_depth - frame_depth)
                > self.icp_sample_distance_threshold
            )[..., 0]
            | (render_depth == 0)[..., 0]
            | (normal_mask)
        ) & (frame_depth > 0)[..., 0]

        render_depth[depth_filling_mask] = frame_depth[depth_filling_mask]
        render_depth[(frame_depth == 0)[..., 0]] = 0
        
        self.last_model_vertex = compute_vertex_map(render_depth, intrinsic)
        self.last_model_normal = compute_normal_map(self.last_model_vertex)

    def update_last_status(
        self,
        frame,
        render_depth,
        frame_depth,
        frame_normal
    ):
        self.icp_tracker.update_last_status_no_normal(
            frame, render_depth, frame_depth, frame_normal
        )

    # 通过ICP或ORB后端优化位姿
    # 如果 ICP 失败，使用 ORB 特征匹配优化位姿
    def refine_icp_pose(self, pose_t1_t0, tracking_success):
        if tracking_success and self.orb_useicp:
            print("success")
            self.orb_backend.track_with_icp_pose(
                self.curr_frame["color_map_orb"],
                self.curr_frame["depth_map_orb"],
                pose_t1_t0.astype(np.float32),
                self.curr_frame["timestamp"],
            )
            time.sleep(0.005)
        else:
            print("use orb feature")
            
            color_map_orb = np.ascontiguousarray(self.curr_frame["color_map_orb"], dtype=np.uint8)
            depth_map_orb = np.ascontiguousarray(self.curr_frame["depth_map_orb"], dtype=np.float32)
            self.orb_backend.track_with_orb_feature(
                # self.curr_frame["color_map_orb"],
                # self.curr_frame["depth_map_orb"],
                color_map_orb,
                depth_map_orb,
                self.curr_frame["timestamp"],
            )
            time.sleep(0.005)
        traj_history = self.orb_backend.get_trajectory_points()
        pose_es_t1, _ = convert_poses(traj_history[-2:])
        return pose_es_t1[-1]

    # 初始化ORB后端
    def initialize_orb(self):
        if not self.use_gt_pose and self.use_orb_backend and self.orb_backend is None:
            import orbslam2
            print("init orb backend")
            self.orb_backend = orbslam2.System(
                self.orb_vocab_path, self.orb_settings_path, orbslam2.Sensor.RGBD
            )
            self.orb_backend.set_use_viewer(False)
            self.orb_backend.initialize(self.orb_useicp)

    # 初始化跟踪器
    def initialize_tracker(self):
        if self.use_orb_backend:
            self.orb_backend.process_image_rgbd(
                self.curr_frame["color_map_orb"],
                self.curr_frame["depth_map_orb"],
                self.curr_frame["timestamp"],
            )
        self.status["initialized"] = True

    # 执行跟踪任务
    def tracking(self, frame, frame_map):
        self.pose_gt.append(self.curr_frame["pose_gt"])
        self.timestampes.append(self.curr_frame["timestamp"])
        p2loss = 0
        tracking_success = True
        if self.use_gt_pose:
            pose_t1_w = self.pose_gt[-1]
        else:
            # initialize
            if not self.status["initialized"]:
                self.initialize_tracker()
                pose_t1_w = np.eye(4)
                # pose_t1_w = self.curr_frame["pose_gt"]
            else:
                if not self.orb_only:
                    pose_t1_t0, tracking_success = self.icp_tracker.predict_pose(self.curr_frame)
                if self.use_orb_backend:
                    pose_t1_t0 = np.eye(4)
                    pose_t1_w = self.refine_icp_pose(pose_t1_t0, tracking_success)
                else:
                    pose_t1_w = self.pose_es[-1] @ pose_t1_t0

        self.icp_tracker.move_last_status()
        self.pose_es.append(pose_t1_w)

        frame.updatePose(pose_t1_w)
        frame_map["vertex_map_w"] = transform_map(
            frame_map["vertex_map_c"], frame.get_c2w
        )
        frame_map["normal_map_w"] = transform_map(
            frame_map["normal_map_c"], get_rot(frame.get_c2w)
        )

        return tracking_success

    # 评估轨迹的绝对轨迹误差（ATE）
    def eval_total_ate(self, pose_es, pose_gt):
        ates = []
        for i in tqdm(range(1, len(pose_gt) + 1)):
            ates.append(self.eval_ate(pose_es, pose_gt, i))
        ates = np.array(ates)
        return ates

    # 保存ATE曲线图用于可视化
    def save_ate_fig(self, ates, save_path, save_name):
        plt.plot(range(len(ates)), ates)
        plt.ylim(0, max(ates) + 0.1)
        plt.title("ate:{}".format(ates[-1]))
        plt.savefig(os.path.join(save_path, "{}.png".format(save_name)))
    

    # 保存关键帧轨迹
    def save_keyframe_traj(self, save_file):
        if self.use_orb_backend:
            poses, stamps = convert_poses(self.orb_backend.get_keyframe_points())
            with open(save_file, "w") as f:
                for pose_id, pose_es_ in enumerate(poses):
                    t = pose_es_[:3, 3]
                    q = R.from_matrix(pose_es_[:3, :3])
                    f.write(str(stamps[pose_id]) + " ")
                    for i in t.tolist():
                        f.write(str(i) + " ")
                    for i in q.as_quat().tolist():
                        f.write(str(i) + " ")
                    f.write("\n")

    # 保存轨迹为TUM格式
    def save_traj_tum(self, save_file):
        poses, stamps = convert_poses(self.orb_backend.get_trajectory_points())
        with open(save_file, "w") as f:
            for pose_id, pose_es_ in enumerate(self.pose_es):
                t = pose_es_[:3, 3]
                q = R.from_matrix(pose_es_[:3, :3])
                f.write(str(stamps[pose_id]) + " ")
                for i in t.tolist():
                    f.write(str(i) + " ")
                for i in q.as_quat().tolist():
                    f.write(str(i) + " ")
                f.write("\n")

    # 保存ORB后端轨迹为TUM格式
    def save_orb_traj_tum(self, save_file):
        if self.use_orb_backend:
            poses, stamps = convert_poses(self.orb_backend.get_trajectory_points())
            with open(save_file, "w") as f:
                for pose_id, pose_es_ in enumerate(poses):
                    t = pose_es_[:3, 3]
                    q = R.from_matrix(pose_es_[:3, :3])
                    f.write(str(stamps[pose_id]) + " ")
                    for i in t.tolist():
                        f.write(str(i) + " ")
                    for i in q.as_quat().tolist():
                        f.write(str(i) + " ")
                    f.write("\n")

    # 保存轨迹（真实和估计）和误差评估结果
    def save_traj(self, save_path):
        save_traj_path = os.path.join(save_path, "save_traj")
        if not self.use_gt_pose and self.use_orb_backend:
            traj_history = self.orb_backend.get_trajectory_points()
            self.pose_es, _ = convert_poses(traj_history)
        pose_es = np.stack(self.pose_es, axis=0)
        pose_gt = np.stack(self.pose_gt, axis=0)
        ates_ba = self.eval_total_ate(pose_es, pose_gt)
        print("ate: ", ates_ba[-1])
        np.save(os.path.join(save_traj_path, "pose_gt.npy"), pose_gt)
        np.save(os.path.join(save_traj_path, "pose_es.npy"), pose_es)
        self.save_ate_fig(ates_ba, save_traj_path, "ate")

        plt.figure()
        plt.plot(pose_es[:, 0, 3], pose_es[:, 1, 3])
        plt.plot(pose_gt[:, 0, 3], pose_gt[:, 1, 3])
        plt.legend(["es", "gt"])
        plt.savefig(os.path.join(save_traj_path, "traj_xy.jpg"))
        
        if self.use_orb_backend:
            self.orb_backend.shutdown()
        
    # 评估单帧的ATE
    def eval_ate(self, pose_es, pose_gt, frame_id=-1):
        pose_es = np.stack(pose_es, axis=0)[:frame_id, :3, 3]
        pose_gt = np.stack(pose_gt, axis=0)[:frame_id, :3, 3]
        ate = eval_ate(pose_gt, pose_es)
        return ate

# Point3D = collections.namedtuple(
#     "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

# BaseImage = collections.namedtuple(
#     "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

class Tracker(Tracker_rtg):
    """
    Tracking main class.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
        eslam (ESLAM): ESLAM object
    """
    def __init__(self, cfg, args, eslam):

        super().__init__(cfg)

        self.save_path = cfg['save_path']
        self.frame_id = 0
        self.preload = cfg['preload']

        self.cfg = cfg
        self.args = args
        self.logger = eslam.logger
        # RTG：read dataset information
        if self.preload:
            self.dataset_cameras = eslam.dataset_rtg.scene_info.train_cameras
        else:
            self.dataset_cameras = None #############################
        self.scale = cfg['scale']
        self.gt_campose_list = []
        self.idx = eslam.idx
        self.bound = eslam.bound
        self.mesher = eslam.mesher
        self.output = eslam.output
        self.verbose = eslam.verbose
        self.renderer = eslam.renderer
        self.gt_c2w_list = eslam.gt_c2w_list
        self.mapping_idx = eslam.mapping_idx
        self.mapping_cnt = eslam.mapping_cnt
        self.estimate_c2w_list = eslam.estimate_c2w_list
        self.truncation = eslam.truncation
        self.tracker_keyframe_dict = []
        self.shared_keyframe_list_re = []
        self.shared_keyframe_list = []
        self.depth_map = eslam.depth_map


        self.cam_lr_T = cfg['tracking']['lr_T']
        self.cam_lr_R = cfg['tracking']['lr_R']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device :", self.device)
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.w_sdf_fs = cfg['tracking']['w_sdf_fs']
        self.w_sdf_center = cfg['tracking']['w_sdf_center']
        self.w_sdf_tail = cfg['tracking']['w_sdf_tail']
        self.w_depth = cfg['tracking']['w_depth']
        self.w_color = cfg['tracking']['w_color']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['tracking']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, shuffle=False,
                                       num_workers=1, pin_memory=True, prefetch_factor=2)

        self.visualizer = Frame_Visualizer(freq=cfg['tracking']['vis_freq'], n_imgs=self.n_img, inside_freq=cfg['tracking']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'tracking_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = eslam.H, eslam.W, eslam.fx, eslam.fy, eslam.cx, eslam.cy

        #self.num_frames = len(frame_ids)
        self.keyframe_device = cfg['keyframe_device']

        self.embedpos_fn = eslam.shared_embedpos_fn
        #self.embedpos_fn = eslam.embedpos_fn
        self.shared_all_planes_list = eslam.shared_all_planes_list
        self.shared_decoders_list = eslam.shared_decoders_list
        self.shared_cur_rf_id = eslam.shared_cur_rf_id
        self.pre_shared_cur_rf_id = self.shared_cur_rf_id[0].clone()
        self.update_para_flag = 1
        self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz = copy.deepcopy(self.shared_all_planes_list[self.shared_cur_rf_id[0]])

        for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
            for i, plane in enumerate(planes):
                plane = plane.to(self.device)
                #plane.share_memory_()
                planes[i] = plane

        for c_planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]:
            for i, plane in enumerate(c_planes):
                plane = plane.to(self.device)
                #plane.share_memory_()
                c_planes[i] = plane
        self.decoders_list = copy.deepcopy(self.shared_decoders_list)

        self.shared_decoders = eslam.shared_decoders_list[self.shared_cur_rf_id[0]]
        self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)   #.to(self.device)
     #  # ##########################
        self.Kf_depth = None
        self.Kf_color = None
        self.Kf_sdf = None
        self.Kf_z_vals = None
        self.Kf_gt_depth = None

        self.Kf_pre_depth = None
        self.Kf_pre_color = None
        self.kf_pre_sdf = None
        self.kf_pre_z_vals = None
        self.Kf_pre_gt_depth = None
        self.eps = 1e-3
        # p3D = self.load_data()
        # self.points3D = p3D
        #self.truncation = cfg['model']['truncation']
        #self.p3d = DataField.p3d
        self.crop_size = cfg['cam']['crop_edge'] if 'crop_edge' in cfg['cam'] else 0
        self.total_pixels = (self.H - self.crop_size*2) * (self.W - self.crop_size*2)
        # self.num_rays_to_save = int(self.total_pixels * cfg['mapping']['n_pixels'])
        self.num_rays_to_save = cfg['tracking']['pixels']
        self.keyframeDatabase = self.create_kf_database(cfg)
        #print(self.keyframeDatabase)

        # for p in self.decoders.parameters():
        #     p.requires_grad_(False)

    #2月29日 创建keyframe database        
    def create_kf_database(self, config):  
        '''
        Create the keyframe database
        '''
        num_kf = int(self.n_img // self.cfg['mapping']['gb_keyframe'] + 1)  
        print('#kf:', num_kf)
        print('#Pixels to save:', self.num_rays_to_save)
        return KeyFrameDatabase(config, 
                                self.H, 
                                self.W, 
                                num_kf, 
                                self.num_rays_to_save, 
                                self.device)



    # 加载3D点和图像数据，支持从二进制文件中读取点云和图像信息。它还处理点的可见性和关联性。
    # def load_data(factor=None, load_colmap_poses=True):
    #     if load_colmap_poses:
    #         ##
    #         class Image(BaseImage):
    #             def qvec2rotmat(self):
    #                 return qvec2rotmat(self.qvec)

    #         def qvec2rotmat(qvec):
    #             return np.array([
    #                 [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
    #                 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
    #                 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
    #                 [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
    #                 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
    #                 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
    #                 [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
    #                 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
    #                 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])
            
    #         def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    #             """Read and unpack the next bytes from a binary file.
    #             :param fid:
    #             :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    #             :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    #             :param endian_character: Any of {@, =, <, >, !}
    #             :return: Tuple of read and unpacked values.
    #             """
    #             data = fid.read(num_bytes)
    #             return struct.unpack(endian_character + format_char_sequence, data)

    #         def read_points3D_binary(path_to_model_file):##读取bin文件函数
    #             """
    #             see: src/base/reconstruction.cc
    #                 void Reconstruction::ReadPoints3DBinary(const std::string& path)
    #                 void Reconstruction::WritePoints3DBinary(const std::string& path)
    #             """
    #             points3D = {}
    #             with open(path_to_model_file, "rb") as fid:
    #                 num_points = read_next_bytes(fid, 8, "Q")[0]
    #                 for _ in range(num_points):
    #                     binary_point_line_properties = read_next_bytes(
    #                         fid, num_bytes=43, format_char_sequence="QdddBBBd")
    #                     point3D_id = binary_point_line_properties[0]
    #                     xyz = np.array(binary_point_line_properties[1:4])
    #                     rgb = np.array(binary_point_line_properties[4:7])
    #                     error = np.array(binary_point_line_properties[7])
    #                     track_length = read_next_bytes(
    #                         fid, num_bytes=8, format_char_sequence="Q")[0]
    #                     track_elems = read_next_bytes(
    #                         fid, num_bytes=8 * track_length,
    #                         format_char_sequence="ii" * track_length)
    #                     image_ids = np.array(tuple(map(int, track_elems[0::2])))
    #                     point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
    #                     points3D[point3D_id] = Point3D(
    #                         id=point3D_id, xyz=xyz, rgb=rgb,
    #                         error=error, image_ids=image_ids,
    #                         point2D_idxs=point2D_idxs)
    #             return points3D

    #         def read_images_binary(path_to_model_file):##读取bin文件函数
    #             """
    #             see: src/base/reconstruction.cc
    #                 void Reconstruction::ReadImagesBinary(const std::string& path)
    #                 void Reconstruction::WriteImagesBinary(const std::string& path)
    #             """
    #             images = {}
    #             with open(path_to_model_file, "rb") as fid:
    #                 num_reg_images = read_next_bytes(fid, 8, "Q")[0]
    #                 for _ in range(num_reg_images):
    #                     binary_image_properties = read_next_bytes(
    #                         fid, num_bytes=64, format_char_sequence="idddddddi")
    #                     image_id = binary_image_properties[0]
    #                     qvec = np.array(binary_image_properties[1:5])
    #                     tvec = np.array(binary_image_properties[5:8])
    #                     camera_id = binary_image_properties[8]
    #                     image_name = ""
    #                     current_char = read_next_bytes(fid, 1, "c")[0]
    #                     while current_char != b"\x00":  # look for the ASCII 0 entry
    #                         image_name += current_char.decode("utf-8")
    #                         current_char = read_next_bytes(fid, 1, "c")[0]
    #                     num_points2D = read_next_bytes(fid, num_bytes=8,
    #                                                 format_char_sequence="Q")[0]
    #                     x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
    #                                             format_char_sequence="ddq" * num_points2D)
    #                     xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
    #                                         tuple(map(float, x_y_id_s[1::3]))])
    #                     point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
    #                     images[image_id] = Image(
    #                         id=image_id, qvec=qvec, tvec=tvec,
    #                         camera_id=camera_id, name=image_name,
    #                         xys=xys, point3D_ids=point3D_ids)
    #             return images

    #         # Read 3D points 
    #         ##bin文件路径   /data0/wjy/ESLAM/Datasets/Apartment/colmap1
    #         image_dir = "./Datasets/Apartment/color"
    #         database_path = "./output/database.db"
    #         output_dir = "/data0/wjy/eslamnew/Datasets/Apartment/color/colmap"
    #         #reconstruction = self.run_pycolmap(image_dir, database_path, output_dir)
    #         points3D = read_points3D_binary('/home/sjtu/ESLAM/Datasets/Replica/room0/sparse/points3D.bin')  #scannet
    #         #points3D里包括点的id，xyz坐标，rgb，和对应存在的image的id
    #         images = read_images_binary('/home/sjtu/ESLAM/Datasets/Replica/room0/sparse/images.bin')
    #         #points3D = read_points3D_binary('/data0/wjy/ESLAM/Datasets/scannet/scene0000_00/colmap/sparse/0/points3D.bin')  #scannet
    #         #points3D里包括点的id，xyz坐标，rgb，和对应存在的image的id
    #         #images = read_images_binary('/data0/wjy/ESLAM/Datasets/scannet/scene0000_00/colmap/sparse/0/images.bin')
 
    #         #points3D = read_points3D_binary('/data0/wjy/ESLAM/Datasets/Replica/office0/colmap/sparse/0/points3D.bin') #replica
    #         #points3D里包括点的id，xyz坐标，rgb，和对应存在的image的id
    #         #images = read_images_binary('/data0/wjy/ESLAM/Datasets/Replica/office0/colmap/sparse/0/images.bin')
    #         #images里有一项也是对应是否出现过上面3D点的list，出现为点的id，不出现为-1
    #     #/data0/wjy/ESLAM/Datasets/Replica/room0/sparse/0
    #     p3D_ids = sorted(points3D.keys())
    #     p3D_id_to_idx = dict(zip(p3D_ids, range(len(points3D))))
    #     p3D_xyz = np.stack([points3D[i].xyz for i in p3D_ids])
    #     track_lengths = np.stack([len(points3D[i].image_ids) for i in p3D_ids])
    #     p3D_observed = []
    #     print(len(images))
    #     for i in range(1,len(images)+1):
    #         try:
    #             image = images[i]
    #             obs = np.stack([p3D_id_to_idx[i]] for i in image.point3D_ids if i != -1)
    #             p3D_observed.append(obs)
    #             p3D = {'points3D' : p3D_xyz, 'p3D_observed' : p3D_observed}
    #         except KeyError:
    #             pass  # 当发生 KeyError 时，什么也不做，直接跳过
    #         #image = images[i]
    #         #obs = np.stack([p3D_id_to_idx[i]] for i in image.point3D_ids if i != -1)
    #         #p3D_observed.append(obs)
    #     #p3D = {'points3D' : p3D_xyz, 'p3D_observed' : p3D_observed}
    #     #P3D_xyz是所有3D点的坐标， p3D_observed是每张图片对应观察到的点的id

    #     return p3D
    
    # def run_pycolmap(image_dir, database_path, output_dir):
    #     # 特征提取
    #     pycolmap.extract_features(database_path=database_path, image_path=image_dir)
    #     print("Features extracted.")

    #     # 特征匹配
    #     pycolmap.match_features(database_path=database_path)
    #     print("Features matched.")

    #     # 稀疏重建
    #     sparse_dir = f"{output_dir}/sparse"
    #     reconstruction = pycolmap.ReconstructionMapper(
    #         database_path=database_path,
    #         image_path=image_dir,
    #         output_path=sparse_dir
    #     )
    #     print("Sparse reconstruction completed.")
    #     return reconstruction
    
    def c2w_to_w2c(self,c2w):
        w2c = np.linalg.inv(c2w.cpu())
        #w2c = np.linalg.inv(c2w)
        return w2c
    '''
    def load_ref_img(self, idx, data={}):
        if self.random_ref:
            if idx==0:
                ref_idx = self.N_imgs-1
            else:
                #ran_idx = random.randint(1, min(self.random_ref, self.N_imgs-idx-1))
                ran_idx = 1
                ref_idx = idx - ran_idx
        image = self.imgs[ref_idx]
        if self.dpt_depth is not None:
            dpt = self.dpt_depth[ref_idx]
            data['ref_dpts'] = dpt
        if self.use_DPT:
            data_in = {"image": np.transpose(image, (1, 2, 0))}
            data_in = self.transform(data_in)
            normalised_ref_img = data_in['image']
            data['normalised_ref_img'] = normalised_ref_img
        if self.with_depth:
            depth = self.depth[ref_idx]
            data['ref_depths'] = depth
        data['ref_imgs'] = image
        data['ref_idxs'] = ref_idx
        ##

        data['T_ref'] = torch.from_numpy(self.c2w_to_w2c(self.c2ws[ref_idx]))
    '''
    '''
    def load_T_r2q(self, data):
        #print("data shape:",data['T_ref'].shape)
        #data T ref 是 reference的绝对位姿
        c2w_ref = torch.from_numpy(self.c2w_to_w2c(data['T_ref']))
        #T_r2q_gt = data['T_tgt'] @ c2w_ref
        gt_c2w = self.gt_c2w.to('cpu')
        T_r2q_gt = torch.matmul(gt_c2w, c2w_ref)
        #T_r2q_gt = mat_to_Pose(T_r2q_gt)
        data['T_r2q_gt'] = T_r2q_gt
        #data['T_r2q_init'].append(optimizer)
        return data['T_r2q_gt']
    '''
    def load_p3d(self, data={}):
        def T_mul_p3d(T, p3d):
            T = T.float()
            p3d = torch.tensor(p3d).float()
            points3D = torch.cat((p3d, torch.ones(p3d.shape[0], 1)), dim=1)
            mul = torch.matmul(T, points3D.t())
            #print("Shape of mul:", mul.shape)
            mul = mul.squeeze(dim=0)
            p3D = mul.t()[:, :3]
            return p3D
        
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        fx = torch.tensor(fx).unsqueeze(0)  # 将焦距信息扩展为2维张量
        fy = torch.tensor(fy).unsqueeze(0)
        cx = torch.tensor(cx).unsqueeze(0)
        cy = torch.tensor(cy).unsqueeze(0)
        f = torch.stack([fx, fy], dim=-1)  # 将焦距信息合并成一个张量
        c = torch.stack([cx, cy], dim=-1)  # 将主点信息合并成一个张量

        def get_valid(p3d):
            fx = self.fx
            fy = self.fy
            cx = self.cx
            cy = self.cy
            fx = torch.tensor(fx)  # 将焦距信息扩展为2维张量
            fy = torch.tensor(fy)
            cx = torch.tensor(cx)
            cy = torch.tensor(cy)
            c = torch.stack([cx, cy], dim=-1)
            f = torch.stack([fx, fy], dim=-1)
            eps = 0.001
            #c = torch.tensor([[318.905426,242.683609]])
            #f = 600.0
            size = torch.tensor([[480,640]])
            #project
            z = p3d[..., -1]
            visible = z > eps
            z = z.clamp(min=eps)
            p2d = p3d[..., :-1] / z.unsqueeze(-1)
            ##denormalize
            p2d = p2d * f + c
            ##in_image
            in_image = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)

            valid = visible & in_image
            return valid

        select_idx = self.selected_keyframe_idx
        #print("select_idx:",select_idx)
        if select_idx == 0 :
            self.random_index = 0
        else :
            self.random_index = torch.randint(0, select_idx, (1,))#随机选取reference的图片
        #print("random_index:",self.random_index)
        #random_index = random_index.item() / 8
        #print("idx:",random_index)
        data['T_ref'] = torch.from_numpy(self.c2w_to_w2c(self.gt_campose_list[self.random_index]))#将reference的Tgt 转置
        #print("data shape:",data['T_ref'].shape)
        p3D = self.points3D['points3D']
        #print("self.points3D:",self.points3D['p3D_observed'][2])
        obs = self.points3D['p3D_observed'][self.random_index]#选取ref图像对应的观察到的点id
        #print("obs:", obs)
        valid = get_valid(T_mul_p3d(data['T_ref'], p3D[obs].squeeze()))#判断点是否在图像上并且可视
        #print("valid:", valid)
        obs = obs[valid.numpy()].squeeze()
        if obs.shape:
            length = obs.shape[0]
        else:
            length = 1
        #print("obs_valid:", obs)
        #print(type(obs))
        #print(length)
        max_num_points3D = 512#设定每张图片最多观察到的点数
        if length > 1:
            obs = np.random.choice(obs, max_num_points3D)#如果大于观察到的点数则随机选取
            data['points3D'] = T_mul_p3d(data['T_ref'], p3D[obs])#得到对应点的 to w 坐标
            #print("data['points3D']:",data['points3D'])
            return data
        else:
            return data

    
    def sdf_losses(self, sdf, z_vals, gt_depth):
        """
        Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

        Args:
        - self: instance of the class containing this method
        - sdf: a tensor of shape (R, N) representing the SDF values
        - z_vals: a tensor of shape (R, N) representing the depth values
        - gt_depth: a tensor of shape (R,) containing the ground truth depth values

        Returns:
        - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
        """

        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

        return sdf_losses

    def fs_losses(self, sdf, z_vals, gt_depth):
        """
        Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

        Args:
        - self: instance of the class containing this method
        - sdf: a tensor of shape (R, N) representing the SDF values
        - z_vals: a tensor of shape (R, N) representing the depth values
        - gt_depth: a tensor of shape (R,) containing the ground truth depth values

        Returns:
        - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
        """

        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

        #sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

        return self.w_sdf_fs * fs_loss

    def project(self, p3d):
        '''Project 3D points into the camera plane and check for visibility.将3D点投影到相机平面，并检查其是否可见'''
        z = p3d[..., -1]
        #print("pro p3d----------------------", p3d.shape)
        valid = z > self.eps
        z = z.clamp(min=self.eps)
        p2d = p3d[..., :2] / z.unsqueeze(-1)
        #print("pro p2d----------------------", p2d)
        return p2d, valid
    

    def undistort_points(self, pts, dist):
        '''
        去畸变
        Undistort normalized 2D coordinates
        and check for validity of the distortion model.
        '''
        dist = torch.tensor(dist).unsqueeze(-2)
        #dist = dist.unsqueeze(-2)  # add point dimension
        ndist = dist.shape[-1]
        undist = pts
        valid = torch.ones(pts.shape[:-1], device=pts.device, dtype=torch.bool)
        if ndist > 0:
            k1, k2 = dist[..., :2].split(1, -1)
            r2 = torch.sum(pts**2, -1, keepdim=True)
            radial = k1*r2 + k2*r2**2
            undist = undist + pts * radial

            # The distortion model is supposedly only valid within the image
            # boundaries. Because of the negative radial distortion, points that
            # are far outside of the boundaries might actually be mapped back
            # within the image. To account for this, we discard points that are
            # beyond the inflection point of the distortion model,
            # e.g. such that d(r + k_1 r^3 + k2 r^5)/dr = 0
            limited = ((k2 > 0) & ((9*k1**2-20*k2) > 0)) | ((k2 <= 0) & (k1 > 0))
            limit = torch.abs(torch.where(
                k2 > 0, (torch.sqrt(9*k1**2-20*k2)-3*k1)/(10*k2), 1/(3*k1)))
            valid = valid & torch.squeeze(~limited | (r2 < limit), -1)

            if ndist > 2:
                p12 = dist[..., 2:]
                p21 = p12.flip(-1)
                uv = torch.prod(pts, -1, keepdim=True)
                undist = undist + 2*p12*uv + p21*(r2 + 2*pts**2)
                # TODO: handle tangential boundaries

        return undist, valid
    
    def dist(self) -> torch.Tensor:
        self._data = tuple(self._data)
        '''Distortion parameters, with shape (..., {0, 2, 4}).返回畸变参数'''
        return self._data[6:]
    
    def undistort(self, pts):
        '''Undistort normalized 2D coordinates
           and check for validity of the distortion model.
        '''
        pts = pts.squeeze(dim=0)
        #print("Shape of pts:", pts.shape)
        assert pts.shape[-1] == 2
        #assert pts.shape[:-2] == self.shape  # allow broadcasting
        return self.undistort_points(pts, self.dist())
    
    
    def denormalize(self, p2d: torch.Tensor) -> torch.Tensor:
        '''Convert normalized 2D coordinates into pixel coordinates.将归一化的2D坐标转换为像素坐标。'''
        # Extract focal lengths (fx, fy) and principal points (cx, cy)
        #f = torch.tensor(self._data)[..., 2:4]
        #c = torch.tensor(self._data)[..., 4:6]
        #f = self._data[..., 2:4]
        #c = self._data[..., 4:6]
        # 获取焦距和主点信息
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        fx = torch.tensor(fx).unsqueeze(0)  # 将焦距信息扩展为2维张量
        fy = torch.tensor(fy).unsqueeze(0)
        cx = torch.tensor(cx).unsqueeze(0)
        cy = torch.tensor(cy).unsqueeze(0)
        f = torch.stack([fx, fy], dim=-1)  # 将焦距信息合并成一个张量
        c = torch.stack([cx, cy], dim=-1)  # 将主点信息合并成一个张量
        #p2d = p2d.view(6, 2)
        #print("f: ", f)
        #print("c: ", c)
        #print("p2d: ", p2d)
        ###################p2d * self.f.unsqueeze(-2) + self.c.unsqueeze(-2)
        return p2d * f+ c
    
    def in_image(self, p2d: torch.Tensor):
        '''Check if 2D points are within the image boundaries.检查2D点是否在图像边界内。'''
        assert p2d.shape[-1] == 2
        # assert p2d.shape[:-2] == self.shape  # allow broadcasting
        W = torch.tensor([self.W])
        H = torch.tensor([self.H])
        #print("W: ", W)
        #print("H: ", H)

        # 使用 torch.cat 进行连接
        size = torch.cat((W.unsqueeze(0), H.unsqueeze(0)), dim=1)
        #size = self.size.unsqueeze(-2)

        valid = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
        #print("valid: ", valid)
        #valid = valid[:4]
        return valid

    def world2image(self, p3d):
        '''Transform 3D points into 2D pixel coordinates.将3D点转换为2D像素坐标，并检查其有效性。'''
        p2d, visible = self.project(p3d)
        p2d, mask = self.undistort(p2d)
        p2d = self.denormalize(p2d)
        #print("visible: ", visible)
        #print("mask: ", mask)
        valid = visible & mask & self.in_image(p2d)
        return p2d, valid
    

    def reprojection_error(self, T_r2q, p2D_q_gt, mask):
        """计算重投影误差"""
        p2D_q, _ = self.world2image(T_r2q)
        err = torch.sum((p2D_q_gt - p2D_q)**2, dim=-1)
        #err = self.scaled_barron(1., 2.)(err)[0]/4
        err = self.masked_mean(err, mask, -1)
        return err
    
    def masked_mean(self, x, mask, dim):
        """计算带掩码的均值，用于处理部分有效数据。"""
        mask = mask.float()
        #print("x: ",x)
        return (mask * x).sum(dim) / mask.sum(dim).clamp(min=1)
    
    ###分割线 下面是loss的实现###

    def scaled_loss(self, x, fn, a):
        """Apply a loss function to a tensor and pre- and post-scale it.
        Args:
            x: the data tensor, should already be squared: `x = y**2`.
            fn: the loss function, with signature `fn(x) -> y`.
            a: the scale parameter.
        Returns:
            The value of the loss, and its first and second derivatives.
        """
        a2 = a**2
        loss, loss_d1, loss_d2 = fn(x/a2)
        return loss*a2, loss_d1, loss_d2/a2

    def barron_loss(self,x, alpha, derivatives: bool = True, eps: float = 1e-7):
        """
        实现Barron的自适应鲁棒损失函数，用于处理噪声数据。
        Parameterized  & adaptive robust loss function.
        Described in:
            A General and Adaptive Robust Loss Function, Barron, CVPR 2019

        Contrary to the original implementation, assume the the input is already
        squared and scaled (basically scale=1). Computes the first derivative, but
        not the second (TODO if needed).
        """
        loss_two = x
        loss_zero = 2 * torch.log1p(torch.clamp(0.5*x, max=33e37))

        # The loss when not in one of the above special cases.
        # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
        beta_safe = torch.abs(alpha - 2.).clamp(min=eps)
        # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
        alpha_safe = torch.where(
            alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha))
        alpha_safe = alpha_safe * torch.abs(alpha).clamp(min=eps)

        loss_otherwise = 2 * (beta_safe / alpha_safe) * (
            torch.pow(x / beta_safe + 1., 0.5 * alpha) - 1.)

        # Select which of the cases of the loss to return.
        loss = torch.where(
            alpha == 0, loss_zero,
            torch.where(alpha == 2, loss_two, loss_otherwise))
        dummy = torch.zeros_like(x)

        if derivatives:
            loss_two_d1 = torch.ones_like(x)
            loss_zero_d1 = 2 / (x + 2)
            loss_otherwise_d1 = torch.pow(x / beta_safe + 1., 0.5 * alpha - 1.)
            loss_d1 = torch.where(
                alpha == 0, loss_zero_d1,
                torch.where(alpha == 2, loss_two_d1, loss_otherwise_d1))

            return loss, loss_d1, dummy
        else:
            return loss, dummy, dummy


    def scaled_barron(self, a, c):
        return lambda x: self.scaled_loss(
                x, lambda y: self.barron_loss(y, y.new_tensor(a)), c)

    def optimize_tracking(self, idx, cam_iter, cam_pose, gt_color, gt_depth, batch_size, optimizer,gt_R,gt_T,gt_cam_pose,current_rays,poses_all):
        """
        Do one iteration of camera tracking. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            cam_iter:
            cam_pose (tensor): camera pose.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """

        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)

        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        c2w = cam_pose_to_matrix(cam_pose)  #每一个keyframe的cam pose应该存下来，用于获取c2w，再将其c2w传入后面的get samples中

        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(self.ignore_edge_H, H-self.ignore_edge_H,
                                                                                 self.ignore_edge_W, W-self.ignore_edge_W,
                                                                                 batch_size, H, W, fx, fy, cx, cy, c2w,
                                                                                 gt_depth, gt_color, device)

        with torch.no_grad():
            det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(
                device) - det_rays_o) / det_rays_d
            t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            inside_mask = t >= batch_gt_depth
            inside_mask = inside_mask & (batch_gt_depth > 0)

        batch_rays_d = batch_rays_d[inside_mask]
        batch_rays_o = batch_rays_o[inside_mask]
        batch_gt_depth = batch_gt_depth[inside_mask]
        batch_gt_color = batch_gt_color[inside_mask]

        if idx % self.cfg['mapping']['gb_keyframe'] == 0:
            self.shared_keyframe_list.append(idx)
            self.tracker_keyframe_dict.append({'rays_o': batch_rays_o.to('cpu'),'gt_depth': batch_gt_depth.to('cpu'),
                                        'rays_d': batch_rays_d.to('cpu')})
            
        depth, color, sdf, z_vals = self.renderer.render_batch_ray(all_planes, self.decoders, batch_rays_d, batch_rays_o,
                                                                   device, self.truncation, gt_depth=batch_gt_depth)
        depth_error = (batch_gt_depth - depth.clone().detach()).abs()
        error_median = depth_error.median()
        depth_mask = (depth_error < 10 * error_median)
        loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])
        # loss = loss + self.fs_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])
        loss = loss + self.w_color * torch.square(batch_gt_color - color)[depth_mask].mean()
        loss = loss + self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()

        #2月29日,Coslam的global ba写法,by wjy
        # if idx.item() % self.cfg['mapping']['gb_keyframe'] == 0: 
        #     #print("enter global ba")
        #     rays, ids = self.keyframeDatabase.sample_global_rays(self.cfg['mapping']['sample'])#
        #     idx_cur = random.sample(range(0, self.H * self.W),max(self.cfg['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.cfg['mapping']['min_pixels_cur']))
        #     current_rays_batch = current_rays[idx_cur, :]
        #     rays = torch.cat([rays.to(device), current_rays_batch.to(device)], dim=0) 
        #     ids_all = torch.cat([ids//self.cfg['mapping']['gb_keyframe'], -torch.ones((len(idx_cur)))]).to(torch.int64)
        #     rays_d_cam = rays[..., :3].to(device)
        #     target_s = rays[..., 3:6].to(device)
        #     target_d_1 = rays[..., 6:7].to(device)

        #     rays_d_1 = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
        #     rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d_1.shape[1], 1).reshape(-1, 3)  #此行代码导致了cuda错误
        #     rays_d = rays_d_1.reshape(-1, 3)
        #     target_d = target_d_1.squeeze(-1)
        #     with torch.no_grad():
        #         det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
        #         det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
        #         t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d
        #         t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
        #         inside_mask = t >= target_d.clone().detach()
        #         inside_mask = inside_mask & (target_d.clone().detach() > 0)
        #     co_batch_rays_d = rays_d[inside_mask]
        #     co_batch_rays_o = rays_o[inside_mask]
        #     co_batch_gt_depth = target_d[inside_mask]
        #     co_batch_gt_color = target_s[inside_mask]

        #     codepth, cocolor, cosdf, coz_vals = self.renderer.render_batch_ray(all_planes, self.decoders, co_batch_rays_d, co_batch_rays_o,
        #                                                            device, self.truncation, gt_depth=co_batch_gt_depth)

        #     depth_error = (co_batch_gt_depth.clone().detach() - codepth.clone().detach()).abs()
        #     error_median = depth_error.median()
        #     depth_mask = (depth_error < 10 * error_median)
        #     loss = loss + self.w_color * torch.square(co_batch_gt_color.clone().detach() - cocolor.clone().detach())[depth_mask].mean()
        #     loss = loss + self.w_depth * torch.square(co_batch_gt_depth[depth_mask].clone().detach() - codepth[depth_mask].clone().detach()).mean()
        #     #print("exit global ba")
        
        # if idx.item() % self.cfg['mapping']['res_keyframe'] == 0 and idx.item() > self.cfg['mapping']['res_keyframe']:
        #     selected_keyframe_idx = random.choice(self.shared_keyframe_list_re)
        #     selected_keyframe_idx = self.shared_keyframe_list_re.index(selected_keyframe_idx)#

        #     self.selected_keyframe_idx=selected_keyframe_idx
        #     data = self.load_p3d()#获取了对应点的 to w 坐标
        #     self._data = data
        #     c2w_ref = torch.from_numpy(self.c2w_to_w2c(data['T_ref']))

        #     T_tgt = torch.inverse(self.gt_c2w)
        #     T_tgt = T_tgt.to('cpu')

        #     T_r2q_gt = torch.matmul(T_tgt, c2w_ref)
        #     data['T_r2q_gt'] = T_r2q_gt
        #     p2D_q_gt, mask = self.world2image(data['T_r2q_gt'])

        #     T_te = torch.inverse(self.estimate_c2w_list[idx-1])
        #     T_te = T_te.to('cpu')
        #     T_r2q = torch.matmul(T_te, c2w_ref) #Tidx 估计 @ Tidx-8 gt 估计？

        #     p2D_q_i, mask_i = self.world2image(T_r2q)
        #     mask = (mask & mask_i).float()

        #     err = self.reprojection_error(p2D_q_i,p2D_q_gt,mask).clamp(max=50)
        #     err = err.to('cuda:0')
        #     reprow = 10
        #     loss = loss + reprow * err
            
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        return loss.item()
    
    def optimize_tracking_eslam(self, cam_pose, gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera tracking. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            cam_pose (tensor): camera pose.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        c2w = cam_pose_to_matrix(cam_pose)
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(self.ignore_edge_H, H-self.ignore_edge_H,
                                                                                 self.ignore_edge_W, W-self.ignore_edge_W,
                                                                                 batch_size, H, W, fx, fy, cx, cy, c2w,
                                                                                 gt_depth, gt_color, device)

        # should pre-filter those out of bounding box depth value
        with torch.no_grad():
            det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(
                device) - det_rays_o) / det_rays_d
            t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            inside_mask = t >= batch_gt_depth
            inside_mask = inside_mask & (batch_gt_depth > 0)

        batch_rays_d = batch_rays_d[inside_mask]
        batch_rays_o = batch_rays_o[inside_mask]
        batch_gt_depth = batch_gt_depth[inside_mask]
        batch_gt_color = batch_gt_color[inside_mask]

        depth, color, sdf, z_vals = self.renderer.render_batch_ray(all_planes, self.decoders, batch_rays_d, batch_rays_o,
                                                                   self.device, self.truncation, gt_depth=batch_gt_depth)

        ## Filtering the rays for which the rendered depth error is greater than 10 times of the median depth error
        depth_error = (batch_gt_depth - depth.detach()).abs()
        error_median = depth_error.median()
        depth_mask = (depth_error < 10 * error_median)

        ## SDF losses
        loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])

        ## Color Loss
        loss = loss + self.w_color * torch.square(batch_gt_color - color)[depth_mask].mean()

        ### Depth loss
        loss = loss + self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


    def update_params_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.
        """

        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')

            self.decoders.load_state_dict(self.shared_decoders_list[self.shared_cur_rf_id[0]].state_dict())
            shared_all_planes = self.shared_all_planes_list[self.shared_cur_rf_id[0]]
            self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz, self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz = shared_all_planes


            for planes, self_planes in zip(
                    [self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz],
                    [self.planes_xy, self.planes_xz, self.planes_yz]):
                for i, plane in enumerate(planes):
                    self_planes[i] = plane.detach().to(self.device)

            for c_planes, self_c_planes in zip(
                    [self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz],
                    [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]):
                for i, c_plane in enumerate(c_planes):
                    self_c_planes[i] = c_plane.detach().to(self.device)

         ##########################
            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def axis_angle_to_matrix(self, data):
        """
        将轴角表示为旋转矩阵
        """
        batch_dims = data.shape[:-1]

        theta = torch.norm(data, dim=-1, keepdim=True)
        omega = data / theta

        omega1 = omega[...,0:1]
        omega2 = omega[...,1:2]
        omega3 = omega[...,2:3]
        zeros = torch.zeros_like(omega1)

        K = torch.concat([torch.concat([zeros, -omega3, omega2], dim=-1)[...,None,:],
                        torch.concat([omega3, zeros, -omega1], dim=-1)[...,None,:],
                        torch.concat([-omega2, omega1, zeros], dim=-1)[...,None,:]], dim=-2)
        I = torch.eye(3).expand(*batch_dims,3,3).to(data)

        return I + torch.sin(theta).unsqueeze(-1) * K + (1. - torch.cos(theta).unsqueeze(-1)) * (K @ K)   
    
    def matrix_from_tensor(self, rot, trans):
        """
        从旋转和平移张量生成变换矩阵
        """
        bs = rot.shape[0]
        T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
        R = self.axis_angle_to_matrix(rot)
        T[:, :3, :3] = R
        T[:, :3, 3] = trans
        return T
    
    def get_camera_rays(self, H, W, fx, fy=None, cx=None, cy=None, type='OpenGL'):
        """Get ray origins, directions from a pinhole camera."""
        #  ----> i
        # |
        # |
        # X
        # j
        i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                        torch.arange(H, dtype=torch.float32), indexing='xy')
        
        # View direction (X, Y, Lambda) / lambda
        # Move to the center of the screen
        #  -------------
        # |      y      |
        # |      |      |
        # |      .-- x  |
        # |             |
        # |             |
        #  -------------

        if cx is None:
            cx, cy = 0.5 * W, 0.5 * H

        if fy is None:
            fy = fx
        if type is  'OpenGL':
            dirs = torch.stack([(i - cx)/fx, -(j - cy)/fy, -torch.ones_like(i)], -1)
        elif type is 'OpenCV':
            dirs = torch.stack([(i - cx)/fx, (j - cy)/fy, torch.ones_like(i)], -1)
        else:
            raise NotImplementedError()

        rays_d = dirs
        return rays_d
    
    def flip_axes(self, estimate):
        # 构造一个对角矩阵，Y,Z轴取反
        flip_mat = torch.diag(torch.tensor([1, -1, -1], dtype=estimate.dtype, device=estimate.device))
        estimate = estimate.squeeze(0)
        R = estimate[:3, :3]
        t = estimate[:3, 3]
        # print("es: ", estimate, estimate.shape)
        # print("R:", R)
        # print("t:", t)

        R_new = R @ flip_mat
        # t_new = flip_mat @ t

        new_pose = torch.eye(4, dtype=estimate.dtype, device=estimate.device)
        new_pose[:3, :3] = R_new
        new_pose[:3, 3] = t
        return new_pose
    
    def flip_rows(self, c2w):
        c2w = c2w.clone()
        c2w[1, :] *= -1
        c2w[2, :] *= -1
        return c2w



    def matrix_to_axis_angle(self, rot):
        """
        :param rot: [N, 3, 3]
        :return:
        """
        return quaternion_to_axis_angle(matrix_to_quaternion(rot))
    
    def tensor_to_pil(self, tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.squeeze(0)  # 去掉 batch 维 [1, H, W, 3]
            tensor = tensor.detach().cpu().numpy()  # 转为 numpy
            if tensor.max() <= 1.0:
                tensor = (tensor * 255).astype(np.uint8)
            else:
                tensor = tensor.astype(np.uint8)
            return Image.fromarray(tensor)
        raise TypeError("Input must be a torch.Tensor")

    def getNextFrame(self):
        frame_info = self.dataset_cameras[self.frame_id]
        # print(frame_info)
        if isinstance(self.cfg, dict):
            args = SimpleNamespace(**self.cfg)
        frame = loadCam(args, self.frame_id, frame_info, 1)
        print("get frame: {}".format(self.frame_id))
        self.frame_id += 1
        return frame

    # 对输入的深度图和法线图进行预处理
    def map_preprocess_mp(self, frame, frame_id):
        self.map_input = super().map_preprocess(frame, frame_id)

    def run(self):
        """
            Runs the tracking thread for the input RGB-D frames.

            Args:
                None

            Returns:
                None
        """
        # 初始化orb后端
        self.initialize_orb()

        tracking_time_sum = 0.0

        device = self.device
        batch ={}
        current_rays = torch.tensor(0.) 
        poses_all = torch.tensor(0.) 
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader, smoothing=0.05)
        for idx, gt_color, gt_depth, gt_c2w, cam_info in self.frame_loader:
            if not self.preload:
                crop_edge = 0                      #裁减会影响跟踪性能，渲染需要裁减。
                image_color = Image.open(cam_info["image_path"][0])
                image_depth = (
                    np.asarray(Image.open(cam_info["depth_path"][0]), dtype=np.float32) / self.cfg["depth_scale"]
                )
                image_color = np.asarray(
                    image_color.resize((image_depth.shape[1], image_depth.shape[0]))
                )
                if crop_edge > 0:
                    image_color = image_color[
                        crop_edge:-crop_edge,
                        crop_edge:-crop_edge,
                        :,
                    ]
                    image_depth = image_depth[
                        crop_edge:-crop_edge,
                        crop_edge:-crop_edge,
                    ]
                frame_info = CameraInfo(
                    uid=int(cam_info["uid"].item()), 
                    R=cam_info["R"].squeeze().cpu().numpy(), 
                    T=cam_info["T"].squeeze().cpu().numpy(), 
                    FovY=cam_info["FovY"].item(), 
                    FovX=cam_info["FovX"].item(), 
                    image=Image.fromarray(image_color), 
                    image_path=cam_info["image_path"][0], 
                    image_name=cam_info["image_name"][0], 
                    width=int(cam_info["width"].item()), 
                    height=int(cam_info["height"].item()), 
                    depth=Image.fromarray(image_depth), 
                    depth_path=cam_info["depth_path"][0], 
                    # pose_gt=cam_info["uid"], 
                    cx=float(cam_info["cx"].item()), 
                    cy=float(cam_info["cy"].item()), 
                    depth_scale=float(cam_info["depth_scale"].item()), 
                    timestamp=float(cam_info["timestamp"].item()), 
                )
                # print("cam_info: ", frame_info)
                if isinstance(self.cfg, dict):
                    args = SimpleNamespace(**self.cfg)
                    frame = loadCam(args, self.frame_id, frame_info, 1)
                # print("get frame: {}".format(self.frame_id))
                self.frame_id += 1

            else:
                # frame数据接口
                frame = self.getNextFrame()

            if frame is None:
                raise RuntimeError("No frame received from the frame loader.")
            frame_id = frame.uid
            # if frame_id == idx[0]:
            #     print("frame_id == idx[0]")
            # print("current tracker frame = %d" % idx)
            # update current map

            move_to_gpu(frame)

            tracking_start_time = time.time()
            self.map_preprocess_mp(frame, frame_id)
            success = self.tracking(frame, self.map_input)
            tracking_ending_time = time.time()
            # print("render_depth: ", self.depth_map.unsqueeze(-1).shape, self.depth_map.device)             # [H, W]
            # print("frame_depth: ", self.map_input["depth_map"].shape, self.map_input["depth_map"].device)  # [H, W, 1]
            # print("normal_map_w: ", self.map_input["normal_map_w"].shape, self.map_input["normal_map_w"].device)  # [H, W, 3]
            # if not self.orb_only:
            #     self.update_last_status(frame, self.depth_map.unsqueeze(-1).clone().to(self.device), self.map_input["depth_map"].clone(), 
            #                         self.map_input["normal_map_w"].clone())
            self.map_input["frame"] = copy.deepcopy(frame)
            self.map_input["frame"] = frame
            self.map_input["poses_new"] = self.get_new_poses()
            move_to_cpu(frame)

            gt_color = gt_color.to(device, non_blocking=True)
            gt_depth = gt_depth.to(device, non_blocking=True)
            gt_c2w = gt_c2w.to(device, non_blocking=True)
            #batch['direction'] = torch.ones(1, self.H, self.W, 3)  #应该等于self.rays_d
            # batch['rgb'] = gt_color
            # batch['depth'] = gt_depth
            # batch['frame_id'] = idx
            # self.rays_d = self.get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy).unsqueeze(0)     #(1, H, W, 3)
            # batch['direction'] = self.rays_d.to(device, non_blocking=True)  #应该等于self.rays_d
            # if not self.verbose:
            #     pbar.set_description(f"Tracking Frame {idx[0]} / {self.n_img}")
            idx = idx[0]

            # initiate mapping every self.every_frame frames
            if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                while self.mapping_idx[0] != idx - 1:
                    time.sleep(0.001)
                pre_c2w = self.estimate_c2w_list[idx - 1].unsqueeze(0).to(device)   #获取上一帧位姿

                self.update_para_flag = 1

            if self.pre_shared_cur_rf_id != self.shared_cur_rf_id[0]:      #检查参考帧是否发生变化如果变化则不更新场景参数
                self.update_para_flag = 0

            if self.update_para_flag == 1:
                self.update_params_from_mapping()

            self.update_params_from_mapping()
            self.pre_shared_cur_rf_id = self.shared_cur_rf_id[0].clone()    #更新共享参考帧ID

            all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)

            if not self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame:  ",  idx.item(), "/", self.n_img)
                print(Style.RESET_ALL)
            new_poses = None
            if idx == 0 or self.gt_camera:
                c2w = gt_c2w
                # if not self.no_vis_on_first_frame:
                #     self.visualizer.save_imgs(idx, 59, gt_depth, gt_color, c2w.squeeze(), all_planes, self.decoders)
            else:       #恒速假设
                """
                if self.const_speed_assumption and idx - 2 >= 0:
                    ## Linear prediction for initialization
                    pre_poses = torch.stack([self.estimate_c2w_list[idx - 2], pre_c2w.squeeze(0)], dim=0)
                    pre_poses = matrix_to_cam_pose(pre_poses)
                    cam_pose = 2 * pre_poses[1:] - pre_poses[0:1]
                else:   #用上一帧位姿
                    ## Initialize with the last known pose
                    cam_pose = matrix_to_cam_pose(pre_c2w) #四元数+三维平移向量

                T = torch.nn.Parameter(cam_pose[:, -3:].clone())
                R = torch.nn.Parameter(cam_pose[:,:4].clone())
                gt_R = gt_c2w[:, :3, :3].clone()
                gt_T = gt_c2w[:, :3, 3].clone()
                #print("gt_R:",gt_R)
                cam_para_list_T = [T]
                cam_para_list_R = [R]
                optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr_T, 'betas':(0.5, 0.999)},
                                                     {'params': cam_para_list_R, 'lr': self.cam_lr_R, 'betas':(0.5, 0.999)}])

                current_min_loss = torch.tensor(float('inf')).float().to(device)

                ####添加到keyframedatabase,进行global的先前准备
                #if idx.item()-1  == 0:
                    #self.keyframeDatabase.add_keyframe(batch, filter_depth=False)
                # 每self.cfg['mapping']['gb_keyframe']帧进行一次全局BA
                if idx.item() % self.cfg['mapping']['gb_keyframe'] == 0:
                    self.keyframeDatabase.add_keyframe(batch, filter_depth=False)
                    poses = torch.stack([self.estimate_c2w_list[i] for i in range(0, idx.item()-1, self.cfg['mapping']['gb_keyframe'])])

                    if len(self.keyframeDatabase.frame_ids) < 2:
                        poses_fixed = torch.nn.parameter.Parameter(poses).to(device)
                        current_pose = self.estimate_c2w_list[idx-1][None,...]
                        poses_all = torch.cat([poses_fixed, current_pose], dim=0)
                    else:
                        poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(device)
                        current_pose = self.estimate_c2w_list[idx-1][None,...]
                        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
                        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_axis_angle(poses[:, :3, :3]))
                        pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(device)#获取当前帧姿态参数的优化器

                        poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)#将优化后的姿态参数添加到poses_all中
                        print("poses_all:",poses_all.size())

                    current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)#根据采样策略选择采样的射线，包括来自关键帧的射线和当前帧的射线
                    current_rays = current_rays.reshape(-1, current_rays.shape[-1])#reshape成张量

                if idx.item() % self.cfg['mapping']['res_keyframe'] == 0:
                    self.gt_campose_list.append(gt_c2w) #只储存keyframe的campose

                    self.shared_keyframe_list_re.append(idx)
                for cam_iter in range(self.num_cam_iters):
                    cam_pose = torch.cat([R, T], -1)

                    self.gt_c2w = gt_c2w
                    self.visualizer.save_imgs(idx, cam_iter, gt_depth, gt_color, cam_pose, all_planes, self.decoders)
                    
                    loss = self.optimize_tracking(idx, cam_iter, cam_pose, gt_color, gt_depth, self.tracking_pixels, optimizer_camera, gt_R, gt_T, gt_c2w, current_rays,poses_all)
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_pose = cam_pose.clone().detach()
                """
                # c2w = cam_pose_to_matrix(candidate_cam_pose)
                if self.use_orb_backend:
                    if self.type == "Replica" or self.type == "ScanNet":
                        # print("BE✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖")
                        # new_poses = self.get_new_poses()
                        # new_poses = [torch.from_numpy(pose).clone().detach() for pose in new_poses]
                        # new_poses = [self.flip_axes(pose) for pose in new_poses]
                        # new_poses = [self.flip_rows(pose) for pose in new_poses]  # 翻转行
                        # new_poses = [self.gt_c2w_list[0].to(pose.dtype) @ pose for pose in new_poses]  # 将估计位姿转换为世界坐标系下的位姿
                        new_poses = []
                        for p in self.get_new_poses():
                            pose = torch.from_numpy(p).clone().detach()
                            pose = self.flip_axes(pose)
                            pose = self.flip_rows(pose)
                            pose = self.gt_c2w_list[0].to(pose.dtype) @ pose
                            new_poses.append(pose)

                    else:
                        new_poses = self.get_new_poses()
                        new_poses = [torch.from_numpy(pose).clone().detach() for pose in new_poses]
                        new_poses = [self.flip_axes(pose) for pose in new_poses]
                else:
                    c2w = torch.from_numpy(self.pose_es[-1]).float().clone().detach()  # 获取最后一帧的估计位姿===
                    assert idx != 0

                    c2w = self.flip_axes(c2w)

                    # print("c2w_defore: ", c2w, c2w.shape, c2w.dtype)
                    if self.type == "Replica" or self.type == "ScanNet":
                        # print("✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖")
                        c2w = self.flip_rows(c2w)  # 翻转行
                        c2w = self.gt_c2w_list[0] @ c2w  # 将估计位姿转换为世界坐标系下的位姿
            if self.use_orb_backend:
                if new_poses is None:
                    self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()
                else:
                    for i, pose in enumerate(new_poses[1:], start=1):
                        self.estimate_c2w_list[i] = pose.to(self.device)
            else:
                # print("c2w: ", c2w, c2w.shape, c2w.dtype)
                self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()


            print("estimate: ", self.estimate_c2w_list[idx], self.estimate_c2w_list[idx].shape, self.estimate_c2w_list[idx].dtype)

            self.gt_c2w_list[idx] = gt_c2w.squeeze(0).clone()
            print("gt_c2w:{} ", self.gt_c2w_list[idx], self.gt_c2w_list[idx].shape, self.gt_c2w_list[idx].dtype)
            if self.type == "ScanNet":
                # print("extra opt")
                pose = self.estimate_c2w_list[idx].unsqueeze(0).to(device)
                cam_pose = matrix_to_cam_pose(pose)
                # print("cam_pose: ", cam_pose, idx)
                T = torch.nn.Parameter(cam_pose[:, -3:].clone())
                R = torch.nn.Parameter(cam_pose[:,:4].clone())
                # print("T", T)
                # print("R", R)
                gt_R = gt_c2w[:, :3, :3].clone()
                gt_T = gt_c2w[:, :3, 3].clone()
                #print("gt_R:",gt_R)
                cam_para_list_T = [T]
                cam_para_list_R = [R]
                optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr_T, 'betas':(0.5, 0.999)},
                                                        {'params': cam_para_list_R, 'lr': self.cam_lr_R, 'betas':(0.5, 0.999)}])

                current_min_loss = torch.tensor(float('inf')).float().to(device)
                for cam_iter in range(self.num_cam_iters):
                    cam_pose = torch.cat([R, T], -1)

                    self.gt_c2w = gt_c2w
                    self.visualizer.save_imgs(idx, cam_iter, gt_depth, gt_color, cam_pose, all_planes, self.decoders)
                    
                    # loss = self.optimize_tracking(idx, cam_iter, cam_pose, gt_color, gt_depth, self.tracking_pixels, optimizer_camera, gt_R, gt_T, gt_c2w, current_rays,poses_all)
                    loss = self.optimize_tracking_eslam(cam_pose, gt_color, gt_depth, self.tracking_pixels, optimizer_camera)
                    if loss < current_min_loss:
                        current_min_loss = loss
                        self.estimate_c2w_list[idx] = cam_pose_to_matrix(cam_pose).clone().detach()
                print("estimate_after_opt: ", self.estimate_c2w_list[idx], self.estimate_c2w_list[idx].shape, self.estimate_c2w_list[idx].dtype)

            # # 检查是否仅是轴翻转导致的差异（例如 z轴或 y轴取反）
            # R_est = self.estimate_c2w_list[idx][:3, :3].double()
            # R_gt = gt_c2w.squeeze(0)[:3, :3].double()

            # # 对比旋转矩阵的差异
            # R_diff = R_est @ R_gt.T
            # print("R_diff: ", R_diff)

            # # 相对位姿误差
            # rel_pose = torch.linalg.inv(gt_c2w.squeeze(0)).double()@ self.estimate_c2w_list[idx].double()
            # print("rel_pose", rel_pose)

            # t_est = self.estimate_c2w_list[idx][:3, 3].double()
            # t_gt = gt_c2w.squeeze(0)[:3, 3].double()
            # print("t_est: ", t_est)
            # print("t_gt: ", t_gt)
            # translation_error = torch.norm(t_est - t_gt)

            # R_est = self.estimate_c2w_list[idx][:3, :3].double()
            # R_gt = gt_c2w.squeeze(0)[:3, :3].double()
            # R_rel = R_est @ R_gt.T
            # angle_error = torch.acos(((torch.trace(R_rel) - 1) / 2).clamp(-1, 1)) * 180 / torch.pi

            # print(f"Translation Error: {translation_error:.4f} m")
            # print(f"Rotation Error: {angle_error:.2f} degrees")

            # from scipy.spatial.transform import Rotation as R

            # R_est = R.from_matrix(self.estimate_c2w_list[idx][:3, :3].cpu().numpy())
            # R_gt = R.from_matrix(gt_c2w.squeeze(0)[:3, :3].cpu().numpy())

            # R_diff = R_est.inv() * R_gt
            # angle = R_diff.magnitude() * 180 / np.pi  # degree
            # axis = R_diff.as_rotvec()
            # print("Rotation Error: ", angle, "deg")
            # print("Rotation Axis: ", axis)


            pre_c2w = c2w.clone()
            self.idx[0] = idx

            tracking_oneframe_time = tracking_ending_time - tracking_start_time
            print("Tracking time: {:.3f} seconds".format(tracking_oneframe_time))
            tracking_time_sum += tracking_oneframe_time
            torch.cuda.empty_cache()
        tracking_mean_time = tracking_time_sum / self.n_img
        print("Average tracking time per frame: {:.3f} seconds".format(tracking_mean_time))
        for i in range(torch.cuda.device_count()):
            print(f"CUDA Device {i}:")
            print(f"  Allocated memory: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"  Reserved memory : {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

        self.save_traj(self.save_path)
        if self.use_orb_backend:
            new_poses = self.get_new_poses()
            new_poses = [torch.from_numpy(pose).clone().detach() for pose in self.get_new_poses()]
            new_poses = [self.flip_axes(pose) for pose in new_poses]
            if new_poses is None:
                self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()
            else:
                for i, pose in enumerate(new_poses[1:], start=1):
                    self.estimate_c2w_list[i] = pose.to(self.device)

        self.logger.log(idx, self.shared_keyframe_list)

# This file is a part of ESLAM.
#
# ESLAM is a NeRF-based SLAM system. It utilizes Neural Radiance Fields (NeRF)
# to perform Simultaneous Localization and Mapping (SLAM) in real-time.
# This software is the implementation of the paper "ESLAM: Efficient Dense SLAM
# System Based on Hybrid Representation of Signed Distance Fields" by
# Mohammad Mahdi Johari, Camilla Carta, and Francois Fleuret.
#
# Copyright 2023 ams-OSRAM AG
#
# Author: Mohammad Mahdi Johari <mohammad.johari@idiap.ch>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/Tracker.py
# which is covered by the following copyright and permission notice:
    #
    # Copyright 2022 Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, Marc Pollefeys
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
# PLG:
# from PIL import Image
# import torch
# import copy
# import os
# import time
# import numpy as np
# import random
# from colorama import Fore, Style
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import struct
# import collections
# from types import SimpleNamespace

# from src.common import (matrix_to_cam_pose, cam_pose_to_matrix, get_samples)
# from src.utils.datasets import get_dataset
# from src.utils.Frame_Visualizer import Frame_Visualizer
# from src.keyframe import KeyFrameDatabase
# from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle
# # import pycolmap
# #import src.dataloading.common as common
# #from src.dataloading.dataset import DataField
# from utils.graphics_utils import (
#     focal2fov,
#     fov2focal,
#     getK,
#     getWorld2View,
#     getWorld2View2,
# )
# from scene.dataset_readers import  CameraInfo

# # RTG:
# # import copy
# import matplotlib.pyplot as plt
# from SLAM.gaussian_pointcloud import *

# import torch.multiprocessing as mp
# # from SLAM.render import Renderer
# from collections import defaultdict
# from tqdm import tqdm
# from scipy.spatial.transform import Rotation as R

# from SLAM.icp import IcpTracker
# from threading import Thread
# from utils.camera_utils import loadCam
# # from scene import Dataset

# np.set_printoptions(suppress=True, precision=6)
# torch.set_printoptions(4, sci_mode=False)
# # RTG:
# def convert_poses(trajs):
#     poses = []
#     stamps = []
#     for traj in trajs:
#         stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = traj
#         pose_ = np.eye(4)
#         pose_[:3, :3] = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
#         pose_[:3, 3] = np.array([t0, t1, t2])
#         poses.append(pose_)
#         stamps.append(stamp)
#     return poses, stamps


# class Tracker_rtg(object):
#     def __init__(self, args):

#         if isinstance(args, dict):
#             args = SimpleNamespace(**args)

#         self.use_gt_pose = args.use_gt_pose
#         self.mode = args.mode
#         self.K = None
#         self.type = args.type

#         self.min_depth = args.min_depth
#         self.max_depth = args.max_depth
#         self.depth_filter = args.depth_filter
#         self.verbose = args.verbose

#         self.icp_tracker = IcpTracker(args)

#         self.status = defaultdict(bool)     #默认值False
#         self.pose_gt = []
#         self.pose_es = []
#         self.timestampes = []
#         self.finish = mp.Condition()

#         self.icp_success_count = 0

#         self.use_orb_backend = args.use_orb_backend
#         self.orb_vocab_path = args.orb_vocab_path
#         self.orb_settings_path = args.orb_settings_path
#         self.orb_backend = None
#         self.orb_useicp = args.orb_useicp

#         self.invalid_confidence_thresh = args.invalid_confidence_thresh

#         if self.mode == "single process":
#             self.initialize_orb()

#     # 根据frame_id获取位姿（从估计的位姿中获取指定帧的位姿）
#     def get_new_poses_byid(self, frame_ids):
#         if self.use_orb_backend and not self.use_gt_pose:
#             new_poses = convert_poses(self.orb_backend.get_trajectory_points())
#             frame_poses = [new_poses[frame_id] for frame_id in frame_ids]
#         else:
#             frame_poses = [self.pose_es[frame_id] for frame_id in frame_ids]
#         return frame_poses

#     # 获取最新的位姿
#     def get_new_poses(self):
#         if self.use_orb_backend and not self.use_gt_pose:
#             new_poses, _ = convert_poses(self.orb_backend.get_trajectory_points())
#         else:
#             new_poses = None
#         return new_poses

#     # 保存无效的跟踪数据（如果估计值与真实值误差超过阈值（0.15），就保存相关数据）
#     def save_invalid_traing(self, path):
#         if np.linalg.norm(self.pose_es[-1][:3, 3] - self.pose_gt[-1][:3, 3]) > 0.15:
#             if self.track_mode == "icp":
#                 frame_id = len(self.pose_es)
#                 torch.save(
#                     self.icp_tracker.vertex_pyramid_t1,
#                     os.path.join(path, "vertex_pyramid_t1_{}.pt".format(frame_id)),
#                 )
#                 torch.save(
#                     self.icp_tracker.vertex_pyramid_t0,
#                     os.path.join(path, "vertex_pyramid_t0_{}.pt".format(frame_id)),
#                 )
#                 torch.save(
#                     self.icp_tracker.normal_pyramid_t1,
#                     os.path.join(path, "normal_pyramid_t1_{}.pt".format(frame_id)),
#                 )
#                 torch.save(
#                     self.icp_tracker.normal_pyramid_t0,
#                     os.path.join(path, "normal_pyramid_t0_{}.pt".format(frame_id)),
#                 )

#     # 对输入帧的深度图和法线图进行预处理
#     def map_preprocess(self, frame, frame_id):
#         depth_map, color_map = (
#             frame.original_depth.permute(1, 2, 0) * 255,
#             frame.original_image.permute(1, 2, 0),
#         )  # [H, W, C], the image is scaled by 255 in function "PILtoTorch"
#         depth_map_orb = (
#             frame.original_depth.permute(1, 2, 0).cpu().numpy()
#             * 255
#             * frame.depth_scale
#         ).astype(np.uint16)
#         intrinsic = frame.get_intrinsic
#         # depth filter
#         if self.depth_filter:
#             depth_map_filter = bilateralFilter_torch(depth_map, 5, 2, 2)
#         else:
#             depth_map_filter = depth_map

#         valid_range_mask = (depth_map_filter > self.min_depth) & (depth_map_filter < self.max_depth)
#         depth_map_filter[~valid_range_mask] = 0.0
#         # update depth map
#         frame.original_depth = depth_map_filter.permute(2, 0, 1) / 255.0
#         # compute geometry info
#         vertex_map_c = compute_vertex_map(depth_map_filter, intrinsic)
#         normal_map_c = compute_normal_map(vertex_map_c)
#         confidence_map = compute_confidence_map(normal_map_c, intrinsic)

#         # confidence_threshold tum: 0.5, others: 0.2
#         invalid_confidence_mask = ((normal_map_c == 0).all(dim=-1)) | (
#             confidence_map < self.invalid_confidence_thresh
#         )[..., 0]

#         depth_map_filter[invalid_confidence_mask] = 0
#         normal_map_c[invalid_confidence_mask] = 0
#         vertex_map_c[invalid_confidence_mask] = 0
#         confidence_map[invalid_confidence_mask] = 0

#         color_map_orb = (
#             (frame.original_image * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
#         )

#         self.update_curr_status(
#             frame,
#             frame_id,
#             depth_map,
#             depth_map_filter,
#             vertex_map_c,
#             normal_map_c,
#             color_map,
#             color_map_orb,
#             depth_map_orb,
#             intrinsic,
#         )

#         frame_map = {}
#         frame_map["depth_map"] = depth_map_filter
#         frame_map["color_map"] = color_map
#         frame_map["normal_map_c"] = normal_map_c
#         frame_map["vertex_map_c"] = vertex_map_c
#         frame_map["confidence_map"] = confidence_map
#         frame_map["invalid_confidence_mask"] = invalid_confidence_mask
#         frame_map["time"] = frame_id

#         return frame_map

#     # 将当前帧的状态（深度图、法线图、顶点图）存储到curr_frame中
#     # 该函数在每次跟踪时调用
#     def update_curr_status(
#         self,
#         frame,
#         frame_id,
#         depth_t1,
#         depth_t1_filter,
#         vertex_t1,
#         normal_t1,
#         color_t1,
#         color_orb,
#         depth_orb,
#         K,
#     ):
#         if self.K is None:
#             self.K = K
#         self.curr_frame = {
#             "K": frame.get_intrinsic,
#             "normal_map": normal_t1,
#             "depth_map": depth_t1,
#             "depth_map_filter": depth_t1_filter,
#             "vertex_map": vertex_t1,
#             "frame_id": frame_id,
#             "pose_gt": frame.get_c2w.cpu().numpy(), # 1
#             "color_map": color_t1,
#             "timestamp": frame.timestamp, # 1
#             "color_map_orb": color_orb, # 1
#             "depth_map_orb": depth_orb, # 1
#         }
#         self.icp_tracker.update_curr_status(depth_t1_filter, self.K)
        
#     # 更新上一帧的状态
#     def update_last_status_v2(
#         self, frame, render_depth, frame_depth, render_normal, frame_normal
#     ):
#         intrinsic = frame.get_intrinsic
#         normal_mask = (
#             1 - F.cosine_similarity(render_normal, frame_normal, dim=-1)
#         ) < self.icp_sample_normal_threshold
#         depth_filling_mask = (
#             (
#                 torch.abs(render_depth - frame_depth)
#                 > self.icp_sample_distance_threshold
#             )[..., 0]
#             | (render_depth == 0)[..., 0]
#             | (normal_mask)
#         ) & (frame_depth > 0)[..., 0]

#         render_depth[depth_filling_mask] = frame_depth[depth_filling_mask]
#         render_depth[(frame_depth == 0)[..., 0]] = 0
        
#         self.last_model_vertex = compute_vertex_map(render_depth, intrinsic)
#         self.last_model_normal = compute_normal_map(self.last_model_vertex)

#     def update_last_status(
#         self,
#         frame,
#         render_depth,
#         frame_depth,
#     ):
#         self.icp_tracker.update_last_status_qm(
#             frame, render_depth, frame_depth
#         )

#     # 通过ICP或ORB后端优化位姿
#     # 如果 ICP 失败，使用 ORB 特征匹配优化位姿
#     def refine_icp_pose(self, pose_t1_t0, tracking_success):
#         if tracking_success and self.orb_useicp:
#             print("success")
#             self.orb_backend.track_with_icp_pose(
#                 self.curr_frame["color_map_orb"],
#                 self.curr_frame["depth_map_orb"],
#                 pose_t1_t0.astype(np.float32),
#                 self.curr_frame["timestamp"],
#             )
#             time.sleep(0.005)
#         else:
#             self.orb_backend.track_with_orb_feature(
#                 self.curr_frame["color_map_orb"],
#                 self.curr_frame["depth_map_orb"],
#                 self.curr_frame["timestamp"],
#             )
#             time.sleep(0.005)
#         traj_history = self.orb_backend.get_trajectory_points()
#         pose_es_t1, _ = convert_poses(traj_history[-2:])
#         return pose_es_t1[-1]

#     # 初始化ORB后端
#     def initialize_orb(self):
#         if not self.use_gt_pose and self.use_orb_backend and self.orb_backend is None:
#             import orbslam2
#             print("init orb backend")
#             self.orb_backend = orbslam2.System(
#                 self.orb_vocab_path, self.orb_settings_path, orbslam2.Sensor.RGBD
#             )
#             self.orb_backend.set_use_viewer(False)
#             self.orb_backend.initialize(self.orb_useicp)

#     # 初始化跟踪器
#     def initialize_tracker(self):
#         if self.use_orb_backend:
#             self.orb_backend.process_image_rgbd(
#                 self.curr_frame["color_map_orb"],
#                 self.curr_frame["depth_map_orb"],
#                 self.curr_frame["timestamp"],
#             )
#         self.status["initialized"] = True

#     # 执行跟踪任务
#     def tracking(self, frame, frame_map):
#         self.pose_gt.append(self.curr_frame["pose_gt"])
#         self.timestampes.append(self.curr_frame["timestamp"])
#         p2loss = 0
#         tracking_success = True
#         if self.use_gt_pose:
#             pose_t1_w = self.pose_gt[-1]
#         else:
#             # initialize
#             if not self.status["initialized"]:
#                 self.initialize_tracker()
#                 pose_t1_w = np.eye(4)
#             else:
#                 pose_t1_t0, tracking_success = self.icp_tracker.predict_pose(self.curr_frame)
#                 if self.use_orb_backend:
#                     pose_t1_w = self.refine_icp_pose(pose_t1_t0, tracking_success)
#                 else:
#                     pose_t1_w = self.pose_es[-1] @ pose_t1_t0

#         self.icp_tracker.move_last_status()
#         self.pose_es.append(pose_t1_w)

#         frame.updatePose(pose_t1_w)
#         frame_map["vertex_map_w"] = transform_map(
#             frame_map["vertex_map_c"], frame.get_c2w
#         )
#         frame_map["normal_map_w"] = transform_map(
#             frame_map["normal_map_c"], get_rot(frame.get_c2w)
#         )

#         return tracking_success

#     # 评估轨迹的绝对轨迹误差（ATE）
#     def eval_total_ate(self, pose_es, pose_gt):
#         ates = []
#         for i in tqdm(range(1, len(pose_gt) + 1)):
#             ates.append(self.eval_ate(pose_es, pose_gt, i))
#         ates = np.array(ates)
#         return ates

#     # 保存ATE曲线图用于可视化
#     def save_ate_fig(self, ates, save_path, save_name):
#         plt.plot(range(len(ates)), ates)
#         plt.ylim(0, max(ates) + 0.1)
#         plt.title("ate:{}".format(ates[-1]))
#         plt.savefig(os.path.join(save_path, "{}.png".format(save_name)))
    

#     # 保存关键帧轨迹
#     def save_keyframe_traj(self, save_file):
#         if self.use_orb_backend:
#             poses, stamps = convert_poses(self.orb_backend.get_keyframe_points())
#             with open(save_file, "w") as f:
#                 for pose_id, pose_es_ in enumerate(poses):
#                     t = pose_es_[:3, 3]
#                     q = R.from_matrix(pose_es_[:3, :3])
#                     f.write(str(stamps[pose_id]) + " ")
#                     for i in t.tolist():
#                         f.write(str(i) + " ")
#                     for i in q.as_quat().tolist():
#                         f.write(str(i) + " ")
#                     f.write("\n")

#     # 保存轨迹为TUM格式
#     def save_traj_tum(self, save_file):
#         poses, stamps = convert_poses(self.orb_backend.get_trajectory_points())
#         with open(save_file, "w") as f:
#             for pose_id, pose_es_ in enumerate(self.pose_es):
#                 t = pose_es_[:3, 3]
#                 q = R.from_matrix(pose_es_[:3, :3])
#                 f.write(str(stamps[pose_id]) + " ")
#                 for i in t.tolist():
#                     f.write(str(i) + " ")
#                 for i in q.as_quat().tolist():
#                     f.write(str(i) + " ")
#                 f.write("\n")

#     # 保存ORB后端轨迹为TUM格式
#     def save_orb_traj_tum(self, save_file):
#         if self.use_orb_backend:
#             poses, stamps = convert_poses(self.orb_backend.get_trajectory_points())
#             with open(save_file, "w") as f:
#                 for pose_id, pose_es_ in enumerate(poses):
#                     t = pose_es_[:3, 3]
#                     q = R.from_matrix(pose_es_[:3, :3])
#                     f.write(str(stamps[pose_id]) + " ")
#                     for i in t.tolist():
#                         f.write(str(i) + " ")
#                     for i in q.as_quat().tolist():
#                         f.write(str(i) + " ")
#                     f.write("\n")

#     # 保存轨迹（真实和估计）和误差评估结果
#     def save_traj(self, save_path):
#         save_traj_path = os.path.join(save_path, "save_traj")
#         if not self.use_gt_pose and self.use_orb_backend:
#             traj_history = self.orb_backend.get_trajectory_points()
#             self.pose_es, _ = convert_poses(traj_history)
#         pose_es = np.stack(self.pose_es, axis=0)
#         pose_gt = np.stack(self.pose_gt, axis=0)
#         ates_ba = self.eval_total_ate(pose_es, pose_gt)
#         print("ate: ", ates_ba[-1])
#         np.save(os.path.join(save_traj_path, "pose_gt.npy"), pose_gt)
#         np.save(os.path.join(save_traj_path, "pose_es.npy"), pose_es)
#         self.save_ate_fig(ates_ba, save_traj_path, "ate")

#         plt.figure()
#         plt.plot(pose_es[:, 0, 3], pose_es[:, 1, 3])
#         plt.plot(pose_gt[:, 0, 3], pose_gt[:, 1, 3])
#         plt.legend(["es", "gt"])
#         plt.savefig(os.path.join(save_traj_path, "traj_xy.jpg"))
        
#         if self.use_orb_backend:
#             self.orb_backend.shutdown()
        
#     # 评估单帧的ATE
#     def eval_ate(self, pose_es, pose_gt, frame_id=-1):
#         pose_es = np.stack(pose_es, axis=0)[:frame_id, :3, 3]
#         pose_gt = np.stack(pose_gt, axis=0)[:frame_id, :3, 3]
#         ate = eval_ate(pose_gt, pose_es)
#         return ate
















# # PLG:
# Point3D = collections.namedtuple(
#     "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

# BaseImage = collections.namedtuple(
#     "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

# class Tracker(Tracker_rtg):
#     """
#     Tracking main class.
#     Args:
#         cfg (dict): config dict
#         args (argparse.Namespace): arguments
#         eslam (ESLAM): ESLAM object
#     """
#     def __init__(self, cfg, args, eslam):

#         super().__init__(cfg)

#         self.save_path = cfg['save_path']
#         self.frame_id = 0

#         self.cfg = cfg
#         self.args = args
#         self.logger = eslam.logger
#         # RTG：read dataset information
#         # self.dataset_cameras = eslam.dataset_rtg.scene_info.train_cameras
#         self.dataset_cameras = None #############################
#         self.scale = cfg['scale']
#         self.gt_campose_list = []
#         self.idx = eslam.idx
#         self.bound = eslam.bound
#         self.mesher = eslam.mesher
#         self.output = eslam.output
#         self.verbose = eslam.verbose
#         self.renderer = eslam.renderer
#         self.gt_c2w_list = eslam.gt_c2w_list
#         self.mapping_idx = eslam.mapping_idx
#         self.mapping_cnt = eslam.mapping_cnt
#         self.estimate_c2w_list = eslam.estimate_c2w_list
#         self.truncation = eslam.truncation
#         self.tracker_keyframe_dict = []
#         self.shared_keyframe_list_re = []
#         self.shared_keyframe_list = []


#         self.cam_lr_T = cfg['tracking']['lr_T']
#         self.cam_lr_R = cfg['tracking']['lr_R']
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print("device :", self.device)
#         self.num_cam_iters = cfg['tracking']['iters']
#         self.gt_camera = cfg['tracking']['gt_camera']
#         self.tracking_pixels = cfg['tracking']['pixels']
#         self.w_sdf_fs = cfg['tracking']['w_sdf_fs']
#         self.w_sdf_center = cfg['tracking']['w_sdf_center']
#         self.w_sdf_tail = cfg['tracking']['w_sdf_tail']
#         self.w_depth = cfg['tracking']['w_depth']
#         self.w_color = cfg['tracking']['w_color']
#         self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
#         self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
#         self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

#         self.every_frame = cfg['mapping']['every_frame']
#         self.no_vis_on_first_frame = cfg['tracking']['no_vis_on_first_frame']

#         self.prev_mapping_idx = -1
#         self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
#         self.n_img = len(self.frame_reader)
#         self.frame_loader = DataLoader(self.frame_reader, batch_size=1, shuffle=False,
#                                        num_workers=1, pin_memory=True, prefetch_factor=2)

#         self.visualizer = Frame_Visualizer(freq=cfg['tracking']['vis_freq'], n_imgs=self.n_img, inside_freq=cfg['tracking']['vis_inside_freq'],
#                                            vis_dir=os.path.join(self.output, 'tracking_vis'), renderer=self.renderer,
#                                            truncation=self.truncation, verbose=self.verbose, device=self.device)

#         self.H, self.W, self.fx, self.fy, self.cx, self.cy = eslam.H, eslam.W, eslam.fx, eslam.fy, eslam.cx, eslam.cy

#         #self.num_frames = len(frame_ids)
#         self.keyframe_device = cfg['keyframe_device']

#         self.embedpos_fn = eslam.shared_embedpos_fn
#         #self.embedpos_fn = eslam.embedpos_fn
#         self.shared_all_planes_list = eslam.shared_all_planes_list
#         self.shared_decoders_list = eslam.shared_decoders_list
#         self.shared_cur_rf_id = eslam.shared_cur_rf_id
#         self.pre_shared_cur_rf_id = self.shared_cur_rf_id[0].clone()
#         self.update_para_flag = 1
#         self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz = copy.deepcopy(self.shared_all_planes_list[self.shared_cur_rf_id[0]])

#         for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
#             for i, plane in enumerate(planes):
#                 plane = plane.to(self.device)
#                 #plane.share_memory_()
#                 planes[i] = plane

#         for c_planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]:
#             for i, plane in enumerate(c_planes):
#                 plane = plane.to(self.device)
#                 #plane.share_memory_()
#                 c_planes[i] = plane
#         #self.decoders_list = copy.deepcopy(self.shared_decoders_list)

#         self.shared_decoders = eslam.shared_decoders_list[self.shared_cur_rf_id[0]]
#         self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)   #.to(self.device)
#      #  # ##########################
#         self.Kf_depth = None
#         self.Kf_color = None
#         self.Kf_sdf = None
#         self.Kf_z_vals = None
#         self.Kf_gt_depth = None

#         self.Kf_pre_depth = None
#         self.Kf_pre_color = None
#         self.kf_pre_sdf = None
#         self.kf_pre_z_vals = None
#         self.Kf_pre_gt_depth = None
#         self.eps = 1e-3
#         # p3D = self.load_data()
#         # self.points3D = p3D
#         #self.truncation = cfg['model']['truncation']
#         #self.p3d = DataField.p3d
#         self.crop_size = cfg['cam']['crop_edge'] if 'crop_edge' in cfg['cam'] else 0
#         self.total_pixels = (self.H - self.crop_size*2) * (self.W - self.crop_size*2)
#         # self.num_rays_to_save = int(self.total_pixels * cfg['mapping']['n_pixels'])
#         self.num_rays_to_save = cfg['tracking']['pixels']
#         self.keyframeDatabase = self.create_kf_database(cfg)
#         #print(self.keyframeDatabase)

#         for p in self.decoders.parameters():
#             p.requires_grad_(False)

#     #2月29日 创建keyframe database        
#     def create_kf_database(self, config):  
#         '''
#         Create the keyframe database
#         '''
#         num_kf = int(self.n_img // self.cfg['mapping']['gb_keyframe'] + 1)  
#         print('#kf:', num_kf)
#         print('#Pixels to save:', self.num_rays_to_save)
#         return KeyFrameDatabase(config, 
#                                 self.H, 
#                                 self.W, 
#                                 num_kf, 
#                                 self.num_rays_to_save, 
#                                 self.device)



#     # 加载3D点和图像数据，支持从二进制文件中读取点云和图像信息。它还处理点的可见性和关联性。
#     # def load_data(factor=None, load_colmap_poses=True):
#     #     if load_colmap_poses:
#     #         ##
#     #         class Image(BaseImage):
#     #             def qvec2rotmat(self):
#     #                 return qvec2rotmat(self.qvec)

#     #         def qvec2rotmat(qvec):
#     #             return np.array([
#     #                 [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
#     #                 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
#     #                 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
#     #                 [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
#     #                 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
#     #                 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
#     #                 [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
#     #                 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
#     #                 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])
            
#     #         def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
#     #             """Read and unpack the next bytes from a binary file.
#     #             :param fid:
#     #             :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
#     #             :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
#     #             :param endian_character: Any of {@, =, <, >, !}
#     #             :return: Tuple of read and unpacked values.
#     #             """
#     #             data = fid.read(num_bytes)
#     #             return struct.unpack(endian_character + format_char_sequence, data)

#     #         def read_points3D_binary(path_to_model_file):##读取bin文件函数
#     #             """
#     #             see: src/base/reconstruction.cc
#     #                 void Reconstruction::ReadPoints3DBinary(const std::string& path)
#     #                 void Reconstruction::WritePoints3DBinary(const std::string& path)
#     #             """
#     #             points3D = {}
#     #             with open(path_to_model_file, "rb") as fid:
#     #                 num_points = read_next_bytes(fid, 8, "Q")[0]
#     #                 for _ in range(num_points):
#     #                     binary_point_line_properties = read_next_bytes(
#     #                         fid, num_bytes=43, format_char_sequence="QdddBBBd")
#     #                     point3D_id = binary_point_line_properties[0]
#     #                     xyz = np.array(binary_point_line_properties[1:4])
#     #                     rgb = np.array(binary_point_line_properties[4:7])
#     #                     error = np.array(binary_point_line_properties[7])
#     #                     track_length = read_next_bytes(
#     #                         fid, num_bytes=8, format_char_sequence="Q")[0]
#     #                     track_elems = read_next_bytes(
#     #                         fid, num_bytes=8 * track_length,
#     #                         format_char_sequence="ii" * track_length)
#     #                     image_ids = np.array(tuple(map(int, track_elems[0::2])))
#     #                     point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
#     #                     points3D[point3D_id] = Point3D(
#     #                         id=point3D_id, xyz=xyz, rgb=rgb,
#     #                         error=error, image_ids=image_ids,
#     #                         point2D_idxs=point2D_idxs)
#     #             return points3D

#     #         def read_images_binary(path_to_model_file):##读取bin文件函数
#     #             """
#     #             see: src/base/reconstruction.cc
#     #                 void Reconstruction::ReadImagesBinary(const std::string& path)
#     #                 void Reconstruction::WriteImagesBinary(const std::string& path)
#     #             """
#     #             images = {}
#     #             with open(path_to_model_file, "rb") as fid:
#     #                 num_reg_images = read_next_bytes(fid, 8, "Q")[0]
#     #                 for _ in range(num_reg_images):
#     #                     binary_image_properties = read_next_bytes(
#     #                         fid, num_bytes=64, format_char_sequence="idddddddi")
#     #                     image_id = binary_image_properties[0]
#     #                     qvec = np.array(binary_image_properties[1:5])
#     #                     tvec = np.array(binary_image_properties[5:8])
#     #                     camera_id = binary_image_properties[8]
#     #                     image_name = ""
#     #                     current_char = read_next_bytes(fid, 1, "c")[0]
#     #                     while current_char != b"\x00":  # look for the ASCII 0 entry
#     #                         image_name += current_char.decode("utf-8")
#     #                         current_char = read_next_bytes(fid, 1, "c")[0]
#     #                     num_points2D = read_next_bytes(fid, num_bytes=8,
#     #                                                 format_char_sequence="Q")[0]
#     #                     x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
#     #                                             format_char_sequence="ddq" * num_points2D)
#     #                     xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
#     #                                         tuple(map(float, x_y_id_s[1::3]))])
#     #                     point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
#     #                     images[image_id] = Image(
#     #                         id=image_id, qvec=qvec, tvec=tvec,
#     #                         camera_id=camera_id, name=image_name,
#     #                         xys=xys, point3D_ids=point3D_ids)
#     #             return images

#     #         # Read 3D points 
#     #         ##bin文件路径   /data0/wjy/ESLAM/Datasets/Apartment/colmap1
#     #         image_dir = "./Datasets/Apartment/color"
#     #         database_path = "./output/database.db"
#     #         output_dir = "/data0/wjy/eslamnew/Datasets/Apartment/color/colmap"
#     #         #reconstruction = self.run_pycolmap(image_dir, database_path, output_dir)
#     #         points3D = read_points3D_binary('/home/sjtu/ESLAM/Datasets/Replica/room0/sparse/points3D.bin')  #scannet
#     #         #points3D里包括点的id，xyz坐标，rgb，和对应存在的image的id
#     #         images = read_images_binary('/home/sjtu/ESLAM/Datasets/Replica/room0/sparse/images.bin')
#     #         #points3D = read_points3D_binary('/data0/wjy/ESLAM/Datasets/scannet/scene0000_00/colmap/sparse/0/points3D.bin')  #scannet
#     #         #points3D里包括点的id，xyz坐标，rgb，和对应存在的image的id
#     #         #images = read_images_binary('/data0/wjy/ESLAM/Datasets/scannet/scene0000_00/colmap/sparse/0/images.bin')
 
#     #         #points3D = read_points3D_binary('/data0/wjy/ESLAM/Datasets/Replica/office0/colmap/sparse/0/points3D.bin') #replica
#     #         #points3D里包括点的id，xyz坐标，rgb，和对应存在的image的id
#     #         #images = read_images_binary('/data0/wjy/ESLAM/Datasets/Replica/office0/colmap/sparse/0/images.bin')
#     #         #images里有一项也是对应是否出现过上面3D点的list，出现为点的id，不出现为-1
#     #     #/data0/wjy/ESLAM/Datasets/Replica/room0/sparse/0
#     #     p3D_ids = sorted(points3D.keys())
#     #     p3D_id_to_idx = dict(zip(p3D_ids, range(len(points3D))))
#     #     p3D_xyz = np.stack([points3D[i].xyz for i in p3D_ids])
#     #     track_lengths = np.stack([len(points3D[i].image_ids) for i in p3D_ids])
#     #     p3D_observed = []
#     #     print(len(images))
#     #     for i in range(1,len(images)+1):
#     #         try:
#     #             image = images[i]
#     #             obs = np.stack([p3D_id_to_idx[i]] for i in image.point3D_ids if i != -1)
#     #             p3D_observed.append(obs)
#     #             p3D = {'points3D' : p3D_xyz, 'p3D_observed' : p3D_observed}
#     #         except KeyError:
#     #             pass  # 当发生 KeyError 时，什么也不做，直接跳过
#     #         #image = images[i]
#     #         #obs = np.stack([p3D_id_to_idx[i]] for i in image.point3D_ids if i != -1)
#     #         #p3D_observed.append(obs)
#     #     #p3D = {'points3D' : p3D_xyz, 'p3D_observed' : p3D_observed}
#     #     #P3D_xyz是所有3D点的坐标， p3D_observed是每张图片对应观察到的点的id

#     #     return p3D
    
#     # def run_pycolmap(image_dir, database_path, output_dir):
#     #     # 特征提取
#     #     pycolmap.extract_features(database_path=database_path, image_path=image_dir)
#     #     print("Features extracted.")

#     #     # 特征匹配
#     #     pycolmap.match_features(database_path=database_path)
#     #     print("Features matched.")

#     #     # 稀疏重建
#     #     sparse_dir = f"{output_dir}/sparse"
#     #     reconstruction = pycolmap.ReconstructionMapper(
#     #         database_path=database_path,
#     #         image_path=image_dir,
#     #         output_path=sparse_dir
#     #     )
#     #     print("Sparse reconstruction completed.")
#     #     return reconstruction
    
#     def c2w_to_w2c(self,c2w):
#         w2c = np.linalg.inv(c2w.cpu())
#         #w2c = np.linalg.inv(c2w)
#         return w2c
#     '''
#     def load_ref_img(self, idx, data={}):
#         if self.random_ref:
#             if idx==0:
#                 ref_idx = self.N_imgs-1
#             else:
#                 #ran_idx = random.randint(1, min(self.random_ref, self.N_imgs-idx-1))
#                 ran_idx = 1
#                 ref_idx = idx - ran_idx
#         image = self.imgs[ref_idx]
#         if self.dpt_depth is not None:
#             dpt = self.dpt_depth[ref_idx]
#             data['ref_dpts'] = dpt
#         if self.use_DPT:
#             data_in = {"image": np.transpose(image, (1, 2, 0))}
#             data_in = self.transform(data_in)
#             normalised_ref_img = data_in['image']
#             data['normalised_ref_img'] = normalised_ref_img
#         if self.with_depth:
#             depth = self.depth[ref_idx]
#             data['ref_depths'] = depth
#         data['ref_imgs'] = image
#         data['ref_idxs'] = ref_idx
#         ##

#         data['T_ref'] = torch.from_numpy(self.c2w_to_w2c(self.c2ws[ref_idx]))
#     '''
#     '''
#     def load_T_r2q(self, data):
#         #print("data shape:",data['T_ref'].shape)
#         #data T ref 是 reference的绝对位姿
#         c2w_ref = torch.from_numpy(self.c2w_to_w2c(data['T_ref']))
#         #T_r2q_gt = data['T_tgt'] @ c2w_ref
#         gt_c2w = self.gt_c2w.to('cpu')
#         T_r2q_gt = torch.matmul(gt_c2w, c2w_ref)
#         #T_r2q_gt = mat_to_Pose(T_r2q_gt)
#         data['T_r2q_gt'] = T_r2q_gt
#         #data['T_r2q_init'].append(optimizer)
#         return data['T_r2q_gt']
#     '''
#     def load_p3d(self, data={}):
#         def T_mul_p3d(T, p3d):
#             T = T.float()
#             p3d = torch.tensor(p3d).float()
#             points3D = torch.cat((p3d, torch.ones(p3d.shape[0], 1)), dim=1)
#             mul = torch.matmul(T, points3D.t())
#             #print("Shape of mul:", mul.shape)
#             mul = mul.squeeze(dim=0)
#             p3D = mul.t()[:, :3]
#             return p3D
        
#         fx = self.fx
#         fy = self.fy
#         cx = self.cx
#         cy = self.cy
#         fx = torch.tensor(fx).unsqueeze(0)  # 将焦距信息扩展为2维张量
#         fy = torch.tensor(fy).unsqueeze(0)
#         cx = torch.tensor(cx).unsqueeze(0)
#         cy = torch.tensor(cy).unsqueeze(0)
#         f = torch.stack([fx, fy], dim=-1)  # 将焦距信息合并成一个张量
#         c = torch.stack([cx, cy], dim=-1)  # 将主点信息合并成一个张量

#         def get_valid(p3d):
#             fx = self.fx
#             fy = self.fy
#             cx = self.cx
#             cy = self.cy
#             fx = torch.tensor(fx)  # 将焦距信息扩展为2维张量
#             fy = torch.tensor(fy)
#             cx = torch.tensor(cx)
#             cy = torch.tensor(cy)
#             c = torch.stack([cx, cy], dim=-1)
#             f = torch.stack([fx, fy], dim=-1)
#             eps = 0.001
#             #c = torch.tensor([[318.905426,242.683609]])
#             #f = 600.0
#             size = torch.tensor([[480,640]])
#             #project
#             z = p3d[..., -1]
#             visible = z > eps
#             z = z.clamp(min=eps)
#             p2d = p3d[..., :-1] / z.unsqueeze(-1)
#             ##denormalize
#             p2d = p2d * f + c
#             ##in_image
#             in_image = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)

#             valid = visible & in_image
#             return valid

#         select_idx = self.selected_keyframe_idx
#         #print("select_idx:",select_idx)
#         if select_idx == 0 :
#             self.random_index = 0
#         else :
#             self.random_index = torch.randint(0, select_idx, (1,))#随机选取reference的图片
#         #print("random_index:",self.random_index)
#         #random_index = random_index.item() / 8
#         #print("idx:",random_index)
#         data['T_ref'] = torch.from_numpy(self.c2w_to_w2c(self.gt_campose_list[self.random_index]))#将reference的Tgt 转置
#         #print("data shape:",data['T_ref'].shape)
#         p3D = self.points3D['points3D']
#         #print("self.points3D:",self.points3D['p3D_observed'][2])
#         obs = self.points3D['p3D_observed'][self.random_index]#选取ref图像对应的观察到的点id
#         #print("obs:", obs)
#         valid = get_valid(T_mul_p3d(data['T_ref'], p3D[obs].squeeze()))#判断点是否在图像上并且可视
#         #print("valid:", valid)
#         obs = obs[valid.numpy()].squeeze()
#         if obs.shape:
#             length = obs.shape[0]
#         else:
#             length = 1
#         #print("obs_valid:", obs)
#         #print(type(obs))
#         #print(length)
#         max_num_points3D = 512#设定每张图片最多观察到的点数
#         if length > 1:
#             obs = np.random.choice(obs, max_num_points3D)#如果大于观察到的点数则随机选取
#             data['points3D'] = T_mul_p3d(data['T_ref'], p3D[obs])#得到对应点的 to w 坐标
#             #print("data['points3D']:",data['points3D'])
#             return data
#         else:
#             return data

    
#     def sdf_losses(self, sdf, z_vals, gt_depth):
#         """
#         Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

#         Args:
#         - self: instance of the class containing this method
#         - sdf: a tensor of shape (R, N) representing the SDF values
#         - z_vals: a tensor of shape (R, N) representing the depth values
#         - gt_depth: a tensor of shape (R,) containing the ground truth depth values

#         Returns:
#         - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
#         """

#         front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
#                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

#         back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
#                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

#         center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
#                                   (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
#                                   torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

#         tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

#         fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
#         center_loss = torch.mean(torch.square(
#             (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
#         tail_loss = torch.mean(torch.square(
#             (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

#         sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

#         return sdf_losses

#     def fs_losses(self, sdf, z_vals, gt_depth):
#         """
#         Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

#         Args:
#         - self: instance of the class containing this method
#         - sdf: a tensor of shape (R, N) representing the SDF values
#         - z_vals: a tensor of shape (R, N) representing the depth values
#         - gt_depth: a tensor of shape (R,) containing the ground truth depth values

#         Returns:
#         - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
#         """

#         front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
#                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

#         back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
#                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

#         center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
#                                   (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
#                                   torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

#         tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

#         fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
#         center_loss = torch.mean(torch.square(
#             (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
#         tail_loss = torch.mean(torch.square(
#             (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

#         #sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

#         return self.w_sdf_fs * fs_loss

#     def project(self, p3d):
#         '''Project 3D points into the camera plane and check for visibility.将3D点投影到相机平面，并检查其是否可见'''
#         z = p3d[..., -1]
#         #print("pro p3d----------------------", p3d.shape)
#         valid = z > self.eps
#         z = z.clamp(min=self.eps)
#         p2d = p3d[..., :2] / z.unsqueeze(-1)
#         #print("pro p2d----------------------", p2d)
#         return p2d, valid
    

#     def undistort_points(self, pts, dist):
#         '''
#         去畸变
#         Undistort normalized 2D coordinates
#         and check for validity of the distortion model.
#         '''
#         dist = torch.tensor(dist).unsqueeze(-2)
#         #dist = dist.unsqueeze(-2)  # add point dimension
#         ndist = dist.shape[-1]
#         undist = pts
#         valid = torch.ones(pts.shape[:-1], device=pts.device, dtype=torch.bool)
#         if ndist > 0:
#             k1, k2 = dist[..., :2].split(1, -1)
#             r2 = torch.sum(pts**2, -1, keepdim=True)
#             radial = k1*r2 + k2*r2**2
#             undist = undist + pts * radial

#             # The distortion model is supposedly only valid within the image
#             # boundaries. Because of the negative radial distortion, points that
#             # are far outside of the boundaries might actually be mapped back
#             # within the image. To account for this, we discard points that are
#             # beyond the inflection point of the distortion model,
#             # e.g. such that d(r + k_1 r^3 + k2 r^5)/dr = 0
#             limited = ((k2 > 0) & ((9*k1**2-20*k2) > 0)) | ((k2 <= 0) & (k1 > 0))
#             limit = torch.abs(torch.where(
#                 k2 > 0, (torch.sqrt(9*k1**2-20*k2)-3*k1)/(10*k2), 1/(3*k1)))
#             valid = valid & torch.squeeze(~limited | (r2 < limit), -1)

#             if ndist > 2:
#                 p12 = dist[..., 2:]
#                 p21 = p12.flip(-1)
#                 uv = torch.prod(pts, -1, keepdim=True)
#                 undist = undist + 2*p12*uv + p21*(r2 + 2*pts**2)
#                 # TODO: handle tangential boundaries

#         return undist, valid
    
#     def dist(self) -> torch.Tensor:
#         self._data = tuple(self._data)
#         '''Distortion parameters, with shape (..., {0, 2, 4}).返回畸变参数'''
#         return self._data[6:]
    
#     def undistort(self, pts):
#         '''Undistort normalized 2D coordinates
#            and check for validity of the distortion model.
#         '''
#         pts = pts.squeeze(dim=0)
#         #print("Shape of pts:", pts.shape)
#         assert pts.shape[-1] == 2
#         #assert pts.shape[:-2] == self.shape  # allow broadcasting
#         return self.undistort_points(pts, self.dist())
    
    
#     def denormalize(self, p2d: torch.Tensor) -> torch.Tensor:
#         '''Convert normalized 2D coordinates into pixel coordinates.将归一化的2D坐标转换为像素坐标。'''
#         # Extract focal lengths (fx, fy) and principal points (cx, cy)
#         #f = torch.tensor(self._data)[..., 2:4]
#         #c = torch.tensor(self._data)[..., 4:6]
#         #f = self._data[..., 2:4]
#         #c = self._data[..., 4:6]
#         # 获取焦距和主点信息
#         fx = self.fx
#         fy = self.fy
#         cx = self.cx
#         cy = self.cy
#         fx = torch.tensor(fx).unsqueeze(0)  # 将焦距信息扩展为2维张量
#         fy = torch.tensor(fy).unsqueeze(0)
#         cx = torch.tensor(cx).unsqueeze(0)
#         cy = torch.tensor(cy).unsqueeze(0)
#         f = torch.stack([fx, fy], dim=-1)  # 将焦距信息合并成一个张量
#         c = torch.stack([cx, cy], dim=-1)  # 将主点信息合并成一个张量
#         #p2d = p2d.view(6, 2)
#         #print("f: ", f)
#         #print("c: ", c)
#         #print("p2d: ", p2d)
#         ###################p2d * self.f.unsqueeze(-2) + self.c.unsqueeze(-2)
#         return p2d * f+ c
    
#     def in_image(self, p2d: torch.Tensor):
#         '''Check if 2D points are within the image boundaries.检查2D点是否在图像边界内。'''
#         assert p2d.shape[-1] == 2
#         # assert p2d.shape[:-2] == self.shape  # allow broadcasting
#         W = torch.tensor([self.W])
#         H = torch.tensor([self.H])
#         #print("W: ", W)
#         #print("H: ", H)

#         # 使用 torch.cat 进行连接
#         size = torch.cat((W.unsqueeze(0), H.unsqueeze(0)), dim=1)
#         #size = self.size.unsqueeze(-2)

#         valid = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
#         #print("valid: ", valid)
#         #valid = valid[:4]
#         return valid

#     def world2image(self, p3d):
#         '''Transform 3D points into 2D pixel coordinates.将3D点转换为2D像素坐标，并检查其有效性。'''
#         p2d, visible = self.project(p3d)
#         p2d, mask = self.undistort(p2d)
#         p2d = self.denormalize(p2d)
#         #print("visible: ", visible)
#         #print("mask: ", mask)
#         valid = visible & mask & self.in_image(p2d)
#         return p2d, valid
    

#     def reprojection_error(self, T_r2q, p2D_q_gt, mask):
#         """计算重投影误差"""
#         p2D_q, _ = self.world2image(T_r2q)
#         err = torch.sum((p2D_q_gt - p2D_q)**2, dim=-1)
#         #err = self.scaled_barron(1., 2.)(err)[0]/4
#         err = self.masked_mean(err, mask, -1)
#         return err
    
#     def masked_mean(self, x, mask, dim):
#         """计算带掩码的均值，用于处理部分有效数据。"""
#         mask = mask.float()
#         #print("x: ",x)
#         return (mask * x).sum(dim) / mask.sum(dim).clamp(min=1)
    
#     ###分割线 下面是loss的实现###

#     def scaled_loss(self, x, fn, a):
#         """Apply a loss function to a tensor and pre- and post-scale it.
#         Args:
#             x: the data tensor, should already be squared: `x = y**2`.
#             fn: the loss function, with signature `fn(x) -> y`.
#             a: the scale parameter.
#         Returns:
#             The value of the loss, and its first and second derivatives.
#         """
#         a2 = a**2
#         loss, loss_d1, loss_d2 = fn(x/a2)
#         return loss*a2, loss_d1, loss_d2/a2

#     def barron_loss(self,x, alpha, derivatives: bool = True, eps: float = 1e-7):
#         """
#         实现Barron的自适应鲁棒损失函数，用于处理噪声数据。
#         Parameterized  & adaptive robust loss function.
#         Described in:
#             A General and Adaptive Robust Loss Function, Barron, CVPR 2019

#         Contrary to the original implementation, assume the the input is already
#         squared and scaled (basically scale=1). Computes the first derivative, but
#         not the second (TODO if needed).
#         """
#         loss_two = x
#         loss_zero = 2 * torch.log1p(torch.clamp(0.5*x, max=33e37))

#         # The loss when not in one of the above special cases.
#         # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
#         beta_safe = torch.abs(alpha - 2.).clamp(min=eps)
#         # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
#         alpha_safe = torch.where(
#             alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha))
#         alpha_safe = alpha_safe * torch.abs(alpha).clamp(min=eps)

#         loss_otherwise = 2 * (beta_safe / alpha_safe) * (
#             torch.pow(x / beta_safe + 1., 0.5 * alpha) - 1.)

#         # Select which of the cases of the loss to return.
#         loss = torch.where(
#             alpha == 0, loss_zero,
#             torch.where(alpha == 2, loss_two, loss_otherwise))
#         dummy = torch.zeros_like(x)

#         if derivatives:
#             loss_two_d1 = torch.ones_like(x)
#             loss_zero_d1 = 2 / (x + 2)
#             loss_otherwise_d1 = torch.pow(x / beta_safe + 1., 0.5 * alpha - 1.)
#             loss_d1 = torch.where(
#                 alpha == 0, loss_zero_d1,
#                 torch.where(alpha == 2, loss_two_d1, loss_otherwise_d1))

#             return loss, loss_d1, dummy
#         else:
#             return loss, dummy, dummy


#     def scaled_barron(self, a, c):
#         return lambda x: self.scaled_loss(
#                 x, lambda y: self.barron_loss(y, y.new_tensor(a)), c)

#     def optimize_tracking(self, idx, cam_iter, cam_pose, gt_color, gt_depth, batch_size, optimizer,gt_R,gt_T,gt_cam_pose,current_rays,poses_all):
#         """
#         Do one iteration of camera tracking. Sample pixels, render depth/color, calculate loss and backpropagation.

#         Args:
#             cam_iter:
#             cam_pose (tensor): camera pose.
#             gt_color (tensor): ground truth color image of the current frame.
#             gt_depth (tensor): ground truth depth image of the current frame.
#             batch_size (int): batch size, number of sampling rays.
#             optimizer (torch.optim): camera optimizer.

#         Returns:
#             loss (float): The value of loss.
#         """

#         all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)

#         device = self.device
#         H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

#         c2w = cam_pose_to_matrix(cam_pose)  #每一个keyframe的cam pose应该存下来，用于获取c2w，再将其c2w传入后面的get samples中

#         batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(self.ignore_edge_H, H-self.ignore_edge_H,
#                                                                                  self.ignore_edge_W, W-self.ignore_edge_W,
#                                                                                  batch_size, H, W, fx, fy, cx, cy, c2w,
#                                                                                  gt_depth, gt_color, device)

#         with torch.no_grad():
#             det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
#             det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
#             t = (self.bound.unsqueeze(0).to(
#                 device) - det_rays_o) / det_rays_d
#             t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
#             inside_mask = t >= batch_gt_depth
#             inside_mask = inside_mask & (batch_gt_depth > 0)

#         batch_rays_d = batch_rays_d[inside_mask]
#         batch_rays_o = batch_rays_o[inside_mask]
#         batch_gt_depth = batch_gt_depth[inside_mask]
#         batch_gt_color = batch_gt_color[inside_mask]

#         if idx % self.cfg['mapping']['gb_keyframe'] == 0:
#             self.shared_keyframe_list.append(idx)
#             self.tracker_keyframe_dict.append({'rays_o': batch_rays_o.to('cpu'),'gt_depth': batch_gt_depth.to('cpu'),
#                                         'rays_d': batch_rays_d.to('cpu')})
            
#         depth, color, sdf, z_vals = self.renderer.render_batch_ray(all_planes, self.decoders, batch_rays_d, batch_rays_o,
#                                                                    device, self.truncation, gt_depth=batch_gt_depth)
#         depth_error = (batch_gt_depth - depth.clone().detach()).abs()
#         error_median = depth_error.median()
#         depth_mask = (depth_error < 10 * error_median)
#         loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])
#         # loss = loss + self.fs_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])
#         loss = loss + self.w_color * torch.square(batch_gt_color - color)[depth_mask].mean()
#         loss = loss + self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()

#         #2月29日,Coslam的global ba写法,by wjy
#         if idx.item() % self.cfg['mapping']['gb_keyframe'] == 0: 
#             #print("enter global ba")
#             rays, ids = self.keyframeDatabase.sample_global_rays(self.cfg['mapping']['sample'])#
#             idx_cur = random.sample(range(0, self.H * self.W),max(self.cfg['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.cfg['mapping']['min_pixels_cur']))
#             current_rays_batch = current_rays[idx_cur, :]
#             rays = torch.cat([rays.to(device), current_rays_batch.to(device)], dim=0) 
#             ids_all = torch.cat([ids//self.cfg['mapping']['gb_keyframe'], -torch.ones((len(idx_cur)))]).to(torch.int64)
#             rays_d_cam = rays[..., :3].to(device)
#             target_s = rays[..., 3:6].to(device)
#             target_d_1 = rays[..., 6:7].to(device)

#             rays_d_1 = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
#             rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d_1.shape[1], 1).reshape(-1, 3)  #此行代码导致了cuda错误
#             rays_d = rays_d_1.reshape(-1, 3)
#             target_d = target_d_1.squeeze(-1)
#             with torch.no_grad():
#                 det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
#                 det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
#                 t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d
#                 t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
#                 inside_mask = t >= target_d.clone().detach()
#                 inside_mask = inside_mask & (target_d.clone().detach() > 0)
#             co_batch_rays_d = rays_d[inside_mask]
#             co_batch_rays_o = rays_o[inside_mask]
#             co_batch_gt_depth = target_d[inside_mask]
#             co_batch_gt_color = target_s[inside_mask]

#             codepth, cocolor, cosdf, coz_vals = self.renderer.render_batch_ray(all_planes, self.decoders, co_batch_rays_d, co_batch_rays_o,
#                                                                    device, self.truncation, gt_depth=co_batch_gt_depth)

#             depth_error = (co_batch_gt_depth.clone().detach() - codepth.clone().detach()).abs()
#             error_median = depth_error.median()
#             depth_mask = (depth_error < 10 * error_median)
#             loss = loss + self.w_color * torch.square(co_batch_gt_color.clone().detach() - cocolor.clone().detach())[depth_mask].mean()
#             loss = loss + self.w_depth * torch.square(co_batch_gt_depth[depth_mask].clone().detach() - codepth[depth_mask].clone().detach()).mean()
#             #print("exit global ba")
        
#         if idx.item() % self.cfg['mapping']['res_keyframe'] == 0 and idx.item() > self.cfg['mapping']['res_keyframe']:
#             selected_keyframe_idx = random.choice(self.shared_keyframe_list_re)
#             selected_keyframe_idx = self.shared_keyframe_list_re.index(selected_keyframe_idx)#

#             self.selected_keyframe_idx=selected_keyframe_idx
#             data = self.load_p3d()#获取了对应点的 to w 坐标
#             self._data = data
#             c2w_ref = torch.from_numpy(self.c2w_to_w2c(data['T_ref']))

#             T_tgt = torch.inverse(self.gt_c2w)
#             T_tgt = T_tgt.to('cpu')

#             T_r2q_gt = torch.matmul(T_tgt, c2w_ref)
#             data['T_r2q_gt'] = T_r2q_gt
#             p2D_q_gt, mask = self.world2image(data['T_r2q_gt'])

#             T_te = torch.inverse(self.estimate_c2w_list[idx-1])
#             T_te = T_te.to('cpu')
#             T_r2q = torch.matmul(T_te, c2w_ref) #Tidx 估计 @ Tidx-8 gt 估计？

#             p2D_q_i, mask_i = self.world2image(T_r2q)
#             mask = (mask & mask_i).float()

#             err = self.reprojection_error(p2D_q_i,p2D_q_gt,mask).clamp(max=50)
#             err = err.to('cuda:0')
#             reprow = 10
#             loss = loss + reprow * err
            
#         optimizer.zero_grad()
#         loss.backward(retain_graph=True)
#         optimizer.step()
#         return loss.item()
    
    

#     def update_params_from_mapping(self):
#         """
#         Update the parameters of scene representation from the mapping thread.
#         """

#         if self.mapping_idx[0] != self.prev_mapping_idx:
#             if self.verbose:
#                 print('Tracking: update the parameters from mapping')

#             self.decoders.load_state_dict(self.shared_decoders_list[self.shared_cur_rf_id[0]].state_dict())
#             shared_all_planes = self.shared_all_planes_list[self.shared_cur_rf_id[0]]
#             self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz, self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz = shared_all_planes


#             for planes, self_planes in zip(
#                     [self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz],
#                     [self.planes_xy, self.planes_xz, self.planes_yz]):
#                 for i, plane in enumerate(planes):
#                     self_planes[i] = plane.detach().to(self.device)

#             for c_planes, self_c_planes in zip(
#                     [self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz],
#                     [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]):
#                 for i, c_plane in enumerate(c_planes):
#                     self_c_planes[i] = c_plane.detach().to(self.device)

#          ##########################
#             self.prev_mapping_idx = self.mapping_idx[0].clone()

#     def axis_angle_to_matrix(self, data):
#         """
#         将轴角表示为旋转矩阵
#         """
#         batch_dims = data.shape[:-1]

#         theta = torch.norm(data, dim=-1, keepdim=True)
#         omega = data / theta

#         omega1 = omega[...,0:1]
#         omega2 = omega[...,1:2]
#         omega3 = omega[...,2:3]
#         zeros = torch.zeros_like(omega1)

#         K = torch.concat([torch.concat([zeros, -omega3, omega2], dim=-1)[...,None,:],
#                         torch.concat([omega3, zeros, -omega1], dim=-1)[...,None,:],
#                         torch.concat([-omega2, omega1, zeros], dim=-1)[...,None,:]], dim=-2)
#         I = torch.eye(3).expand(*batch_dims,3,3).to(data)

#         return I + torch.sin(theta).unsqueeze(-1) * K + (1. - torch.cos(theta).unsqueeze(-1)) * (K @ K)   
    
#     def matrix_from_tensor(self, rot, trans):
#         """
#         从旋转和平移张量生成变换矩阵
#         """
#         bs = rot.shape[0]
#         T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
#         R = self.axis_angle_to_matrix(rot)
#         T[:, :3, :3] = R
#         T[:, :3, 3] = trans
#         return T
    
#     def get_camera_rays(self, H, W, fx, fy=None, cx=None, cy=None, type='OpenGL'):
#         """Get ray origins, directions from a pinhole camera."""
#         #  ----> i
#         # |
#         # |
#         # X
#         # j
#         i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
#                         torch.arange(H, dtype=torch.float32), indexing='xy')
        
#         # View direction (X, Y, Lambda) / lambda
#         # Move to the center of the screen
#         #  -------------
#         # |      y      |
#         # |      |      |
#         # |      .-- x  |
#         # |             |
#         # |             |
#         #  -------------

#         if cx is None:
#             cx, cy = 0.5 * W, 0.5 * H

#         if fy is None:
#             fy = fx
#         if type is  'OpenGL':
#             dirs = torch.stack([(i - cx)/fx, -(j - cy)/fy, -torch.ones_like(i)], -1)
#         elif type is 'OpenCV':
#             dirs = torch.stack([(i - cx)/fx, (j - cy)/fy, torch.ones_like(i)], -1)
#         else:
#             raise NotImplementedError()

#         rays_d = dirs
#         return rays_d
    
#     def flip_axes(self, estimate):
#         # 构造一个对角矩阵，Y,Z轴取反
#         flip_mat = torch.diag(torch.tensor([1, -1, -1], dtype=estimate.dtype, device=estimate.device))
#         estimate = estimate.squeeze(0)
#         R = estimate[:3, :3]
#         t = estimate[:3, 3]
#         # print("es: ", estimate, estimate.shape)
#         # print("R:", R)
#         # print("t:", t)

#         R_new = R @ flip_mat
#         # t_new = flip_mat @ t

#         new_pose = torch.eye(4, dtype=estimate.dtype, device=estimate.device)
#         new_pose[:3, :3] = R_new
#         new_pose[:3, 3] = t
#         return new_pose
    
#     def flip_rows(self, c2w):
#         c2w = c2w.clone()
#         c2w[1, :] *= -1
#         c2w[2, :] *= -1
#         return c2w



#     def matrix_to_axis_angle(self, rot):
#         """
#         :param rot: [N, 3, 3]
#         :return:
#         """
#         return quaternion_to_axis_angle(matrix_to_quaternion(rot))
    
#     def tensor_to_pil(self, tensor):
#         if isinstance(tensor, torch.Tensor):
#             tensor = tensor.squeeze(0)  # 去掉 batch 维 [1, H, W, 3]
#             tensor = tensor.detach().cpu().numpy()  # 转为 numpy
#             if tensor.max() <= 1.0:
#                 tensor = (tensor * 255).astype(np.uint8)
#             else:
#                 tensor = tensor.astype(np.uint8)
#             return Image.fromarray(tensor)
#         raise TypeError("Input must be a torch.Tensor")

#     def getNextFrame(self):
#         frame_info = self.dataset_cameras[self.frame_id]
#         # print(frame_info)
#         if isinstance(self.cfg, dict):
#             args = SimpleNamespace(**self.cfg)
#         frame = loadCam(args, self.frame_id, frame_info, 1)
#         print("get frame: {}".format(self.frame_id))
#         self.frame_id += 1
#         return frame

#     # 对输入的深度图和法线图进行预处理
#     def map_preprocess_mp(self, frame, frame_id):
#         self.map_input = super().map_preprocess(frame, frame_id)

#     def run(self):
#         """
#             Runs the tracking thread for the input RGB-D frames.

#             Args:
#                 None

#             Returns:
#                 None
#         """
#         # 初始化orb后端
#         self.initialize_orb()

#         tracking_time_sum = 0.0

#         device = self.device
#         batch ={}
#         current_rays = torch.tensor(0.) 
#         poses_all = torch.tensor(0.) 
#         if self.verbose:
#             pbar = self.frame_loader
#         else:
#             pbar = tqdm(self.frame_loader, smoothing=0.05)
#         for idx, gt_color, gt_depth, gt_c2w, cam_info in pbar:
#             # print("0", gt_depth.shape)
#             # print("1", gt_color.shape)
#             crop_edge = 0
#             image_color = Image.open(cam_info["image_path"][0])
#             # print("2", image_color.size)    
#             image_depth = (
#                 np.asarray(Image.open(cam_info["depth_path"][0]), dtype=np.float32) / self.cfg["depth_scale"]
#             )
#             image_color = np.asarray(
#                 image_color.resize((image_depth.shape[1], image_depth.shape[0]))
#             )
#             # print("3", image_color.shape)
#             # print("4", image_depth.shape)
#             if crop_edge > 0:
#                 image_color = image_color[
#                     crop_edge:-crop_edge,
#                     crop_edge:-crop_edge,
#                     :,
#                 ]
#                 image_depth = image_depth[
#                     crop_edge:-crop_edge,
#                     crop_edge:-crop_edge,
#                 ]
#             frame_info = CameraInfo(
#                 uid=int(cam_info["uid"].item()), 
#                 R=cam_info["R"].squeeze().cpu().numpy(), 
#                 T=cam_info["T"].squeeze().cpu().numpy(), 
#                 FovY=cam_info["FovY"].item(), 
#                 FovX=cam_info["FovX"].item(), 
#                 image=Image.fromarray(image_color), 
#                 image_path=cam_info["image_path"][0], 
#                 image_name=cam_info["image_name"][0], 
#                 width=int(cam_info["width"].item()), 
#                 height=int(cam_info["height"].item()), 
#                 depth=Image.fromarray(image_depth), 
#                 depth_path=cam_info["depth_path"][0], 
#                 # pose_gt=cam_info["uid"], 
#                 cx=float(cam_info["cx"].item()), 
#                 cy=float(cam_info["cy"].item()), 
#                 depth_scale=float(cam_info["depth_scale"].item()), 
#                 timestamp=float(cam_info["timestamp"].item()), 
#             )
#             # print("cam_info: ", frame_info)
#             if isinstance(self.cfg, dict):
#                 args = SimpleNamespace(**self.cfg)
#                 frame = loadCam(args, self.frame_id, frame_info, 1)
#             print("get frame: {}".format(self.frame_id))
#             self.frame_id += 1


#             # frame数据接口
#             # frame = self.getNextFrame()

#             if frame is None:
#                 raise RuntimeError("No frame received from the frame loader.")
#             frame_id = frame.uid
#             # if frame_id == idx[0]:
#             #     print("frame_id == idx[0]")
#             # print("current tracker frame = %d" % idx)
#             # update current map

#             move_to_gpu(frame)

#             tracking_start_time = time.time()
#             self.map_preprocess_mp(frame, frame_id)
#             success = self.tracking(frame, self.map_input)
#             tracking_ending_time = time.time()

#             # self.map_input["frame"] = copy.deepcopy(frame)
#             self.map_input["frame"] = frame
#             self.map_input["poses_new"] = self.get_new_poses()
#             move_to_cpu(frame)

#             gt_color = gt_color.to(device, non_blocking=True)
#             gt_depth = gt_depth.to(device, non_blocking=True)
#             gt_c2w = gt_c2w.to(device, non_blocking=True)
#             #batch['direction'] = torch.ones(1, self.H, self.W, 3)  #应该等于self.rays_d
#             batch['rgb'] = gt_color
#             batch['depth'] = gt_depth
#             batch['frame_id'] = idx
#             self.rays_d = self.get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy).unsqueeze(0)     #(1, H, W, 3)
#             batch['direction'] = self.rays_d.to(device, non_blocking=True)  #应该等于self.rays_d
#             if not self.verbose:
#                 pbar.set_description(f"Tracking Frame {idx[0]}")
#             idx = idx[0]

#             # initiate mapping every self.every_frame frames
#             if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
#                 while self.mapping_idx[0] != idx - 1:
#                     time.sleep(0.001)
#                 pre_c2w = self.estimate_c2w_list[idx - 1].unsqueeze(0).to(device)   #获取上一帧位姿

#                 self.update_para_flag = 1

#             if self.pre_shared_cur_rf_id != self.shared_cur_rf_id[0]:      #检查参考帧是否发生变化如果变化则不更新场景参数
#                 self.update_para_flag = 0

#             if self.update_para_flag == 1:
#                 self.update_params_from_mapping()

#             #self.update_params_from_mapping()
#             self.pre_shared_cur_rf_id = self.shared_cur_rf_id[0].clone()    #更新共享参考帧ID

#             all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)

#             if not self.verbose:
#                 print(Fore.MAGENTA)
#                 print("Tracking Frame:  ",  idx.item())
#                 print(Style.RESET_ALL)
#             new_poses = None
#             if idx == 0 or self.gt_camera:
#                 c2w = gt_c2w
#                 if not self.no_vis_on_first_frame:
#                     self.visualizer.save_imgs(idx, 59, gt_depth, gt_color, c2w.squeeze(), all_planes, self.decoders)
#             else:       #恒速假设
#                 """
#                 if self.const_speed_assumption and idx - 2 >= 0:
#                     ## Linear prediction for initialization
#                     pre_poses = torch.stack([self.estimate_c2w_list[idx - 2], pre_c2w.squeeze(0)], dim=0)
#                     pre_poses = matrix_to_cam_pose(pre_poses)
#                     cam_pose = 2 * pre_poses[1:] - pre_poses[0:1]
#                 else:   #用上一帧位姿
#                     ## Initialize with the last known pose
#                     cam_pose = matrix_to_cam_pose(pre_c2w) #四元数+三维平移向量

#                 T = torch.nn.Parameter(cam_pose[:, -3:].clone())
#                 R = torch.nn.Parameter(cam_pose[:,:4].clone())
#                 gt_R = gt_c2w[:, :3, :3].clone()
#                 gt_T = gt_c2w[:, :3, 3].clone()
#                 #print("gt_R:",gt_R)
#                 cam_para_list_T = [T]
#                 cam_para_list_R = [R]
#                 optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr_T, 'betas':(0.5, 0.999)},
#                                                      {'params': cam_para_list_R, 'lr': self.cam_lr_R, 'betas':(0.5, 0.999)}])

#                 current_min_loss = torch.tensor(float('inf')).float().to(device)

#                 ####添加到keyframedatabase,进行global的先前准备
#                 #if idx.item()-1  == 0:
#                     #self.keyframeDatabase.add_keyframe(batch, filter_depth=False)
#                 # 每self.cfg['mapping']['gb_keyframe']帧进行一次全局BA
#                 if idx.item() % self.cfg['mapping']['gb_keyframe'] == 0:
#                     self.keyframeDatabase.add_keyframe(batch, filter_depth=False)
#                     poses = torch.stack([self.estimate_c2w_list[i] for i in range(0, idx.item()-1, self.cfg['mapping']['gb_keyframe'])])

#                     if len(self.keyframeDatabase.frame_ids) < 2:
#                         poses_fixed = torch.nn.parameter.Parameter(poses).to(device)
#                         current_pose = self.estimate_c2w_list[idx-1][None,...]
#                         poses_all = torch.cat([poses_fixed, current_pose], dim=0)
#                     else:
#                         poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(device)
#                         current_pose = self.estimate_c2w_list[idx-1][None,...]
#                         cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
#                         cur_rot = torch.nn.parameter.Parameter(self.matrix_to_axis_angle(poses[:, :3, :3]))
#                         pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(device)#获取当前帧姿态参数的优化器

#                         poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)#将优化后的姿态参数添加到poses_all中
#                         print("poses_all:",poses_all.size())

#                     current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)#根据采样策略选择采样的射线，包括来自关键帧的射线和当前帧的射线
#                     current_rays = current_rays.reshape(-1, current_rays.shape[-1])#reshape成张量

#                 if idx.item() % self.cfg['mapping']['res_keyframe'] == 0:
#                     self.gt_campose_list.append(gt_c2w) #只储存keyframe的campose

#                     self.shared_keyframe_list_re.append(idx)
#                 for cam_iter in range(self.num_cam_iters):
#                     cam_pose = torch.cat([R, T], -1)

#                     self.gt_c2w = gt_c2w
#                     self.visualizer.save_imgs(idx, cam_iter, gt_depth, gt_color, cam_pose, all_planes, self.decoders)
                    
#                     loss = self.optimize_tracking(idx, cam_iter, cam_pose, gt_color, gt_depth, self.tracking_pixels, optimizer_camera, gt_R, gt_T, gt_c2w, current_rays,poses_all)
#                     if loss < current_min_loss:
#                         current_min_loss = loss
#                         candidate_cam_pose = cam_pose.clone().detach()
#                 """
#                 # c2w = cam_pose_to_matrix(candidate_cam_pose)
#                 if self.use_orb_backend:
#                     if self.type == "Replica" or self.type == "ScanNet":
#                         new_poses = self.get_new_poses()
#                         new_poses = [torch.from_numpy(pose).clone().detach() for pose in new_poses]
#                         new_poses = [self.flip_axes(pose) for pose in new_poses]
#                         new_poses = [self.flip_rows(pose) for pose in new_poses]  # 翻转行
#                         new_poses = [self.gt_c2w_list[0].to(pose.dtype) @ pose for pose in new_poses]  # 将估计位姿转换为世界坐标系下的位姿
#                     else:
#                         new_poses = self.get_new_poses()
#                         new_poses = [torch.from_numpy(pose).clone().detach() for pose in new_poses]
#                         new_poses = [self.flip_axes(pose) for pose in new_poses]
#                 else:
#                     c2w = torch.from_numpy(self.pose_es[-1]).float().clone().detach()  # 获取最后一帧的估计位姿===
#                     assert idx != 0

#                     c2w = self.flip_axes(c2w)

#                     # print("c2w_defore: ", c2w, c2w.shape, c2w.dtype)
#                     if self.type == "Replica" or self.type == "ScanNet":
#                         print("✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖✖")
#                         c2w = self.flip_rows(c2w)  # 翻转行
#                         c2w = self.gt_c2w_list[0] @ c2w  # 将估计位姿转换为世界坐标系下的位姿
#             if self.use_orb_backend:
#                 if new_poses is None:
#                     self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()
#                 else:
#                     for i, pose in enumerate(new_poses[1:], start=1):
#                         self.estimate_c2w_list[i] = pose.to(self.device)
#             else:
#                 # print("c2w: ", c2w, c2w.shape, c2w.dtype)
#                 self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()


#             print("estimate: ", self.estimate_c2w_list[idx], self.estimate_c2w_list[idx].shape, self.estimate_c2w_list[idx].dtype)

#             self.gt_c2w_list[idx] = gt_c2w.squeeze(0).clone()
#             print("gt_c2w:{} ", self.gt_c2w_list[idx], self.gt_c2w_list[idx].shape, self.gt_c2w_list[idx].dtype)

#             # # 检查是否仅是轴翻转导致的差异（例如 z轴或 y轴取反）
#             # R_est = self.estimate_c2w_list[idx][:3, :3].double()
#             # R_gt = gt_c2w.squeeze(0)[:3, :3].double()

#             # # 对比旋转矩阵的差异
#             # R_diff = R_est @ R_gt.T
#             # print("R_diff: ", R_diff)

#             # # 相对位姿误差
#             # rel_pose = torch.linalg.inv(gt_c2w.squeeze(0)).double()@ self.estimate_c2w_list[idx].double()
#             # print("rel_pose", rel_pose)

#             # t_est = self.estimate_c2w_list[idx][:3, 3].double()
#             # t_gt = gt_c2w.squeeze(0)[:3, 3].double()
#             # print("t_est: ", t_est)
#             # print("t_gt: ", t_gt)
#             # translation_error = torch.norm(t_est - t_gt)

#             # R_est = self.estimate_c2w_list[idx][:3, :3].double()
#             # R_gt = gt_c2w.squeeze(0)[:3, :3].double()
#             # R_rel = R_est @ R_gt.T
#             # angle_error = torch.acos(((torch.trace(R_rel) - 1) / 2).clamp(-1, 1)) * 180 / torch.pi

#             # print(f"Translation Error: {translation_error:.4f} m")
#             # print(f"Rotation Error: {angle_error:.2f} degrees")

#             # from scipy.spatial.transform import Rotation as R

#             # R_est = R.from_matrix(self.estimate_c2w_list[idx][:3, :3].cpu().numpy())
#             # R_gt = R.from_matrix(gt_c2w.squeeze(0)[:3, :3].cpu().numpy())

#             # R_diff = R_est.inv() * R_gt
#             # angle = R_diff.magnitude() * 180 / np.pi  # degree
#             # axis = R_diff.as_rotvec()
#             # print("Rotation Error: ", angle, "deg")
#             # print("Rotation Axis: ", axis)


#             pre_c2w = c2w.clone()
#             self.idx[0] = idx

#             tracking_oneframe_time = tracking_ending_time - tracking_start_time
#             print("Tracking time: {:.3f} seconds".format(tracking_oneframe_time))
#             tracking_time_sum += tracking_oneframe_time
#         tracking_mean_time = tracking_time_sum / self.n_img
#         print("Average tracking time per frame: {:.3f} seconds".format(tracking_mean_time))
#         self.save_traj(self.save_path)
#         if self.use_orb_backend:
#             new_poses = self.get_new_poses()
#             new_poses = [torch.from_numpy(pose).clone().detach() for pose in self.get_new_poses()]
#             new_poses = [self.flip_axes(pose) for pose in new_poses]
#             if new_poses is None:
#                 self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()
#             else:
#                 for i, pose in enumerate(new_poses[1:], start=1):
#                     self.estimate_c2w_list[i] = pose.to(self.device)

#         self.logger.log(idx, self.shared_keyframe_list)

