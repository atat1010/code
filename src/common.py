import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
import torch.nn.functional as F

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]

    return K


def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
    """
    Hierarchical sampling in NeRF paper.
    """
    # Get pdf
    # weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    pdf = weights

    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def random_select(l, k):
    """
    Random select k values from 0..l.

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])

def get_rays_from_uv(i, j, c2ws, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i, device=device)], -1)
    dirs = dirs.unsqueeze(-2)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2ws[:, None, :3, :3], -1)
    rays_o = c2ws[:, None, :3, -1].expand(rays_d.shape)

    return rays_o, rays_d

def select_uv(i, j, n, b, depths, colors, device='cuda:0'):
    """
    Select n uv from dense uv.

    """
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n * b,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]  # (n * b)
    j = j[indices]  # (n * b)

    indices = indices.reshape(b, -1)
    i = i.reshape(b, -1)
    j = j.reshape(b, -1)

    depths = depths.reshape(b, -1)
    colors = colors.reshape(b, -1, 3)

    depths = torch.gather(depths, 1, indices)  # (b, n)
    colors = torch.gather(colors, 1, indices.unsqueeze(-1).expand(-1, -1, 3))  # (b, n, 3)

    return i, j, depths, colors

def get_sample_uv(H0, H1, W0, W1, n, b, depths, colors, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    depths = depths[:, H0:H1, W0:W1]
    colors = colors[:, H0:H1, W0:W1]

    i, j = torch.meshgrid(torch.linspace(W0, W1 - 1, W1 - W0, device=device), torch.linspace(H0, H1 - 1, H1 - H0, device=device))

    i = i.t()  # transpose
    j = j.t()
    i, j, depth, color = select_uv(i, j, n, b, depths, colors, device=device)
    # visualize_sampled_points_on_image(colors, j, i, count=n)
    # save_sampled_points_on_image(colors, j, i, save_path="/home/qm/qmslam/qmslam/output/paper_pic", idx=0, count=n)
    return i, j, depth, color

def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2ws, depths, colors, device):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.

    """
    b = c2ws.shape[0]
    i, j, sample_depth, sample_color = get_sample_uv(
        H0, H1, W0, W1, n, b, depths, colors, device=device)


    rays_o, rays_d = get_rays_from_uv(i, j, c2ws, H, W, fx, fy, cx, cy, device)

    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), sample_depth.reshape(-1), sample_color.reshape(-1, 3)


import torch
import torch.nn.functional as F

def get_edge_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy,
                     c2ws, depths, colors, device, edge_thresh=0.2):
    """
    支持 colors 为 [B, H, W, 3] 格式。
    """
    b = c2ws.shape[0]

    # 裁剪深度和颜色图
    depths_crop = depths[:, H0:H1, W0:W1]                  # (B, h, w)
    colors_crop = colors[:, H0:H1, W0:W1, :]                # (B, h, w, 3)

    # 转灰度图 (B, h, w)
    gray = (0.299 * colors_crop[:, :, :, 0] +
            0.587 * colors_crop[:, :, :, 1] +
            0.114 * colors_crop[:, :, :, 2]).float()

    # Sobel 卷积
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)

    gray = gray.unsqueeze(1)  # [B, 1, h, w]
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze(1)  # [B, h, w]
    grad_mag = grad_mag / (grad_mag.max() + 1e-8)

    rays_o_list, rays_d_list, depth_list, color_list = [], [], [], []

    for idx in range(b):
        mask = grad_mag[idx] > edge_thresh
        ys, xs = torch.nonzero(mask, as_tuple=True)

        if ys.numel() == 0:
            continue

        num = min(n, ys.shape[0])
        perm = torch.randperm(ys.shape[0], device=device)[:num]
        j = ys[perm] + H0   # row
        i_ = xs[perm] + W0  # col

        # visualize_sampled_points_on_image(colors, j, i_, idx=idx, count=num)
        # save_sampled_points_on_image(colors, j, i_, save_path="/home/qm/qmslam/qmslam/output/paper_pic_edge", idx=0, count=num)
        # return



        # 获取深度与颜色
        d = depths[idx, j, i_]
        c = colors[idx, j, i_, :]  # [n, 3]

        # 射线方向
        i_expand = i_.unsqueeze(0)         # [1, n]
        j_expand = j.unsqueeze(0)          # [1, n]
        c2w = c2ws[idx].unsqueeze(0)       # [1, 4, 4]
        rays_o, rays_d = get_rays_from_uv(
            i_expand, j_expand, c2w, H, W, fx, fy, cx, cy, device)

        rays_o_list.append(rays_o.squeeze(0))
        rays_d_list.append(rays_d.squeeze(0))
        depth_list.append(d)
        color_list.append(c)

    if len(rays_o_list) == 0:
        return (
            torch.empty(0, 3, device=device),
            torch.empty(0, 3, device=device),
            torch.empty(0, device=device),
            torch.empty(0, 3, device=device)
        )

    return (
        torch.cat(rays_o_list, dim=0),
        torch.cat(rays_d_list, dim=0),
        torch.cat(depth_list, dim=0),
        torch.cat(color_list, dim=0)
    )

import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_sampled_points_on_image(colors, j, i_, idx=0, title="Sampled edge points", count=None):
    """
    在图像上可视化采样点，并加上采样数量作为图例 label。

    参数：
    - colors: [B, H, W, 3] 图像张量
    - j, i_: 采样点的像素坐标
    - idx: 第几帧
    - title: 图标题
    - count: 显示的采样点数量（int，可选）
    """

    img = colors[idx].detach().cpu().numpy()  # [H, W, 3]
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    j_np = j.detach().cpu().numpy()
    i_np = i_.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.scatter(i_np, j_np, c='red', s=6,
                label=f"Sampled points: {count if count is not None else len(i_np)}")
    plt.title(title)
    plt.axis('off')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def save_sampled_points_on_image(colors, j, i_, save_path, idx=0, title="Sampled edge points", count=None, dpi=600):
    """
    在图像上可视化采样点，并保存为图片文件。

    参数：
    - colors: [B, H, W, 3] 图像张量
    - j, i_: 采样点的像素坐标
    - save_path: 保存路径（包含文件名和扩展名）
    - idx: 第几帧
    - title: 图标题
    - count: 显示的采样点数量（int，可选）
    - dpi: 图像分辨率，默认300
    """

    img = colors[idx].detach().cpu().numpy()  # [H, W, 3]
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    j_np = j.detach().cpu().numpy()
    i_np = i_.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.scatter(i_np, j_np, c='red', s=4)
    # plt.title(title)
    plt.axis('off')
    # plt.legend(loc='upper right')
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # 关闭图像以释放内存
    
    print(f"图片已保存到: {save_path}")

# 使用示例：
# save_sampled_points_on_image(colors, j, i_, "sampled_points.png", idx=0, title="边缘采样点", count=100)

def matrix_to_cam_pose(batch_matrices, RT=True):
    """
    Convert transformation matrix to quaternion and translation.
    Args:
        batch_matrices: (B, 4, 4)
        RT: if True, return (B, 7) with [R, T], else return (B, 7) with [T, R]
    Returns:
        (B, 7) with [R, T] or [T, R]
    """
    if RT:
        return torch.cat([matrix_to_quaternion(batch_matrices[:,:3,:3]), batch_matrices[:,:3,3]], dim=-1)
    else:
        return torch.cat([batch_matrices[:, :3, 3], matrix_to_quaternion(batch_matrices[:, :3, :3])], dim=-1)

def cam_pose_to_matrix(batch_poses):
    """
    Convert quaternion and translation to transformation matrix.
    Args:
        batch_poses: (B, 7) with [R, T] or [T, R]
    Returns:
        (B, 4, 4) transformation matrix
    """
    c2w = torch.eye(4, device=batch_poses.device).unsqueeze(0).repeat(batch_poses.shape[0], 1, 1)
    c2w[:,:3,:3] = quaternion_to_matrix(batch_poses[:,:4])
    c2w[:,:3,3] = batch_poses[:,4:]

    return c2w

def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def normalize_3d_coordinate(p, bound):
    """
    Normalize 3d coordinate to [-1, 1] range.
    Args:
        p: (N, 3) 3d coordinate
        bound: (3, 2) min and max of each dimension
    Returns:
        (N, 3) normalized 3d coordinate

    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p
