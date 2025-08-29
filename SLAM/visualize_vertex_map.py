import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from icp import *

def visualize_vertex_map(vertex_map):
    """
    可视化顶点图
    :param vertex_map: 顶点图，形状为 [H, W, 3]
    """
    # 将顶点图转换为点云
    points = vertex_map.reshape(-1, 3).cpu().numpy()  # 转换为 numpy 数组
    points = points[points[:, 2] > 0]  # 过滤掉无效点（深度为0）

    # 使用 open3d 可视化点云
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud], window_name="Vertex Map Point Cloud")

# 示例调用
vertex_map = compute_vertex(depth_map, K)  # 生成顶点图
visualize_vertex_map(vertex_map)