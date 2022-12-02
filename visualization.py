import numpy as np
import open3d as o3d
from random import sample
from ray_stuff import ray_batch_to_points

def visualize_batch(rays, batch_size, near, far, num_samples, rgb_data):
    cur_indices = sample(range(rays.shape[0]), batch_size)
    cur_batch = rays[cur_indices, :]
    cur_points, cur_directions, distances = ray_batch_to_points(cur_batch,\
        near, far, num_samples, False, 0.7)
    point_vector = o3d.utility.Vector3dVector(cur_points)
    bounding_box = draw_bounding_box()
    pcd = o3d.geometry.PointCloud(point_vector)
    colors = rgb_data[cur_indices, :]
    colors = np.stack([colors] * num_samples, axis = 1).reshape((-1, 3)).astype(np.float64)
    colors = o3d.utility.Vector3dVector(colors)
    pcd.colors = colors
    o3d.visualization.draw_geometries([pcd, bounding_box])

def draw_bounding_box():
    from train import array_from_file
    bounding_box_parameters = array_from_file('bottles/bbox.txt')
    bounding_box = bounding_box_parameters[0, :6].reshape((2, 3))
    lower_point = np.stack([bounding_box[0,:]] * 4, axis = 0)
    upper_point = np.stack([bounding_box[1,:]] * 4, axis = 0)
    lower_point[1, 0] = upper_point[0, 0]
    lower_point[2, 1] = upper_point[0, 1]
    lower_point[3, 2] = upper_point[0, 2]
    upper_point[1, 0] = lower_point[0, 0]
    upper_point[2, 1] = lower_point[0, 1]
    upper_point[3, 2] = lower_point[0, 2]
    bounding_box = np.concatenate([lower_point, upper_point], axis = 0)
    lines = [[0, 1],[0, 2],[0, 3],
             [4, 5],[4, 6],[4, 7],
             [1, 6],[1, 7],[2, 5],[2, 7],[3, 5],[3, 6]]
    points = o3d.utility.Vector3dVector(bounding_box)
    lines = o3d.utility.Vector2iVector(lines)
    line_set = o3d.geometry.LineSet(points = points, lines = lines)
    return line_set