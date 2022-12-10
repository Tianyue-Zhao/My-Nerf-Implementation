import torch
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from random import sample
from ray_stuff import ray_from_pixels, ray_batch_to_points, cumprod_exclusive
from network import embed_tensor, implicit_network
from math import ceil

def visualize_batch(rays, batch_size, near, far, num_samples, rgb_data):
    #near = 2
    #far = 4.5
    #num_samples = 100
    #batch_size = 1600
    cur_indices = sample(range(rays.shape[0]), batch_size)
    cur_batch = rays[cur_indices, :]
    cur_points, cur_directions, distances = ray_batch_to_points(cur_batch,\
        near, far, num_samples, False, 0.7)
    point_vector = o3d.utility.Vector3dVector(cur_points)
    bounding_box = draw_bounding_box()
    pcd = o3d.geometry.PointCloud(point_vector)
    colors = rgb_data[cur_indices, :]
    colors = np.stack([colors] * num_samples, axis = 1).reshape((-1, 3)).astype(np.float64)
    #colors = o3d.utility.Vector3dVector(colors)
    #pcd.colors = colors
    o3d.visualization.draw_geometries([pcd, bounding_box])

def visualize_implicit_field(implicit_function, device):
    sigma_threshold = 0.2
    num_points = 150
    batch_points = 400000

    bounding_box_parameters = array_from_file('bottles/bbox.txt')
    bounding_box = bounding_box_parameters[0, :6].reshape((2, 3))
    values = np.linspace(bounding_box[0, :], bounding_box[1, :], num_points)
    x, y, z = np.meshgrid(values[:, 0], values[:, 1], values[:, 2])
    points = np.stack([x, y, z], axis = 3)
    points = points.reshape((-1, 3))
    display_points = []

    for i in range(ceil(points.shape[0] / batch_points)):
        cur_low = i * batch_points
        cur_high = min((i + 1) * batch_points, points.shape[0])
        cur_points_numpy = points[cur_low : cur_high, :]
        cur_points = torch.tensor(cur_points_numpy, device = device, dtype = torch.float32)
        directions = torch.zeros(cur_points.shape, device = device, dtype = torch.float32)
        directions[:, 0] = 1
        cur_points = embed_tensor(cur_points, L = 10)
        directions = embed_tensor(directions, L = 4)
        sigma_value, rgb_value = implicit_function(cur_points, directions)
        sigma_value = sigma_value.detach().cpu().numpy()
        sigma_value = sigma_value.reshape(-1)
        print(np.max(sigma_value))
        print(np.min(sigma_value))
        cur_points = cur_points_numpy[sigma_value > sigma_threshold, :]
        display_points.append(cur_points)
        torch.cuda.empty_cache()
        del cur_points
        del directions
        del sigma_value
        del rgb_value

    points = np.concatenate(display_points, axis = 0)
    points = o3d.utility.Vector3dVector(points)
    pcd = o3d.geometry.PointCloud(points)
    bounding_box = draw_bounding_box()
    o3d.visualization.draw_geometries([pcd, bounding_box])

def draw_bounding_box():
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

def array_from_file(filename):
    input_file = open(filename, 'r')
    lines = input_file.readlines()
    input_file.close()
    data_list = []
    for line in lines:
        line = line.strip()
        if(len(line) == 0):
            continue
        line = line.split(' ')
        data_list.append([float(item) for item in line])
    return np.asarray(data_list)

def depth_from_picture(evaluation_model, device):
    pictures = ['1_val_0031', '1_val_0032', '1_val_0033', '1_val_0034']
    intrinsics = array_from_file('bottles/intrinsics.txt')
    batch = 1024
    near = 2
    far = 4.5
    num_samples = 200
    encoding_position = 10
    encoding_direction = 4
    for picture in pictures:
        image = cv2.imread("bottles/rgb/" + picture + ".png")
        extrinsic = array_from_file("bottles/pose/" + picture + ".txt")
        rays = ray_from_pixels(800, intrinsics, extrinsic)
        depth = np.zeros(800 ** 2)
        for i in range(ceil(rays.shape[0] / batch)):
            cur_low = i * batch
            cur_high = min((i + 1) * batch, rays.shape[0])

            cur_rays = rays[cur_low:cur_high, :]
            cur_points, cur_directions, distances = ray_batch_to_points(cur_rays, near, far, num_samples, False, 0.7)
            cur_points = torch.tensor(cur_points, device = device, dtype = torch.float32)
            cur_directions = torch.tensor(cur_directions, device = device, dtype = torch.float32)
            distances = torch.tensor(distances, device = device, dtype = torch.float32)
            cur_points = embed_tensor(cur_points, L = encoding_position)
            cur_directions = embed_tensor(cur_directions, L = encoding_direction)
            sigma_value, rgb_value = evaluation_model(cur_points, cur_directions)

            # Find weight values in the way as in ray marching
            cur_points = torch.reshape(cur_points, (-1, num_samples, 3))
            sigma_value = torch.reshape(sigma_value, (-1, num_samples))
            rgb_value = torch.reshape(rgb_value, (-1, num_samples, 3))
            interval_lengths = distances[:, 1:] - distances[:, :-1] # The lengths of the intervals between cur_points
            interval_lengths = torch.cat([interval_lengths,\
                1e9 * torch.ones((interval_lengths.shape[0], 1), device = interval_lengths.device)], dim = 1)
            alpha = 1 - torch.exp(-sigma_value * interval_lengths * 100) # Each point approximates values for the interval after it
            weights = alpha * cumprod_exclusive(1 - alpha + 1e-9)
            weights, indices = torch.sort(weights, dim = 1)
            distances = torch.gather(distances, 1, indices[:, num_samples - 1:]).detach().cpu().numpy()
            depth[cur_low : cur_high] = distances[:, 0]
        #mapped_depth = 245 + (70 - 245) * (depth - near) / 1.8
        #mapped_depth[mapped_depth < 0] = 0
        #mapped_depth = mapped_depth.reshape((800, 800))
        #mapped_depth = np.stack([mapped_depth] * 3, axis = 2).astype(np.uint8)
        # The below is a mapping that is not as originally intended but works
        # very well rather unexpectedly
        mapped_depth = 255 + (70 - 255) * depth / (far - near)
        mapped_depth = mapped_depth.reshape((800, 800))
        mapped_depth = np.maximum(np.zeros_like(mapped_depth), mapped_depth - 0.7)
        mapped_depth = np.stack([mapped_depth] * 3, axis = 2).astype(np.uint8)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image)
        ax2.imshow(mapped_depth)
        plt.savefig('evaluation_pictures/' + picture + '_depth.png')