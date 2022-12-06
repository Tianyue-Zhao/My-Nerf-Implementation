import torch
import numpy as np

def ray_from_pixels(positions, intrinsic, extrinsic):
    focal_length = intrinsic[0, 0]
    n = 100
    focal_length /= 4
    camera_frame = [(positions[:, 0] - n) / focal_length,
                    (positions[:, 1] - n) / focal_length,
                    np.ones(positions.shape[0])]
    camera_frame = np.stack(camera_frame, axis = 1)
    transformation = extrinsic
    camera_frame = np.concatenate([camera_frame, np.ones((positions.shape[0], 1))], axis = 1)
    world_frame = transformation @ camera_frame.T
    world_frame = world_frame.T
    camera_location = transformation[:3, 3]
    camera_location = np.stack([camera_location] * positions.shape[0], axis = 0)
    directions = world_frame[:, :3] - camera_location
    directions /= np.linalg.norm(directions, axis = 1)[:, None]
    return np.concatenate([camera_location, directions], axis = 1)

# Alternative way with origins on the image plane
def ray_from_pixels_plane(positions, intrinsic, extrinsic):
    focal_length = intrinsic[0, 0]
    n = 100
    focal_length /= 4
    camera_frame = [(positions[:, 0] - n) / focal_length,
                    (positions[:, 1] - n) / focal_length,
                    np.ones(positions.shape[0])]
    camera_frame = np.stack(camera_frame, axis = 1)
    transformation = extrinsic
    camera_origin = np.concatenate([camera_frame, np.ones((positions.shape[0], 1))], axis = 1)
    camera_origin = transformation @ camera_origin.T
    camera_origin = camera_origin.T
    camera_origin = camera_origin[:, :3]
    camera_direction = transformation[:3, :3] @ camera_frame.T
    camera_direction = camera_direction.T
    camera_direction /= np.linalg.norm(camera_direction, axis = 1)[:, None]
    return np.concatenate([camera_origin, camera_direction], axis = 1)

# Stratified sampling
def ray_batch_to_points(rays, near, far, num_samples, inverse, perturb):
    t_val = np.linspace(0, 1, num = num_samples)
    if(inverse):
        distances = 1 / (1 / near * (1 - t_val) + 1/ far * t_val)
    else:
        distances = near * (1 - t_val) + far * t_val
    midpoints = (distances[1:] + distances[:-1]) / 2
    upper_bound = np.zeros(distances.shape)
    upper_bound[:-1] = midpoints
    upper_bound[-1] = distances[-1]
    lower_bound = np.zeros(distances.shape)
    lower_bound[1:] = midpoints
    lower_bound[0] = distances[0]
    random_values = np.random.rand(rays.shape[0], num_samples)
    distances = lower_bound + (upper_bound - lower_bound) * random_values
    distances = distances[:, :, None]
    directions = rays[:, None, 3:]
    points = rays[:, None, :3]
    points = points + distances * directions
    points = points.reshape((-1, 3))
    directions = np.concatenate([directions] * num_samples, axis = 1)
    directions = directions.reshape((-1, 3))
    return points, directions, distances

def ray_march(points, directions, distances, sigma_value, rgb_value, num_samples):
    # Reshape the values to distinguish between different rays
    points = torch.reshape(points, (-1, num_samples, 3))
    directions = torch.reshape(directions, (-1, num_samples, 3))
    distances = torch.reshape(distances, (-1, num_samples))
    sigma_value = torch.reshape(sigma_value, (-1, num_samples))
    rgb_value = torch.reshape(rgb_value, (-1, num_samples, 3))
    # Compute the alpha values
    interval_lengths = distances[:, 1:] - distances[:, :-1] # The lengths of the intervals between points
    interval_lengths = torch.cat([interval_lengths,\
        1e9 * torch.ones((interval_lengths.shape[0], 1), device = interval_lengths.device)], dim = 1)
    alpha = 1 - torch.exp(-sigma_value * interval_lengths * 100) # Each point approximates values for the interval after it
    weights = alpha * torch.cumprod(1 - alpha + 1e-9, dim = 1)
    # Compute the output rgb values with the weights
    rgb_value = torch.sum(weights[..., None] * rgb_value, dim = 1)
    return rgb_value