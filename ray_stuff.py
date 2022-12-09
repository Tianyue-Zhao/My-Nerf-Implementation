import torch
import numpy as np

def ray_from_pixels(image_dimension, intrinsic, extrinsic):
    positions = []
    for i in range(image_dimension):
        for k in range(image_dimension):
            positions.append([k, i])
    positions = np.asarray(positions)
    focal_length = intrinsic[0, 0]
    downsample_factor = 800 / image_dimension
    n = 400 / downsample_factor
    focal_length = focal_length / downsample_factor
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
def ray_from_pixels_plane(image_dimension, intrinsic, extrinsic):
    positions = []
    for i in range(image_dimension):
        for k in range(image_dimension):
            positions.append([k, i])
    positions = np.asarray(positions)
    focal_length = intrinsic[0, 0]
    downsample_factor = 800 / image_dimension
    n = 400 / downsample_factor
    focal_length = focal_length / downsample_factor
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
    return np.concatenate([camera_origin, camera_direction], axis = 1, dtype = np.float32)

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
    directions = rays[:, None, 3:]
    points = rays[:, None, :3]
    points = points + distances[:, :, None] * directions
    points = points.reshape((-1, 3))
    directions = np.concatenate([directions] * num_samples, axis = 1)
    directions = directions.reshape((-1, 3))
    return points, directions, distances

def sample_points_weighted(rays, sigma_value, distances, num_samples, fine_samples):
    num_total = num_samples + fine_samples
    points = rays[:, None, :3]
    points = torch.tensor(points, device = sigma_value.device, dtype = torch.float32)
    directions = rays[:, None, 3:]
    directions = torch.tensor(directions, device = sigma_value.device, dtype = torch.float32)
    distances = distances[:, :]
    interval_lengths = distances[:, 1:] - distances[:, :-1]

    # Translate sigma to weights
    sigma_value = torch.reshape(sigma_value, (-1, num_samples))
    interval_lengths = torch.cat([interval_lengths,\
        1e9 * torch.ones((interval_lengths.shape[0], 1), device = interval_lengths.device)], dim = 1)
    alpha = 1 - torch.exp(-sigma_value * interval_lengths * 100) # Each point approximates values for the interval after it
    weights = alpha * torch.cumprod(1 - alpha + 1e-9, dim = 1)

    # Sample points to translate from the weight distribution
    weights = weights[:, 1:-1] + 1e-5
    pdf = weights / torch.sum(weights, dim = 1, keepdim = True)
    cdf = torch.cumsum(pdf, dim = 1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim = 1)
    samples = torch.rand((cdf.shape[0], fine_samples), device = sigma_value.device)
    samples = samples.contiguous()
    indices = torch.searchsorted(cdf, samples, right = True) # Rays x fine_samples
    indices = torch.min((num_samples - 2) * torch.ones_like(indices), indices)
    if(torch.min(indices) < 1):
        print("At most 0 somehow")
        print(torch.min(indices))
        print(cdf.shape)
        exit()

    # Sample on distribution
    new_distances = 0.5 * (distances[:, 1:] + distances[:, :-1]) # Taking the midpoints according to the original code
    interval_lengths = new_distances[:, 1:] - new_distances[:, :-1] # The lengths of the intervals between points
    near_distances = torch.gather(new_distances, 1, indices - 1) # The near distance of the interval each selected point is in
    interval_lengths = torch.gather(interval_lengths, 1, indices - 1)
    probability_interval = torch.gather(pdf, 1, indices - 1)
    probability_start = torch.gather(cdf, 1, indices - 1)
    new_distances = near_distances + (samples - probability_start) * interval_lengths / probability_interval

    # Calculate points and merge
    distances = torch.cat([distances, new_distances], dim = 1)
    distances, sort_indices = torch.sort(distances, dim = 1)
    points = points + distances[:, :, None] * directions
    points = points.reshape((-1, 3))
    directions = torch.cat([directions] * num_total, dim = 1)
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