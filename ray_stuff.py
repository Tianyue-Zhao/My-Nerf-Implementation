import numpy as np

def ray_from_pixels(positions, intrinsic, extrinsic):
    focal_length = intrinsic[0, 0]
    n = intrinsic[0, 2]
    camera_frame = [(n - positions[:,0]) / focal_length,
                    (positions[:,1] - n) / focal_length,
                    -np.ones(positions.shape[0])]
    camera_frame = np.stack(camera_frame, axis = 1)
    transformation = extrinsic
    camera_frame = np.concatenate([camera_frame, np.ones((positions.shape[0], 1))], axis = 1)
    world_frame = transformation @ camera_frame.T
    world_frame = world_frame.T
    camera_location = transformation[:3, 3]
    camera_location = np.stack([camera_location] * positions.shape[0], axis = 0)
    directions = world_frame - camera_location
    directions /= np.linalg.norm(directions, axis = 1)
    return np.concatenate([camera_location, directions], axis=1)

# Stratified sampling
def ray_batch_to_points(rays, near, far, num_samples, inverse, perturb):
    t_val = np.linspace(0, 1, num = num_samples)
    if(inverse):
        distances = 1 / (1 / near * (1 - t_val) + 1/ far * t_val)
    else:
        distances = near * (1 - t_val) + far * t_val
    midpoints = (distances[1:] + distances[:-1]) / 2
    upper_bound = np.concatenate([midpoints, distances[-1]])
    lower_bound = np.concatenate([distances[0], midpoints])
    random_values = np.random.rand(t_val.shape)
    distances = lower_bound + (upper_bound - lower_bound) * random_values
    return rays[:3] + distances * rays[3:]