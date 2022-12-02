import cv2
import os
import torch
import pickle
import numpy as np
from random import sample
from math import ceil
from network import embed_tensor, implicit_network
from ray_stuff import ray_from_pixels, ray_batch_to_points, ray_march
from visualization import visualize_batch

# Configuration variables
image_dimension = 200
downsample = True

near = 1.5
far = 4.5
num_samples = 100
# Hierarichical sampling comes later

train_steps = 5000
batch_size = 3200 # Number of rays each batch
lr = 0.0001
encoding_position = 10
encoding_direction = 4
evaluation_run = False
evaluation_poses = ["1_val_0016", "1_val_0018", "1_val_0020", "1_val_0022"]
evaluation_path = "evaluation_pictures/"
evaluation_batch = 400

# Path names
train_pickle_name = 'train_information.data'
train_image_path = 'bottles/rgb/'
train_pose_path = 'bottles/pose/'
intrinsic_matrix_path = 'bottles/intrinsics.txt'
implicit_weight_file = 'implicit_weights.data'
load_weights = False

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

def evaluate():
    intrinsics = array_from_file(intrinsic_matrix_path)
    for pose in evaluation_poses:
        extrinsic = array_from_file(train_pose_path + pose + '.txt')
        positions = []
        for i in range(image_dimension):
            for k in range(image_dimension):
                positions.append([i, k])
        positions = np.asarray(positions)
        evaluation_rays = ray_from_pixels(positions, intrinsics, extrinsic)
        rgb_predicted = np.zeros((image_dimension ** 2, 3), dtype = np.float32)

        for i in range(ceil(evaluation_rays.shape[0] / evaluation_batch)):
            cur_low = i * evaluation_batch
            cur_high = min((i + 1) * evaluation_batch, evaluation_rays.shape[0])

            cur_rays = evaluation_rays[cur_low:cur_high, :]
            cur_points, cur_directions, distances = ray_batch_to_points(cur_rays, near, far, num_samples, False, 0.7)
            cur_points = torch.tensor(cur_points, device = device, dtype = torch.float32)
            cur_directions = torch.tensor(cur_directions, device = device, dtype = torch.float32)
            distances = torch.tensor(distances, device = device, dtype = torch.float32)
            # Call the positional encoding and ask the implicit function about sigma and rgb
            cur_points = embed_tensor(cur_points, L = encoding_position)
            cur_directions = embed_tensor(cur_directions, L = encoding_direction)
            sigma_value, rgb_value = implicit_function(cur_points, cur_directions)
            rgb_batch = ray_march(cur_points, cur_directions, distances, sigma_value, rgb_value, num_samples)
            rgb_batch = rgb_batch.detach().cpu().numpy()
            rgb_predicted[cur_low:cur_high, :] = rgb_batch

        rgb_predicted *= 255
        rgb_predicted = rgb_predicted.reshape((image_dimension, image_dimension, 3)).astype(np.uint8)
        cv2.imwrite(evaluation_path + pose + '.png', rgb_predicted)

# Load the training data
# Have not done training / validation split
if(not os.path.exists(train_pickle_name)):
    images = os.listdir(train_image_path)
    image_list = []
    pose_list = []
    name_list = []
    for image in images:
        if(not image.endswith('.png')):
            continue
        cur_image = cv2.imread(train_image_path + image)
        cur_image = cv2.resize(cur_image, (image_dimension, image_dimension))
        cur_image = cur_image.astype(np.float32) / 255
        image_list.append(cur_image)
        image = image.strip('.png') + '.txt'
        cur_pose = array_from_file(train_pose_path + image)
        pose_list.append(cur_pose)
        name_list.append(image.strip('.txt'))
    # Make a list of rays in origin and direction representation
    ray_list = []
    positions = []
    for i in range(image_dimension):
        for k in range(image_dimension):
            positions.append([k, i])
    positions = np.asarray(positions)
    intrinsics = array_from_file(intrinsic_matrix_path)
    for i in range(len(image_list)):
        ray_list.append(ray_from_pixels(positions, intrinsics, pose_list[i]))
    rays = np.asarray(ray_list).reshape((-1, 6))
    data_dict = {'name': name_list, 'pose': pose_list, 'image': image_list, 'rays': rays}
    output_file = open(train_pickle_name, 'wb')
    pickle.dump(data_dict, output_file)
    output_file.close()
else:
    input_file = open(train_pickle_name, 'rb')
    data_dict = pickle.load(input_file)
    input_file.close()
    name_list = data_dict['name']
    pose_list = data_dict['pose']
    image_list = data_dict['image']
    rays = data_dict['rays']
num_images = len(image_list)

rgb_data = np.asarray(image_list).reshape((-1, 3))

device = "cuda" if torch.cuda.is_available() else "cpu"
implicit_function = implicit_network(6 * encoding_position, 6 * encoding_direction).to(device)
if(load_weights):
    implicit_function.load_state_dict(torch.load(implicit_weight_file))

if(evaluation_run):
    #evaluate()
    visualize_batch(rays, batch_size, near, far, num_samples, rgb_data)
    exit()

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(implicit_function.parameters(), lr = lr)

for i in range(train_steps):
    cur_indices = sample(range(rays.shape[0]), batch_size)
    cur_batch = rays[cur_indices, :]
    cur_points, cur_directions, distances = ray_batch_to_points(cur_batch,\
        near, far, num_samples, False, 0.7);
    cur_points = torch.tensor(cur_points, device = device, dtype = torch.float32)
    cur_directions = torch.tensor(cur_directions, device = device, dtype = torch.float32)
    distances = torch.tensor(distances, device = device, dtype = torch.float32)

    # Call the positional encoding and ask the implicit function about sigma and rgb
    cur_points = embed_tensor(cur_points, L = encoding_position)
    cur_directions = embed_tensor(cur_directions, L = encoding_direction)
    sigma_value, rgb_value = implicit_function(cur_points, cur_directions)

    rgb_predicted = ray_march(cur_points, cur_directions, distances, sigma_value, rgb_value, num_samples)
    # Get the picture rgb values
    cur_indices = np.asarray(cur_indices)
    rgb_picture = rgb_data[cur_indices, :]
    rgb_picture = torch.tensor(rgb_picture, device = device, dtype = torch.float32)
    optimizer.zero_grad()
    cur_loss = loss_function(rgb_predicted, rgb_picture)
    cur_loss.backward()
    optimizer.step()
    if(i % 100 == 0):
        print("Training step " + str(i) + ", loss value is currently " + str(cur_loss.item()))

torch.cuda.empty_cache()
torch.save(implicit_function.state_dict(), implicit_weight_file)
evaluate()