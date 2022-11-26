import cv2
import os
import torch
import pickle
import numpy as np
from network import embed_tensor, implicit_network
from ray_stuff import ray_from_pixels, 
from random import shuffle

# Configuration variables
image_dimensions = (200,200)
downsample = True

near = 1.5
far = 4.5
num_samples = 300
# Hierarichical sampling comes later

train_pickle_name = 'train_information.data'
train_image_path = 'bottles/rgb/'
train_pose_path = 'bottles/pose/'
intrinsic_matrix_path = 'bottles/intrinsics.txt'

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

# Load the training data
# Have not done training / validation split
if(not os.path.exists(train_pickle_name)):
    images = os.listdir(train_image_path)
    image_list = []
    pose_list = []
    name_list = []
    for image in images:
        image = str(image)
        cur_image = cv2.imread(train_image_path + image)
        cur_image = cur_image.astype(np.float32) / 255
        image_list.append(cur_image)
        image = image.strip('.png') + '.txt'
        cur_pose = array_from_file(train_pose_path + image)
        pose_list.append(cur_pose)
        name_list.append(image.strip('.txt'))
    data_dict = {'name': name_list, 'pose': pose_list, 'image': image_list}
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
num_images = len(image_list)
intrinsics = array_from_file(intrinsic_matrix_path)

# Make a list of rays in origin and direction representation
ray_list = []
positions = [[[i, k] for k in range(image_dimensions[0])] for i in range(image_dimensions[0])]
positions = np.asarray(positions)
for i in range(num_images):
    ray_list.append(ray_from_pixels(positions, intrinsics, pose_list[i]))
shuffle(ray_list)