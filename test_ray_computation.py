import cv2
import open3d as o3d
import numpy as np

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

def create_arrow_from_vector(origin, vector):
    '''
    origin: origin of the arrow
    vector: direction of the arrow
    '''
    v = np.array(vector)
    v /= np.linalg.norm(v)
    z = np.array([0,0,1])
    angle = np.arccos(z@v)
    
    arrow = o3d.geometry.TriangleMesh.create_arrow(0.05, 0.1, 0.25, 0.2)
    arrow.paint_uniform_color([1,0,1])
    T = np.eye(4)
    T[:3, 3] = np.array(origin)
    T[:3,:3] = o3d.geometry.get_rotation_matrix_from_axis_angle(np.cross(z, v) * angle)
    arrow.transform(T)
    return arrow

intrinsic_file = 'bottles/intrinsics.txt'
extrinsic_list = ['0000', '0010', '0030', '0050']
extrinsic_list = ['bottles/pose/0_train_' + item + '.txt' for item in extrinsic_list]

def ray_from_pixels(positions, intrinsic, extrinsic):
    focal_length = intrinsic[0, 0]
    n = intrinsic[0, 2]
    camera_frame = [(positions[:,0] - n) / focal_length,
                    (n - positions[:,1]) / focal_length,
                    -np.ones(positions.shape[0])]
    camera_frame = np.stack(camera_frame, axis = 1)
    transformation = extrinsic
    camera_frame = np.concatenate([camera_frame, np.ones((positions.shape[0], 1))], axis = 1)
    world_frame = transformation @ camera_frame.T
    world_frame = world_frame.T
    camera_location = transformation[:3, 3]
    print(camera_location)
    print(world_frame)
    arrow_list = []
    for i in range(positions.shape[0]):
        arrow_list.append(create_arrow_from_vector(camera_location, world_frame[i, :3] - camera_location))
        direction_vector = world_frame[i, :3] - camera_location
        direction_vector /= np.linalg.norm(direction_vector)
    return arrow_list

W = 800
H = 800

intrinsic = array_from_file(intrinsic_file)
extrinsics = [array_from_file(item) for item in extrinsic_list]

arrow_list = []
for i in range(len(extrinsics)):
    arrow_list += ray_from_pixels(np.asarray([[0,0], [799, 0], [799, 799], [0, 799]]),
                                  intrinsic, extrinsics[i])
arrow_list.append(create_arrow_from_vector(np.zeros(3), np.asarray([0., 0., 1.])))
arrow_list.append(create_arrow_from_vector(np.zeros(3), np.asarray([0., 1., 0.])))
arrow_list.append(create_arrow_from_vector(np.zeros(3), np.asarray([0., -1., 0.])))
arrow_list.append(create_arrow_from_vector(np.zeros(3), np.asarray([1., 0., 0.])))
arrow_list.append(create_arrow_from_vector(np.zeros(3), np.asarray([-1., 0., 0.])))
o3d.visualization.draw_geometries(arrow_list)