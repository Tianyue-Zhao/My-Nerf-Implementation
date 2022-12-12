## Introduction
My NeRF implementation based mostly on the original NeRF paper.
Features stratified and hierarchical sampling, written with PyTorch.
The ray sampler assume a pose matrix that follows OpenCV camera conventions. Slight modifications are needed for this to work with other camera conventions.

## Structure
train.py: Main training script with various configuration options in it
Contains the data loading, loss updates, and evaluation

ray_stuff.py: Script with various ray conversion functions, forming a ray sampler and a ray marcher
Ray sampler consists of ray_from_pixels, ray_batch_to_points, and sample_points_weighted (hierarchical sampling).
Ray marcher is the ray_march function.

visualization.py: Script with various neat visualization tools. Mainly one tool to visualize a batch of sampled rays, and another to visualize the sigma (density) portion of a trained radiance field.