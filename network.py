import torch
import numpy as np
    
# pos_dim is the number of dimensions for the position
# dir_dim is the number of dimensions for the direction
# These depend on the embedding selected
class implicit_network(torch.nn.Module):
    def __init__(self, pos_dim, dir_dim, depth = 8, feature_dim = 256):
        super().__init__()
        layers = []
        prev_dim = pos_dim
        # Right now we write a skip from the beginning to the 5th layer
        for i in range(4):
            layers.append(torch.nn.Linear(prev_dim, feature_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = feature_dim
        self.part_1 = torch.nn.Sequential(*layers)
        prev_dim = pos_dim + feature_dim
        layers = []
        for i in range(4, depth):
            layers.append(torch.nn.Linear(prev_dim, feature_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = feature_dim
        self.part_2 = torch.nn.Sequential(*layers)
        self.sigma_output = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 1),
            #torch.nn.ReLU()
        )
        #self.intermediate = torch.nn.Sequential(
        #    torch.nn.Linear(feature_dim, feature_dim),
        #    torch.nn.ReLU()
        #)
        self.intermediate = torch.nn.Linear(feature_dim, feature_dim)
        self.rgb_out = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + dir_dim, feature_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_dim // 2, 3)
        )
    
    def forward(self, pos_input, dir_input):
        # Get the sigma output
        tmp = self.part_1(pos_input)
        tmp = torch.cat([tmp, pos_input], dim = 1)
        tmp = self.part_2(tmp)
        sigma_value = self.sigma_output(tmp)
        sigma_value = torch.sigmoid(sigma_value)
        tmp = self.intermediate(tmp)
        tmp = torch.cat([tmp, dir_input], dim = 1)
        rgb_value = self.rgb_out(tmp)
        rgb_value = torch.sigmoid(rgb_value)
        return sigma_value, rgb_value
    
def embed_tensor(input_tensor, L = 10):
    tensor_list = [input_tensor]
    for i in range(L):
        cur_tensor = torch.sin((2 ** i) * np.pi * input_tensor)
        tensor_list.append(cur_tensor)
        cur_tensor = torch.cos((2 ** i) * np.pi * input_tensor)
        tensor_list.append(cur_tensor)
    return torch.cat(tensor_list, dim = 1)