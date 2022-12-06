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
        self.intermediate = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.ReLU()
        )
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

class NeRF(torch.nn.Module):
  r"""
  Neural radiance fields module.
  """
  def __init__(
    self,
    d_input, d_viewdirs,
    n_layers: int = 8,
    d_filter: int = 256,
    skip: tuple = (4,),
  ):
    super().__init__()
    self.d_input = d_input
    self.skip = skip
    self.act = torch.nn.functional.relu
    self.d_viewdirs = d_viewdirs

    # Create model layers
    self.layers = torch.nn.ModuleList(
      [torch.nn.Linear(self.d_input, d_filter)] +
      [torch.nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
       else torch.nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
    )

    # Bottleneck layers
    if self.d_viewdirs is not None:
      # If using viewdirs, split alpha and RGB
      self.alpha_out = torch.nn.Linear(d_filter, 1)
      self.rgb_filters = torch.nn.Linear(d_filter, d_filter)
      self.branch = torch.nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
      self.output = torch.nn.Linear(d_filter // 2, 3)
    else:
      # If no viewdirs, use simpler output
      self.output = torch.nn.Linear(d_filter, 4)
  
  def forward(
    self,
    x: torch.Tensor,
    viewdirs: torch.Tensor
  ) -> torch.Tensor:
    r"""
    Forward pass with optional view direction.
    """

    # Cannot use viewdirs if instantiated with d_viewdirs = None
    if self.d_viewdirs is None and viewdirs is not None:
      raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

    # Apply forward pass up to bottleneck
    x_input = x
    for i, layer in enumerate(self.layers):
      x = self.act(layer(x))
      if i in self.skip:
        x = torch.cat([x, x_input], dim=-1)

    # Apply bottleneck
    if self.d_viewdirs is not None:
      # Split alpha from network output
      alpha = self.alpha_out(x)

      # Pass through bottleneck to get RGB
      x = self.rgb_filters(x)
      x = torch.concat([x, viewdirs], dim=-1)
      x = self.act(self.branch(x))
      x = self.output(x)
    else:
      # Simple output
      x = self.output(x)
    return alpha, x