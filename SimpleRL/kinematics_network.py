import numpy as np
from torch import nn
import torch

from Util.forward_kinematics import calculate_distance


class KinematicsNetwork(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_joints * 3 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Softmax(dim=1),
            nn.Linear(64, num_joints*2)
        )

    def forward(self, model_input):
        param, goal = model_input

        # Flatten the param
        flatten_param = self.flatten(param)

        # Concatenate flatten_param and goal along the second dimension
        flatten_input = torch.cat((flatten_param, goal), dim=1)

        angles = self.linear_relu_stack(flatten_input)

        sin_x = angles[:, 0]
        cos_y = angles[:, 1]
        first_angle= torch.atan2(sin_x, cos_y).unsqueeze(dim=-1)

        sin_x = angles[:, 2]
        cos_y = angles[:, 3]
        second_angle= torch.atan2(sin_x, cos_y).unsqueeze(dim=-1)
        return torch.cat([first_angle, second_angle], dim=1).to(param.device)

def loss_fn(param, goal):
    distances = calculate_distance(param, goal)
    return distances.mean()
