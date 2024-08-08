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
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, model_input):
        param, goal = model_input

        # Flatten the param
        flatten_param = self.flatten(param)

        # Concatenate flatten_param and goal along the second dimension
        flatten_input = torch.cat((flatten_param, goal), dim=1)

        logits = self.linear_relu_stack(flatten_input)
        return logits


def loss_fn(param, goal):
    distances = calculate_distance(param, goal)
    return distances.mean()
