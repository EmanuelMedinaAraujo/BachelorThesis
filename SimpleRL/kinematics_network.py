from torch import nn
import torch

from Util.forward_kinematics import update_theta_values, calculate_eef_positions


class KinematicsNetwork(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_joints * 3 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
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


def loss_fn(param, pred, goal):
    # Update theta values
    updated_param = update_theta_values(param, pred)
    eef_positions = calculate_eef_positions(updated_param)

    # Calculate the mean distance between the eef position and the goal position
    return torch.square(eef_positions - goal).sum(dim=1).sqrt().mean()
