from torch import nn
import torch

from Util.forward_kinematics import calculate_distance


class KinematicsNetwork(nn.Module):
    def __init__(self, num_joints, num_layer, layer_sizes):
        super().__init__()
        self.num_joints = num_joints
        self.flatten = nn.Flatten()
        stack_list = [nn.Linear(num_joints * 3 + 2, layer_sizes[0]),nn.ReLU()]
        for i in range((num_layer-2)):
            stack_list.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            stack_list.append(nn.ReLU())
        stack_list.append(nn.Linear(layer_sizes[-1], num_joints*2))
        self.linear_relu_stack = nn.Sequential(*stack_list)
        print(self.linear_relu_stack)

    def forward(self, model_input):
        param, goal = model_input

        # Flatten the param
        flatten_param = self.flatten(param)

        # Concatenate flatten_param and goal along the second dimension
        flatten_input = torch.cat((flatten_param, goal), dim=1)

        angles = self.linear_relu_stack(flatten_input)

        all_angles = None
        for joint_number in range(self.num_joints):
            index = 2*joint_number
            sin_x = angles[:, index]
            cos_y = angles[:, index+1]
            angle = torch.atan2(sin_x, cos_y).unsqueeze(dim=-1)
            if all_angles is None:
                all_angles = angle
            else:
                all_angles = torch.cat([all_angles, angle], dim=1).to(param.device)

        return all_angles

def loss_fn(param, goal):
    distances = calculate_distance(param, goal)
    return distances.mean()
