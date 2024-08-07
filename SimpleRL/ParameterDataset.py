import torch
from torch.utils.data import Dataset
from math import pi
from Util import dh_conventions
from DataGeneration.ParameterGenerator import ParameterGenerator
from Util.forward_kinematics import forward_kinematics



class CustomParameterDataset(Dataset):
    def __init__(self, length=10000, device_to_use = None, num_of_joints = 2):
        self.dh_parameters = []
        self.goal = []
        generator = ParameterGenerator(num_joints = num_of_joints, amount_parameters_to_generate = length, device = device_to_use)
        self.dh_parameters = generator.get_random_dh_parameters()
        self.goal = generate_achievable_goal(self.dh_parameters, device_to_use)
        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.dh_parameters)

    def __getitem__(self, idx):
        param = self.dh_parameters[idx]
        goal = self.goal[idx]
        # if self.transform:
        #     input = self.transform(param, goal)
        # if self.target_transform:
        #     target = self.target_transform(target)
        return param, goal

def generate_achievable_goal( dh_parameter: torch.Tensor, device_to_use):
    # Create random theta values from -2Π top 2Π
    theta_values = (-2*pi - 2*pi ) * torch.rand(dh_parameter.shape[0], dh_parameter.shape[1]) + 2*pi
    theta_values = theta_values.unsqueeze(-1)

    # Ensure that the same device is used
    theta_values=theta_values.to(device_to_use)
    dh_parameter = dh_parameter.to(device_to_use)

    # Add theta values to parameter to obtain valid parameter
    valid_parameter = torch.cat((dh_parameter, theta_values), dim=-1).to(device_to_use)

    fk = forward_kinematics(dh_conventions.dh_to_homogeneous(valid_parameter))

    # Sum of second column of DH parameters
    return fk.get_matrix()[..., :2,3].to(device_to_use)
