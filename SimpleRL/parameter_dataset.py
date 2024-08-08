import torch
from torch.utils.data import Dataset
from math import pi
from Util import dh_conventions
from DataGeneration.parameter_generator import ParameterGenerator
from Util.forward_kinematics import forward_kinematics


class CustomParameterDataset(Dataset):
    def __init__(self, length=10000,
                 device_to_use=None,
                 num_of_joints=2,
                 parameter_convention='DH',
                 tensor_type=torch.float32,
                 min_len=0.1,
                 max_len=20.):
        self.dh_parameters = []
        self.goal = []
        generator = ParameterGenerator(batch_size=length,
                                       device=device_to_use,
                                       tensor_type=tensor_type,
                                       num_joints=num_of_joints,
                                       parameter_convention=parameter_convention,
                                       min_len=min_len,
                                       max_len=max_len
                                       )
        self.dh_parameters = generator.get_random_dh_parameters()
        self.goal = generate_achievable_goal(self.dh_parameters, device_to_use)

    def __len__(self):
        return len(self.dh_parameters)

    def __getitem__(self, idx):
        return self.dh_parameters[idx], self.goal[idx]


def generate_achievable_goal(dh_parameter: torch.Tensor, device_to_use):
    # Create random theta values from -2Π top 2Π
    theta_values = (-2 * pi - 2 * pi) * torch.rand(dh_parameter.shape[0], dh_parameter.shape[1]) + 2 * pi
    theta_values = theta_values.unsqueeze(-1)

    # Ensure that the same device is used
    theta_values = theta_values.to(device_to_use)
    dh_parameter = dh_parameter.to(device_to_use)

    # Add theta values to parameter to obtain valid parameter
    valid_parameter = torch.cat((dh_parameter, theta_values), dim=-1).to(device_to_use)

    fk = forward_kinematics(dh_conventions.dh_to_homogeneous(valid_parameter))

    # Sum of second column of DH parameters
    return fk.get_matrix()[..., :2, 3].to(device_to_use)
