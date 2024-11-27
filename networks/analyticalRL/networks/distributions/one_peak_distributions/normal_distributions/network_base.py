from abc import ABC, abstractmethod

import torch
from torch import nn

from networks.analyticalRL.networks.distributions.one_peak_distributions.three_output_param_dist_network_base import \
    ThreeOutputParameterDistrNetworkBase


class NormalizeSigmaLayer(nn.Module):
    def __init__(self, num_joints, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_joints = num_joints

    def forward(self, x):
        self.num_joints = 1
        is_single_parameter = True if x.dim() == 1 else False

        if is_single_parameter:
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[2::3] = True
            # Apply sigmoid to every third entry
            x[mask] = torch.sigmoid(x[mask])
            return x
        else:
            # Create a mask to select every third entry in each batch
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[:, 2::3] = True

            # Apply sigmoid to every third entry
            x[mask] = torch.sigmoid(x[mask])
            return x


class NormalDistrNetworkBase(ThreeOutputParameterDistrNetworkBase, ABC):
    """
    This class is used to create a neural network that predicts the angles of the joints of a planar robotic arm.
    The network takes two inputs, the parameters (DH or MDH) of the arm and the goal position.
    The network uses a linear stack of layers with ReLU activation functions.

    The output of the network is the parameter for a normal distribution for each joint.
    The loss function is the mean of the distances between the end effector positions of the parameters and the goal.
    """

    @abstractmethod
    def __init__(self, num_joints, num_layer, layer_sizes, logger, error_tolerance):
        super().__init__(num_joints, num_layer, layer_sizes, logger, error_tolerance)

    def create_layer_stack_list(self, layer_sizes, num_joints, num_layer, output_per_joint):
        stack_list = super().create_layer_stack_list(layer_sizes, num_joints, num_layer, output_per_joint)
        # Only apply sigmoid on parameter3 (sigma)
        stack_list.append(NormalizeSigmaLayer(num_joints))
        return stack_list

    @staticmethod
    def map_three_parameters(parameter1, parameter2, parameter3):
        # Use atan2 to calculate angle
        mu = torch.atan2(parameter1, parameter2)
        # Map sigma to positive values from [0,1] to [0,1]
        sigma = parameter3 * 0.5
        sigma = sigma.clamp(min=1e-6)
        return mu, sigma
