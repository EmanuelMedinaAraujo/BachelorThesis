from abc import ABC, abstractmethod

import numpy as np
from torch import nn

from analyticalRL.networks.distributions.one_peak_distributions.two_param_dist_network_base import \
    TwoParameterDistrNetworkBase


class NormalDistrNetworkBase(TwoParameterDistrNetworkBase, ABC):
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

    def create_layer_stack_list(self, layer_sizes, num_joints, num_layer):
        stack_list = super().create_layer_stack_list(layer_sizes, num_joints, num_layer)
        stack_list.append(nn.Sigmoid())
        return stack_list

    @staticmethod
    def map_two_parameters(parameter1, parameter2):
        # Map mu from [0,1] to [-pi,pi]
        mu = ((parameter1 * 2) - 1) * np.pi
        # Map sigma to positive values from [0,1] to [1,2]
        sigma = parameter2 + 1
        sigma = sigma.clamp(min=1e-6)
        return mu, sigma
