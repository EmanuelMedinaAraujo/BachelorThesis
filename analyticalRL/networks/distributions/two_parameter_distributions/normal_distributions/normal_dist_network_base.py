from abc import ABC, abstractmethod

import numpy as np
from torch import nn

from analyticalRL.networks.distributions.two_parameter_distributions.two_param_dist_network_base import \
    TwoParameterDistributionNetworkBase


class NormalDistributionKinematicsNetworkBase(TwoParameterDistributionNetworkBase, ABC):
    """
    This class is used to create a neural network that predicts the angles of the joints of a planar robotic arm.
    The network takes two inputs, the parameters (DH or MDH) of the arm and the goal position.
    The network uses a linear stack of layers with ReLU activation functions.

    The output of the network is the parameter for a normal distribution for each joint.
    The loss function is the mean of the distances between the end effector positions of the parameters and the goal.
    """

    @abstractmethod
    def __init__(self, num_joints, num_layer, layer_sizes, logger):
        super().__init__(num_joints, num_layer, layer_sizes, logger)

    def create_layer_stack_list(self, layer_sizes, num_joints, num_layer):
        stack_list = super().create_layer_stack_list(layer_sizes, num_joints, num_layer)
        stack_list.append(nn.Softmax(dim=-1))
        return stack_list

    def forward(self, model_input):
        network_output = super().forward(model_input)
        param, goal = model_input
        is_single_parameter = True if param.dim() == 2 else False

        all_distributions = None
        for joint_number in range(self.num_joints):
            index = 2 * joint_number

            if is_single_parameter:
                mu_output = network_output[index]
                sigma_output = network_output[index + 1]
            else:
                mu_output = network_output[:, index]
                sigma_output = network_output[:, index + 1]

            # Map mu from [0,1] to [-pi,pi]
            mu = ((mu_output * 2) - 1) * np.pi

            # Map sigma to positive values from [0,1] to [1,2]
            sigma = sigma_output + 1
            sigma = sigma.clamp(min=1e-6)

            # Ensure mu and sigma have to correct shape
            mu = mu.unsqueeze(-1) if mu.dim() == 1 else mu
            sigma = sigma.unsqueeze(-1) if sigma.dim() == 1 else sigma

            all_distributions = super().cat_distribution(all_distributions, is_single_parameter, mu, param, sigma)

        return all_distributions
