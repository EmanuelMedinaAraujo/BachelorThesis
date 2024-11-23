import numpy as np
import torch
from torch import nn

from analyticalRL.networks.distributions.one_peak_distributions.two_param_dist_network_base import \
    TwoParameterDistrNetworkBase


class BetaDistrRSampleMeanNetwork(TwoParameterDistrNetworkBase):

    def __init__(self, num_joints, num_layer, layer_sizes, logger, error_tolerance):
        super().__init__(num_joints, num_layer, layer_sizes, logger, error_tolerance)

    def create_layer_stack_list(self, layer_sizes, num_joints, num_layer):
        stack_list = super().create_layer_stack_list(layer_sizes, num_joints, num_layer,2)
        stack_list.append(nn.ReLU())
        return stack_list

    @staticmethod
    def extract_loss_variable_from_parameters(p, q, ground_truth, is_single_parameter, joint_number):
        beta_dist = torch.distributions.Beta(p, q)
        angle = beta_dist.rsample(torch.Size([1000])).mean(dim=0)
        # Map angle from [0,1] to [-pi, pi]
        return (2 * angle - 1) * np.pi

    @staticmethod
    def map_two_parameters(parameter1, parameter2):
        epsilon = 1e-2
        p = parameter1 + epsilon
        q = parameter2 + epsilon
        return p, q
