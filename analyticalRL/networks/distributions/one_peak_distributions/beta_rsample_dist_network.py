import numpy as np
import torch
from torch import nn

from analyticalRL.networks.distributions.one_peak_distributions.three_output_param_dist_network_base import \
    ThreeOutputParameterDistrNetworkBase


class BetaDistrRSampleMeanNetwork(ThreeOutputParameterDistrNetworkBase):

    @staticmethod
    def map_three_parameters(parameter1, parameter2, parameter3):
        pass

    def __init__(self, num_joints, num_layer, layer_sizes, logger, error_tolerance):
        super().__init__(num_joints, num_layer, layer_sizes, logger, error_tolerance)

    def create_layer_stack_list(self, layer_sizes, num_joints, num_layer, output_per_joint):
        stack_list = super().create_layer_stack_list(layer_sizes, num_joints, num_layer, 2)
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

    def forward(self, model_input):
        flatten_input = self.flatten_model_input(model_input)

        # Pass the flatten input through the linear_relu_stack
        network_output = self.linear_relu_stack(flatten_input)
        param, goal = model_input
        is_single_parameter = True if param.dim() == 2 else False

        all_distributions = None
        for joint_number in range(self.num_joints):
            index = 2 * joint_number

            if is_single_parameter:
                parameter1 = network_output[index]
                parameter2 = network_output[index + 1]
            else:
                parameter1 = network_output[:, index]
                parameter2 = network_output[:, index + 1]

            parameter1, parameter2 = self.map_two_parameters(parameter1, parameter2)

            # Ensure the parameters have to correct shape
            parameter1 = parameter1.unsqueeze(-1) if parameter1.dim() == 1 else parameter1
            parameter2 = parameter2.unsqueeze(-1) if parameter2.dim() == 1 else parameter2

            if is_single_parameter:
                distribution = torch.cat([parameter1.unsqueeze(-1), parameter2.unsqueeze(-1)])
            else:
                distribution = torch.cat([parameter1, parameter2], dim=-1)
            if all_distributions is None:
                all_distributions = distribution.unsqueeze(0 if is_single_parameter else 1)
            else:
                if is_single_parameter:
                    all_distributions = torch.cat([all_distributions, distribution.unsqueeze(0)]).to(param.device)
                else:
                    all_distributions = torch.cat([all_distributions, distribution.unsqueeze(1)], dim=1).to(
                        param.device)

        return all_distributions
