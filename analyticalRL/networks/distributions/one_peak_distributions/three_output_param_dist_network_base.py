from abc import ABC, abstractmethod

import torch
from torch import Tensor

from analyticalRL.networks.kinematics_network_base_class import KinematicsNetworkBase


class ThreeOutputParameterDistrNetworkBase(KinematicsNetworkBase, ABC):

    @abstractmethod
    def __init__(self, num_joints, num_layer, layer_sizes, logger, error_tolerance):
        super().__init__(num_joints, num_layer, layer_sizes, logger, error_tolerance, output_per_joint=3)

    @staticmethod
    def extract_two_dist_parameters(is_single_parameter, joint_number, pred):
        if is_single_parameter:
            distribution_params = pred[joint_number]
            parameter1 = distribution_params[0].unsqueeze(-1)
            parameter2 = distribution_params[1].unsqueeze(-1)
        else:
            distribution_params = pred[:, joint_number]
            parameter1 = distribution_params[:, 0].unsqueeze(-1)
            parameter2 = distribution_params[:, 1].unsqueeze(-1)
        return parameter1, parameter2

    @staticmethod
    @abstractmethod
    def map_three_parameters(parameter1, parameter2, parameter3):
        pass

    @staticmethod
    @abstractmethod
    def extract_loss_variable_from_parameters(mu, sigma, ground_truth, is_single_parameter, joint_number):
        pass

    def calculate_batch_loss(self, all_loss_variables, goal, param):
        distances = self.calc_distances(param=param, angles_pred=all_loss_variables.squeeze(), goal=goal)
        return distances.mean(), torch.le(distances, self.error_tolerance).int().sum().item()

    def loss_fn(self, param, pred: Tensor, goal, ground_truth):
        is_single_parameter = True if param.dim() == 2 else False

        all_loss_variables = None
        for joint_number in range(self.num_joints):
            mu, sigma = self.extract_two_dist_parameters(is_single_parameter, joint_number, pred)

            loss_variable = self.extract_loss_variable_from_parameters(mu, sigma, ground_truth, is_single_parameter,
                                                                       joint_number).unsqueeze(-1)
            if all_loss_variables is None:
                all_loss_variables = loss_variable
            else:
                if is_single_parameter:
                    all_loss_variables = torch.cat([all_loss_variables, loss_variable]).to(param.device)
                else:
                    all_loss_variables = torch.cat([all_loss_variables, loss_variable], dim=1).to(param.device)
        return self.calculate_batch_loss(all_loss_variables, goal, param)

    def forward(self, model_input):
        network_output = super().forward(model_input)
        param, goal = model_input
        is_single_parameter = True if param.dim() == 2 else False

        all_distributions = None
        for joint_number in range(self.num_joints):
            index = 2 * joint_number

            if is_single_parameter:
                parameter1 = network_output[index]
                parameter2 = network_output[index + 1]
                parameter3 = network_output[index + 2]
            else:
                parameter1 = network_output[:, index]
                parameter2 = network_output[:, index + 1]
                parameter3 = network_output[:, index + 2]

            parameter1, parameter2 = self.map_three_parameters(parameter1, parameter2, parameter3)

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
