from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from networks.analyticalRL.networks.kinematics_network_base_class import KinematicsNetworkBase


class NormalizeWeightsLayer(nn.Module):
    def __init__(self, num_joints):
        super(NormalizeWeightsLayer, self).__init__()
        self.num_joints = num_joints

    def forward(self, x):
        is_single_parameter = True if x.dim() == 1 else False

        if is_single_parameter:
            structured_input_view = x.view(-1, self.num_joints, 4)
            # Do Softmax on weights (every fourth row) of the input, because they should sum up to 1
            softmax_weights = torch.softmax(structured_input_view[:, :, 3], dim=1)
            return torch.cat([structured_input_view[:, :, :3], softmax_weights.unsqueeze(-1)], dim=-1).flatten()
        else:
            structured_input_view = x.view(-1, self.num_joints, 2, 4)
            # Do Softmax on weights (every fourth row) of the input, because they should sum up to 1
            softmax_weights = torch.softmax(structured_input_view[:, :, :, 3], dim=2)
            return torch.cat([structured_input_view[:, :, :, :3], softmax_weights.unsqueeze(-1)], dim=-1).flatten(
                start_dim=1)


class TwoPeakNormalDistrNetworkBase(KinematicsNetworkBase, ABC):

    @abstractmethod
    def __init__(self, num_joints, num_layer, layer_sizes, logger, error_tolerance):
        super().__init__(num_joints, num_layer, layer_sizes, logger, error_tolerance, output_per_joint=8)

    @staticmethod
    def map_eight_parameters_ranges(parameter1, parameter2, parameter3, parameter4, parameter5, parameter6, parameter7,
                                    parameter8):
        # Use atan2 to calculate angle
        mu1 = torch.atan2(parameter1, parameter2)
        # Map sigma to positive values from [0,1] to [0,2]
        sigma1 = parameter3 * 2
        sigma1 = sigma1.clamp(min=1e-6)

        mu2 = torch.atan2(parameter5, parameter6)
        sigma2 = parameter7 * 2
        sigma2 = sigma2.clamp(min=1e-6)

        # Keep weights unchanged
        # Ensure the parameters have to correct shape
        mu1 = mu1.unsqueeze(-1) if mu1.dim() == 1 else mu1
        sigma1 = sigma1.unsqueeze(-1) if sigma1.dim() == 1 else sigma1
        weight1 = parameter4.unsqueeze(-1) if parameter4.dim() == 1 else parameter4
        mu2 = mu2.unsqueeze(-1) if mu2.dim() == 1 else mu2
        sigma2 = sigma2.unsqueeze(-1) if sigma2.dim() == 1 else sigma2
        weight2 = parameter8.unsqueeze(-1) if parameter8.dim() == 1 else parameter8

        return mu1, sigma1, weight1, mu2, sigma2, weight2

    @staticmethod
    def sample_component(mu1, mu2, sigma1, sigma2, weight1, weight2, batch_size=1):
        # Sample from categorical distribution to choose the component
        weights = torch.cat([weight1.unsqueeze(dim=-1), weight2.unsqueeze(dim=-1)], dim=1)
        cat_dist = torch.distributions.Categorical(probs=weights)
        component = cat_dist.sample(torch.Size([batch_size]))  # Sample which component to use

        # Use the chosen component to select mu and sigma
        mu = mu1 * (component == 0).float() + mu2 * (component == 1).float()
        sigma = sigma1 * (component == 0).float() + sigma2 * (component == 1).float()
        return mu.squeeze(), sigma.squeeze()

    @staticmethod
    def extract_loss_variable_from_parameters(mu1, sigma1, weight1, mu2, sigma2, weight2):
        mu, sigma = TwoPeakNormalDistrNetworkBase.sample_component(mu1, mu2, sigma1, sigma2, weight1, weight2)
        with torch.no_grad():
            noise = torch.randn(mu.size()).to(mu1.device)

        # Reparameterized sampling
        samples = mu + sigma * noise
        return samples

    def calculate_batch_loss(self, all_loss_variables, goal, param):
        distances = self.calc_distances(param=param, angles_pred=all_loss_variables.squeeze(),
                                        goal=goal)  # type: Tensor
        return distances.mean(), torch.le(distances, self.error_tolerance).int().sum().item()

    def loss_fn(self, param, pred: Tensor, goal, ground_truth):
        is_single_parameter = True if param.dim() == 2 else False

        all_loss_variables = None
        for joint_number in range(self.num_joints):
            mu1, sigma1, weight1, mu2, sigma2, weight2 = self.extract_six_dist_parameters(is_single_parameter,
                                                                                          joint_number, pred)

            loss_variable = self.extract_loss_variable_from_parameters(mu1, sigma1, weight1, mu2, sigma2,
                                                                       weight2).unsqueeze(-1)
            if all_loss_variables is None:
                all_loss_variables = loss_variable
            else:
                if is_single_parameter:
                    all_loss_variables = torch.cat([all_loss_variables, loss_variable]).to(param.device)
                else:
                    all_loss_variables = torch.cat([all_loss_variables, loss_variable], dim=1).to(param.device)
        return self.calculate_batch_loss(all_loss_variables, goal, param)

    @staticmethod
    def extract_six_dist_parameters(is_single_parameter, joint_number, pred):
        if is_single_parameter:
            distribution_params = pred[joint_number]
            parameter1 = torch.tensor([distribution_params[0]]).to(pred.device)
            parameter2 = torch.tensor([distribution_params[1]]).to(pred.device)
            parameter3 = torch.tensor([distribution_params[2]]).to(pred.device)
            parameter4 = torch.tensor([distribution_params[3]]).to(pred.device)
            parameter5 = torch.tensor([distribution_params[4]]).to(pred.device)
            parameter6 = torch.tensor([distribution_params[5]]).to(pred.device)
        else:
            distribution_params = pred[:, joint_number]
            parameter1 = distribution_params[:, 0]
            parameter2 = distribution_params[:, 1]
            parameter3 = distribution_params[:, 2]
            parameter4 = distribution_params[:, 3]
            parameter5 = distribution_params[:, 4]
            parameter6 = distribution_params[:, 5]
        return parameter1, parameter2, parameter3, parameter4, parameter5, parameter6

    def collect_distributions(self, is_single_parameter, network_output):
        all_distributions = None
        for joint_number in range(self.num_joints):
            index = 8 * joint_number

            if is_single_parameter:
                parameter1 = network_output[index]
                parameter2 = network_output[index + 1]
                parameter3 = network_output[index + 2]
                parameter4 = network_output[index + 3]
                parameter5 = network_output[index + 4]
                parameter6 = network_output[index + 5]
                parameter7 = network_output[index + 6]
                parameter8 = network_output[index + 7]
            else:
                parameter1 = network_output[:, index]
                parameter2 = network_output[:, index + 1]
                parameter3 = network_output[:, index + 2]
                parameter4 = network_output[:, index + 3]
                parameter5 = network_output[:, index + 4]
                parameter6 = network_output[:, index + 5]
                parameter7 = network_output[:, index + 6]
                parameter8 = network_output[:, index + 7]

            (mu1, sigma1, weight1, mu2, sigma2, weight2) = self.map_eight_parameters_ranges(parameter1,
                                                                                            parameter2,
                                                                                            parameter3,
                                                                                            parameter4,
                                                                                            parameter5,
                                                                                            parameter6,
                                                                                            parameter7,
                                                                                            parameter8)

            if is_single_parameter:
                distribution = torch.cat([mu1.unsqueeze(-1), sigma1.unsqueeze(-1), weight1.unsqueeze(-1),
                                          mu2.unsqueeze(-1), sigma2.unsqueeze(-1), weight2.unsqueeze(-1),
                                          ])
            else:
                distribution = torch.cat(
                    [mu1, sigma1, weight1, mu2, sigma2, weight2],
                    dim=-1)
            if all_distributions is None:
                all_distributions = distribution.unsqueeze(0 if is_single_parameter else 1)
            else:
                if is_single_parameter:
                    all_distributions = torch.cat([all_distributions, distribution.unsqueeze(0)]).to(network_output.device)
                else:
                    all_distributions = torch.cat([all_distributions, distribution.unsqueeze(1)], dim=1).to(
                        network_output.device)
        return all_distributions
