from abc import ABC, abstractmethod

from analyticalRL.networks.kinematics_network_base_class import KinematicsNetworkBase


class TwoParameterDistributionNetworkBase(KinematicsNetworkBase, ABC):

    @abstractmethod
    def __init__(self, num_joints, num_layer, layer_sizes, logger):
        super().__init__(num_joints, num_layer, layer_sizes, logger)

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
    def cat_distribution(all_distributions, is_single_parameter, mu, param, sigma):
        if is_single_parameter:
            distribution = torch.cat([mu.unsqueeze(-1), sigma.unsqueeze(-1)])
        else:
            distribution = torch.cat([mu, sigma], dim=-1)
        if all_distributions is None:
            all_distributions = distribution.unsqueeze(0 if is_single_parameter else 1)
        else:
            if is_single_parameter:
                all_distributions = torch.cat([all_distributions, distribution.unsqueeze(0)]).to(param.device)
            else:
                all_distributions = torch.cat([all_distributions, distribution.unsqueeze(1)], dim=1).to(param.device)
        return all_distributions
