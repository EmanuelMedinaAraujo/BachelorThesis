import numpy as np
import torch
from torch import Tensor, nn

from analyticalRL.kinematics_network import KinematicsNetwork
from analyticalRL.kinematics_network_base import KinematicsNetworkBase
from custom_logging.custom_loggger import GeneralLogger


class KinematicsNetworkNormDist(KinematicsNetworkBase):
    """
    This class is used to create a neural network that predicts the angles of the joints of a planar robotic arm.
    The network takes two inputs, the parameters (DH or MDH) of the arm and the goal position.
    The network uses a linear stack of layers with ReLU activation functions.

    The output of the network is the parameter for a normal distribution for each joint.
    The loss function is the mean of the distances between the end effector positions of the parameters and the goal.
    """

    def create_layer_stack_list(self, layer_sizes, num_joints, num_layer):
        stack_list = super().create_layer_stack_list(layer_sizes, num_joints, num_layer)
        stack_list.append(nn.Softmax(dim=-1))
        return stack_list

    @staticmethod
    def calc_distances(param, pred, goal):
        raise NotImplementedError("calc_distances is not implemented for KinematicsNetworkNormDist")

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

            # Map sigma to positive values from [0,1] to [1,2]
            sigma = sigma_output+1
            sigma = sigma.clamp(min=1e-6)

            # Map mu from [0,1] to [-pi,pi]
            mu = ((mu_output * 2) - 1) * np.pi

            # Ensure mu and sigma have to correct shape
            mu = mu.unsqueeze(-1) if mu.dim() == 1 else mu
            sigma = sigma.unsqueeze(-1) if sigma.dim() == 1 else sigma

            if is_single_parameter:
                distribution = torch.cat([mu.unsqueeze(-1), sigma.unsqueeze(-1)])
            else:
                distribution = torch.cat([mu, sigma], dim=-1)

            if all_distributions is None:
                all_distributions = distribution
            else:
                if is_single_parameter:
                    all_distributions = torch.cat([all_distributions.unsqueeze(0), distribution.unsqueeze(0)]).to(param.device)
                else:
                    all_distributions = torch.cat([all_distributions.unsqueeze(1), distribution.unsqueeze(1)], dim=1).to(param.device)

        return all_distributions

    def loss_fn(self, param, pred: Tensor, goal, ground_truth):
        is_single_parameter = True if param.dim() == 2 else False

        all_prob_losses = None
        all_angles = None
        for joint_number in range(self.num_joints):
            if is_single_parameter:
                distribution_params = pred[joint_number]
                mu = distribution_params[0].unsqueeze(-1)
                sigma = distribution_params[1].unsqueeze(-1)
            else:
                distribution_params = pred[:, joint_number]
                mu = distribution_params[:, 0].unsqueeze(-1)
                sigma = distribution_params[:, 1].unsqueeze(-1)

            normal_dist = torch.distributions.Normal(loc=mu, scale=sigma)

            # Calculate probability of ground truth using normal distribution with mu and sigma
            value = ground_truth[joint_number] if is_single_parameter else ground_truth[:, joint_number]
            expected_truth_prob = torch.exp(normal_dist.log_prob(value))

            prob_loss = torch.exp(-expected_truth_prob)
            if all_prob_losses is None:
                all_prob_losses = prob_loss
            else:
                if is_single_parameter:
                    all_prob_losses = torch.cat((all_prob_losses, prob_loss), dim=0)
                else:
                    all_prob_losses = torch.cat((all_prob_losses, prob_loss), dim=1)
        # Use distance for loss
        #     angle = mu.unsqueeze(-1)
        #     if all_angles is None:
        #         all_angles = angle
        #     else:
        #         if is_single_parameter:
        #             all_angles = torch.cat([all_angles, angle]).to(param.device)
        #         else:
        #             all_angles = torch.cat([all_angles, angle], dim=1).to(param.device)
        # distances = KinematicsNetwork.calc_distances(param=param, pred=all_angles.squeeze(), goal=goal)
        # return distances.mean()
        return all_prob_losses.mean(dim=0).mean()
