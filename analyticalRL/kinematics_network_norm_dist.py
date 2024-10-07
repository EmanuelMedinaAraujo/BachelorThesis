import numpy as np
import torch
from torch import Tensor, nn

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
        stack_list.append(nn.Softmax())
        return stack_list


    @staticmethod
    def calc_distances(param, pred, goal):
        raise NotImplementedError("calc_distances is not implemented for KinematicsNetworkNormDist")


    def loss_fn(self, param, pred: Tensor, goal, ground_truth):
        is_single_parameter = True if param.dim() == 2 else False

        all_prob_losses = None
        for joint_number in range(self.num_joints):
            index = 2 * joint_number

            if is_single_parameter:
                mu_output = pred[index]
                sigma_output = pred[index + 1]
            else:
                mu_output = pred[:, index]
                sigma_output = pred[:, index + 1]

            # Map sigma to positive values from [0,1] to [1,2]
            sigma = sigma_output+ 1

            # Map mu from [0,1] to [-pi,pi]
            mu = ((mu_output*2)-1) * np.pi

            # Ensure mu and sigma are broadcasted correctly
            mu = mu.unsqueeze(-1) if mu.dim() == 1 else mu
            sigma = sigma.unsqueeze(-1) if sigma.dim() == 1 else sigma

            # Calculate probability of ground truth using normal distribution with mu and sigma
            value = ground_truth[joint_number] if is_single_parameter else ground_truth[:, joint_number]
            normal_dist = torch.distributions.Normal(loc=mu, scale=sigma)

            expected_truth_prob = torch.exp(normal_dist.log_prob(value))
            # Clip the probabilities to avoid numerical instability with very small values
            expected_truth_prob = torch.clamp(expected_truth_prob, min=1e-3)

            prob_loss = torch.exp(-expected_truth_prob)
            if all_prob_losses is None:
                all_prob_losses = prob_loss
            else:
                if is_single_parameter:
                    all_prob_losses = torch.cat((all_prob_losses, prob_loss), dim=0)
                else:
                    all_prob_losses = torch.cat((all_prob_losses, prob_loss), dim=1)

        return all_prob_losses.mean(dim=0).mean()
