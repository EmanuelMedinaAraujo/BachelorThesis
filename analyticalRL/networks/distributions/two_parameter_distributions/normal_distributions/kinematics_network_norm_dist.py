import torch
from torch import Tensor

from normal_dist_network_base import NormalDistributionKinematicsNetworkBase


# noinspection SpellCheckingInspection
class KinematicsNetworkNormDist(NormalDistributionKinematicsNetworkBase):
    """
    This class is used to create a neural network that predicts the angles of the joints of a planar robotic arm.
    The network takes two inputs, the parameters (DH or MDH) of the arm and the goal position.
    The network uses a linear stack of layers with ReLU activation functions.

    The output of the network is the parameter for a normal distribution for each joint.
    The loss function is the mean of the distances between the end effector positions of the parameters and the goal.
    """

    def __init__(self, num_joints, num_layer, layer_sizes, logger):
        super().__init__(num_joints, num_layer, layer_sizes, logger)

    def loss_fn(self, param, pred: Tensor, goal, ground_truth):
        is_single_parameter = True if param.dim() == 2 else False

        all_prob_losses = None
        # all_angles = None
        for joint_number in range(self.num_joints):
            mu, sigma = self.extract_two_dist_parameters(is_single_parameter, joint_number, pred)

            normal_dist = torch.distributions.Normal(loc=mu, scale=sigma)

            # Calculate probability of ground truth using normal distribution with mu and sigma
            value = ground_truth[joint_number] if is_single_parameter else ground_truth[:, joint_number]
            # expected_truth_prob = torch.exp(normal_dist.log_prob(value))

            # prob_loss = torch.exp(-expected_truth_prob)
            prob_loss = -normal_dist.log_prob(value)
            if all_prob_losses is None:
                all_prob_losses = prob_loss
            else:
                if is_single_parameter:
                    all_prob_losses = torch.cat((all_prob_losses, prob_loss), dim=0)
                else:
                    all_prob_losses = torch.cat((all_prob_losses, prob_loss), dim=1)
        # Use mu as angle and distance for loss
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
