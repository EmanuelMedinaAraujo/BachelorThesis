import torch

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

    def __init__(self, num_joints, num_layer, layer_sizes, logger: GeneralLogger, distribution_tolerance):
        """
        Initializes the KinematicsNetwork.

        Args:
            num_joints: The number of joints in the planar robotic arm
            num_layer: The number of layers in the network
            layer_sizes: A list containing the number of neurons in each layer
        """
        super().__init__(num_joints, num_layer, layer_sizes, logger)
        self.distribution_tolerance = distribution_tolerance

    @staticmethod
    def calc_distances(param, pred, goal):
        raise NotImplementedError("calc_distances is not implemented for KinematicsNetworkNormDist")

    def loss_fn(self, param, pred, goal, ground_truth):
        is_single_parameter = True if param.dim() == 2 else False

        all_prob_losses = None
        all_mu = None
        all_values = None
        for joint_number in range(self.num_joints):
            index = 2 * joint_number
            if is_single_parameter:
                mu = pred[index]
                sigma = pred[index + 1]
            else:
                mu = pred[:, index]
                sigma = pred[:, index + 1]

            # Map sigma to positive values from [-inf,inf] to [0,2]
            sigma = ( torch.sigmoid(sigma)+0.4)

            # Map mu from [-inf,inf] to [-pi,pi]
            mu = ((2 * torch.softmax(mu, dim=0) - 1) * torch.pi).unsqueeze(-1)

            # Calculate probability of ground truth using normal distribution with mu and sigma
            value = ground_truth[joint_number] if is_single_parameter else ground_truth[:, joint_number]
            normal_dist = torch.distributions.Normal(mu, sigma)

            # expected_truth_prob = normal_dist.cdf(value + self.distribution_tolerance) - normal_dist.cdf(
            #     value - self.distribution_tolerance)
            # expected_truth_prob = torch.abs(expected_truth_prob)
            expected_truth_prob = torch.exp(normal_dist.log_prob(value))

            # Calculate the expected probability
            # pred_prob = normal_dist.cdf(mu + self.distribution_tolerance) - normal_dist.cdf(
            #     mu - self.distribution_tolerance)
            # pred_prob = torch.abs(pred_prob)
            pred_prob = torch.exp(normal_dist.log_prob(mu))

            prob_loss = torch.exp( -expected_truth_prob)
            prob_loss = prob_loss.unsqueeze(-1)
            if all_prob_losses is None:
                all_prob_losses = prob_loss
            else:
                if is_single_parameter:
                    all_prob_losses = torch.cat([all_prob_losses, prob_loss]).to(param.device)
                else:
                    all_prob_losses = torch.cat([all_prob_losses, prob_loss], dim=1).to(param.device)

            # if all_mu is None:
            #     all_mu = mu
            # else:
            #     if is_single_parameter:
            #         all_mu = torch.cat([all_mu, mu]).to(param.device)
            #     else:
            #         all_mu = torch.cat([all_mu, mu], dim=-1).to(param.device)
            # if all_values is None:
            #     all_values = value
            # else:
            #     if is_single_parameter:
            #         all_values = torch.cat([all_values, value]).to(param.device)
            #     else:
            #         all_values = torch.cat([all_values, value], dim=-1).to(param.device)

        return all_prob_losses.mean() if is_single_parameter else all_prob_losses.mean(dim=0).mean()
        # return -abs(all_values - all_mu).mean()
