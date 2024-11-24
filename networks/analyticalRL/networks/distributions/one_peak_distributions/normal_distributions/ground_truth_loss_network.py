import torch

from networks.analyticalRL.networks.distributions.one_peak_distributions.normal_distributions.network_base import \
    NormalDistrNetworkBase


class NormalDistrGroundTruthLossNetwork(NormalDistrNetworkBase):
    """
    This class is used to create a neural network that predicts the angles of the joints of a planar robotic arm.
    The network takes two inputs, the parameters (DH or MDH) of the arm and the goal position.
    The network uses a linear stack of layers with ReLU activation functions.

    The output of the network is the parameter for a normal distribution for each joint.
    The loss function is the mean of the distances between the end effector positions of the parameters and the goal.
    """

    def __init__(self, num_joints, num_layer, layer_sizes, logger, error_tolerance):
        super().__init__(num_joints, num_layer, layer_sizes, logger, error_tolerance)

    @staticmethod
    def extract_loss_variable_from_parameters(mu, sigma, ground_truth, is_single_parameter, joint_number):
        normal_dist = torch.distributions.Normal(loc=mu, scale=sigma)

        # Ground truth is the angle from which the goal was generated
        value = ground_truth[joint_number] if is_single_parameter else ground_truth[:, joint_number]

        # Calculate probability of ground truth using normal distribution with mu and sigma
        # expected_truth_prob = torch.exp(normal_dist.log_prob(value))
        # return torch.exp(-expected_truth_prob)
        return -normal_dist.log_prob(value)

    def calculate_batch_loss(self, all_loss_variables, goal, param):
        return all_loss_variables.mean(dim=0).mean() , torch.le(all_loss_variables, self.error_tolerance).int().sum().item()