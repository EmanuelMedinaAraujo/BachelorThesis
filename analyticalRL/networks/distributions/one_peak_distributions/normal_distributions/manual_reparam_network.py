import torch

from analyticalRL.networks.distributions.one_peak_distributions.normal_distributions.network_base import \
    NormalDistrNetworkBase


class NormalDistrManualReparameterizationNetwork(NormalDistrNetworkBase):
    """
    This class is used to create a neural network that predicts the angles of the joints of a planar robotic arm.
    The network takes two inputs, the parameters (DH or MDH) of the arm and the goal position.
    The network uses a linear stack of layers with ReLU activation functions.

    The output of the network is the parameter for a normal distribution for each joint.
    The loss function is the mean of the distances between the end effector positions of the parameters and the goal.
    """

    def __init__(self, num_joints, num_layer, layer_sizes, logger):
        super().__init__(num_joints, num_layer, layer_sizes, logger)

    @staticmethod
    def extract_loss_variable_from_parameters(mu, sigma, ground_truth, is_single_parameter, joint_number):
        normal_dist = torch.distributions.Normal(loc=0., scale=1.)
        # Return angle as loss variable
        return mu + normal_dist.sample() * sigma
