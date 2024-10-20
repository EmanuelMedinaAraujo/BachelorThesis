import torch
from torch import Tensor

from analyticalRL.networks.simple_kinematics_network import SimpleKinematicsNetwork
from custom_logging.custom_loggger import GeneralLogger
from normal_dist_network_base import NormalDistributionKinematicsNetworkBase


class KinematicsNetworkRandomSampleDist(NormalDistributionKinematicsNetworkBase):
    """
    This class is used to create a neural network that predicts the angles of the joints of a planar robotic arm.
    The network takes two inputs, the parameters (DH or MDH) of the arm and the goal position.
    The network uses a linear stack of layers with ReLU activation functions.

    The output of the network is the parameter for a normal distribution for each joint.
    The loss function is the mean of the distances between the end effector positions of the parameters and the goal.
    """

    def __init__(self, num_joints, num_layer, layer_sizes, logger: GeneralLogger):
        super().__init__(num_joints, num_layer, layer_sizes, logger)

    def loss_fn(self, param, pred: Tensor, goal, ground_truth):
        is_single_parameter = True if param.dim() == 2 else False

        all_angles = None
        for joint_number in range(self.num_joints):
            mu, sigma = self.extract_dist_parameters(is_single_parameter, joint_number, pred)

            normal_dist = torch.distributions.Normal(loc=mu, scale=sigma)
            angle = normal_dist.rsample().unsqueeze(-1)

            if all_angles is None:
                all_angles = angle
            else:
                if is_single_parameter:
                    all_angles = torch.cat([all_angles, angle]).to(param.device)
                else:
                    all_angles = torch.cat([all_angles, angle], dim=1).to(param.device)
        # Use distance for loss
        distances = SimpleKinematicsNetwork.calc_distances(param=param, angles_pred=all_angles.squeeze(), goal=goal)
        return distances.mean()
