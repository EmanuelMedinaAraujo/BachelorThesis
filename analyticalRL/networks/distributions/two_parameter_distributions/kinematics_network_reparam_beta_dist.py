import numpy as np
import torch
from torch import Tensor, nn

from analyticalRL.networks.distributions.two_parameter_distributions.two_param_dist_network_base import \
    TwoParameterDistributionNetworkBase
from analyticalRL.networks.simple_kinematics_network import SimpleKinematicsNetwork


class KinematicsNetworkBetaDist(TwoParameterDistributionNetworkBase):
    """
    This class is used to create a neural network that predicts the angles of the joints of a planar robotic arm.
    The network takes two inputs, the parameters (DH or MDH) of the arm and the goal position.
    The network uses a linear stack of layers with ReLU activation functions.

    The output of the network is the parameter for a normal distribution for each joint.
    The loss function is the mean of the distances between the end effector positions of the parameters and the goal.
    """

    def __init__(self, num_joints, num_layer, layer_sizes, logger):
        super().__init__(num_joints, num_layer, layer_sizes, logger)

    def create_layer_stack_list(self, layer_sizes, num_joints, num_layer):
        stack_list = super().create_layer_stack_list(layer_sizes, num_joints, num_layer)
        stack_list.append(nn.ReLU())
        return stack_list

    def forward(self, model_input):
        network_output = super().forward(model_input)
        param, goal = model_input
        is_single_parameter = True if param.dim() == 2 else False

        all_distributions = None
        for joint_number in range(self.num_joints):
            index = 2 * joint_number

            if is_single_parameter:
                p_output = network_output[index]
                q_output = network_output[index + 1]
            else:
                p_output = network_output[:, index]
                q_output = network_output[:, index + 1]

            # Ensure p and q have to correct shape
            p = p_output.unsqueeze(-1) if p_output.dim() == 1 else p_output
            q = q_output.unsqueeze(-1) if q_output.dim() == 1 else q_output

            epsilon = 1e-2
            p = p + epsilon
            q = q + epsilon

            all_distributions = super().cat_distribution(all_distributions, is_single_parameter, p, param, q)

        return all_distributions

    def loss_fn(self, param, pred: Tensor, goal, ground_truth):
        is_single_parameter = True if param.dim() == 2 else False

        all_angles = None
        for joint_number in range(self.num_joints):
            p, q = self.extract_two_dist_parameters(is_single_parameter, joint_number, pred)

            beta_dist = torch.distributions.Beta(p, q)
            angle = beta_dist.rsample(torch.Size([1000])).mean(dim=0).unsqueeze(-1)
            # Map angle from [0,1] to [-pi, pi]
            angle = (2 * angle - 1) * np.pi

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
