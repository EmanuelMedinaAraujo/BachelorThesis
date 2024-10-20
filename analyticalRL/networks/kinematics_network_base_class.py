from abc import abstractmethod, ABC

import torch
from torch import nn

from custom_logging.custom_loggger import GeneralLogger
from util.forward_kinematics import calculate_parameter_goal_distances, update_theta_values


class KinematicsNetworkBase(nn.Module, ABC):

    @abstractmethod
    def __init__(self, num_joints, num_layer, layer_sizes, logger: GeneralLogger):
        """
        Initializes the KinematicsNetwork.

        Args:
            num_joints: The number of joints in the planar robotic arm
            num_layer: The number of layers in the network
            layer_sizes: A list containing the number of neurons in each layer
        """
        super().__init__()
        self.num_joints = num_joints
        self.flatten = nn.Flatten()
        stack_list = self.create_layer_stack_list(layer_sizes, num_joints, num_layer)
        self.linear_relu_stack = nn.Sequential(*stack_list)
        logger.log_network_architecture(self.linear_relu_stack)

    @staticmethod
    def create_layer_stack_list(layer_sizes, num_joints, num_layer):
        stack_list = [nn.Linear(num_joints * 3 + 2, layer_sizes[0]), nn.ReLU()]
        for i in range(num_layer - 1):
            stack_list.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            stack_list.append(nn.ReLU())
        stack_list.append(nn.Linear(layer_sizes[-1], num_joints * 2))
        return stack_list

    @staticmethod
    def calc_distances(param, angles_pred, goal):
        """
        Calculates the distances between the end effector positions of the parameters with the given angles and the goal.
        """
        # Update theta values with predictions
        updated_param = update_theta_values(parameters=param, new_theta_values=angles_pred)
        distances = calculate_parameter_goal_distances(updated_param, goal)
        return distances

    def forward(self, model_input):
        param, goal = model_input

        is_single_parameter = True if param.dim() == 2 else False

        # Flatten the param
        if is_single_parameter:
            # If the input is a single parameter
            flatten_param = torch.flatten(param)
        else:
            flatten_param = self.flatten(param)

        # Concatenate flatten_param and goal along the second dimension
        if is_single_parameter:
            # Input is a single parameter
            flatten_input = torch.cat((flatten_param, goal))
        else:
            flatten_input = torch.cat((flatten_param, goal), dim=1)

        # Pass the flatten input through the linear_relu_stack
        return self.linear_relu_stack(flatten_input)

    @abstractmethod
    def loss_fn(self, param, pred, goal, ground_truth):
        pass
