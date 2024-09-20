import torch
from torch import nn

from util.forward_kinematics import calculate_distances


class KinematicsNetwork(nn.Module):
    """
    This class is used to create a neural network that predicts the angles of the joints of a planar robotic arm.
    The network takes two inputs, the parameters (DH or MDH) of the arm and the goal position.
    The network uses a linear stack of layers with ReLU activation functions.

    The output of the network is the two values for each joint from which the angle can be calculated.
    The loss function is the mean of the distances between the end effector positions of the parameters and the goal.
    """

    def __init__(self, num_joints, num_layer, layer_sizes):
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
        stack_list = [nn.Linear(num_joints * 3 + 2, layer_sizes[0]), nn.ReLU()]
        for i in range((num_layer - 2)):
            stack_list.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            stack_list.append(nn.ReLU())
        stack_list.append(nn.Linear(layer_sizes[-1], num_joints * 2))
        self.linear_relu_stack = nn.Sequential(*stack_list)
        print(self.linear_relu_stack)

    def forward(self, model_input):
        param, goal = model_input

        is_single_parameter = False
        # Flatten the param
        flatten_param = self.flatten(param)
        if param.dim() == 2:
            # If the input is a single parameter
            is_single_parameter = True
            flatten_param = torch.flatten(param)

        # Concatenate flatten_param and goal along the second dimension
        if is_single_parameter:
            flatten_input = torch.cat((flatten_param, goal))
        else:
            flatten_input = torch.cat((flatten_param, goal), dim=1)

        # Pass the flatten input through the linear_relu_stack
        network_output = self.linear_relu_stack(flatten_input)

        # Calculate the angles from the network output.
        # The network outputs 2 values for each joint, sin(x) and cos(y).
        # The angle is then calculated as atan2(sin(x), cos(y)).
        # This is done for each joint.
        all_angles = None
        for joint_number in range(self.num_joints):
            index = 2 * joint_number
            if is_single_parameter:
                sin_x = network_output[index]
                cos_y = network_output[index + 1]
            else:
                sin_x = network_output[:, index]
                cos_y = network_output[:, index + 1]
            angle = torch.atan2(sin_x, cos_y).unsqueeze(dim=-1)
            if all_angles is None:
                all_angles = angle
            else:
                if is_single_parameter:
                    all_angles = torch.cat([all_angles, angle]).to(param.device)
                else:
                    all_angles = torch.cat([all_angles, angle], dim=1).to(param.device)

        return all_angles


def loss_fn(param, goal):
    """
    Calculates the loss for the given parameters and goal.
    The loss is calculated as the mean of the distances between the end effector positions of the parameters and the goal.
    """
    distances = calculate_distances(param, goal)
    return distances.mean()
