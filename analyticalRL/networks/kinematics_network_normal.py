import torch

from analyticalRL.networks.kinematics_network_base_class import KinematicsNetworkBase
from util.forward_kinematics import calculate_parameter_goal_distances, update_theta_values


class KinematicsNetwork(KinematicsNetworkBase):
    """
    This class is used to create a neural network that predicts the angles of the joints of a planar robotic arm.
    The network takes two inputs, the parameters (DH or MDH) of the arm and the goal position.
    The network uses a linear stack of layers with ReLU activation functions.

    The output of the network is the two values for each joint from which the angle can be calculated.
    The loss function is the mean of the distances between the end effector positions of the parameters and the goal.
    """

    def forward(self, model_input):
        network_output = super().forward(model_input)
        param, goal = model_input

        is_single_parameter = True if param.dim() == 2 else False

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

    @staticmethod
    def calc_distances(param, pred, goal):
        """
        Calculates the distances between the end effector positions of the parameters and the goal.
        """
        # Update theta values with predictions
        updated_param = update_theta_values(parameters=param, new_theta_values=pred)
        distances = calculate_parameter_goal_distances(updated_param, goal)
        return distances

    def loss_fn(self, param, pred, goal, ground_truth):
        """
        Calculates the loss for the given parameters and goal.
        The loss is calculated as the mean of the distances between the end effector positions of the parameters and the goal.
        """
        distances = KinematicsNetwork.calc_distances(param=param, pred=pred, goal=goal)
        return distances.mean()
