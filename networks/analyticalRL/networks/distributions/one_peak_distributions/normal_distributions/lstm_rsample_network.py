import torch
from torch import nn
from torch.autograd import Variable
from networks.analyticalRL.networks.distributions.one_peak_distributions.normal_distributions.network_base import \
    NormalDistrNetworkBase
from custom_logging.custom_loggger import GeneralLogger


class NormalDistrRandomSampleLSTMDistNetwork(NormalDistrNetworkBase):
    """
    This class is used to create a neural network that predicts the angles of the joints of a planar robotic arm.
    The network takes two inputs, the parameters (DH or MDH) of the arm and the goal position.
    The network uses a linear stack of layers with ReLU activation functions.

    The output of the network is the parameter for a normal distribution for each joint.
    The loss function is the mean of the distances between the end effector positions of the parameters and the goal.
    """

    def __init__(self, num_joints, num_layer, layer_sizes, logger: GeneralLogger, error_tolerance, hidden_size, lstm_layers):
        super().__init__(num_joints, num_layer, layer_sizes, logger, error_tolerance)
        self.hidden_size = hidden_size  # number of features in hidden state
        self.num_layers = lstm_layers  # number of stacked LSTM layers

        self.input_size = num_joints * 3 + 2
        self.lstm_output_size = num_joints * 3
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, proj_size=self.lstm_output_size, batch_first=True)

    def forward_in_lstm(self, flatten_input, is_single_parameter):
        if is_single_parameter:
            h_0 = Variable(torch.zeros(self.num_layers, self.lstm_output_size)).to(flatten_input.device)
            c_0 = Variable(torch.zeros(self.num_layers, self.hidden_size)).to(flatten_input.device)
        else:
            h_0 = Variable(torch.zeros(self.num_layers, flatten_input.size(0), self.lstm_output_size)).to(
                flatten_input.device)
            c_0 = Variable(torch.zeros(self.num_layers, flatten_input.size(0), self.hidden_size)).to(
                flatten_input.device)
        # Propagate input through LSTM
        output, _ = self.lstm(flatten_input, (h_0, c_0))  # lstm with input, hidden, and internal state
        return output

    def flatten_model_input(self, model_input):
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
        # Create a self.num_layer tensor where each element is flatten_input
        return flatten_input.unsqueeze(0 if is_single_parameter else 1)

    @staticmethod
    def extract_loss_variable_from_parameters(mu, sigma, ground_truth, is_single_parameter, joint_number):
        normal_dist = torch.distributions.Normal(loc=mu, scale=sigma)
        # Return angle as loss variable
        return normal_dist.rsample()

    def forward(self, model_input):
        flatten_input = self.flatten_model_input(model_input)
        param, _ = model_input
        is_single_parameter = True if param.dim() == 2 else False

        # Propagate input through LSTM
        output = self.forward_in_lstm(flatten_input,
                                         is_single_parameter)  # lstm with input, hidden, and internal state
        network_output = output.squeeze(1)  # reshaping the data for Dense layer next

        if is_single_parameter:
            network_output = network_output.squeeze(0)
        all_distributions = None
        for joint_number in range(self.num_joints):
            index = 3 * joint_number

            if is_single_parameter:
                parameter1 = network_output[index]
                parameter2 = network_output[index + 1]
                parameter3 = network_output[index + 2]
            else:
                parameter1 = network_output[:, index]
                parameter2 = network_output[:, index + 1]
                parameter3 = network_output[:, index + 2]

            # Use atan2 to calculate angle
            mu1 = torch.atan2(parameter1, parameter2)
            # Map sigma to positive values from [0,1] to [0,2]
            sigma1 = parameter3 * 2
            sigma1 = sigma1.clamp(min=1e-6)

            # Ensure the parameters have to correct shape
            parameter1 = mu1.unsqueeze(-1) if mu1.dim() == 1 else mu1
            parameter2 = sigma1.unsqueeze(-1) if sigma1.dim() == 1 else sigma1

            if is_single_parameter:
                distribution = torch.cat([parameter1.unsqueeze(-1), parameter2.unsqueeze(-1)])
            else:
                distribution = torch.cat([parameter1, parameter2], dim=-1)
            if all_distributions is None:
                all_distributions = distribution.unsqueeze(0 if is_single_parameter else 1)
            else:
                if is_single_parameter:
                    all_distributions = torch.cat([all_distributions, distribution.unsqueeze(0)]).to(param.device)
                else:
                    all_distributions = torch.cat([all_distributions, distribution.unsqueeze(1)], dim=1).to(
                        param.device)

        return all_distributions

