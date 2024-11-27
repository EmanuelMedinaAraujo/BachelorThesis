from abc import ABC, abstractmethod

import torch
from torch.autograd import Variable

from networks.analyticalRL.networks.distributions.two_peak_distributions.two_peak_norm_dist_network_base import \
    TwoPeakNormalDistrNetworkBase


class TwoPeakNormalLstmDistrNetworkBase(TwoPeakNormalDistrNetworkBase, ABC):

    @abstractmethod
    def __init__(self, num_joints, num_layer, layer_sizes, logger, error_tolerance):
        super().__init__(num_joints, num_layer, layer_sizes, logger, error_tolerance)

    # noinspection DuplicatedCode
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
        flatten_input = super().flatten_model_input(model_input)
        param, _ = model_input
        is_single_parameter = True if param.dim() == 2 else False
        # Create a self.num_layer tensor where each element is flatten_input
        return flatten_input.unsqueeze(0 if is_single_parameter else 1)
