import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd import Variable

from analyticalRL.networks.distributions.two_peak_distributions.two_peak_norm_dist_network_base import \
    TwoPeakNormalDistrNetworkBase

class TwoPeakNormalLstmVariantDistrNetwork(TwoPeakNormalDistrNetworkBase):

    def __init__(self, num_joints, num_layer, layer_sizes, logger, error_tolerance):
        super().__init__(num_joints, num_layer, layer_sizes, logger, error_tolerance)
        self.hidden_size = 256  # number of features in hidden state
        self.input_size = num_joints * 3 + 2
        self.num_layers = 1  # number of stacked LSTM layers
        self.output_size = num_joints * 8
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.dense_layer = nn.Linear(self.hidden_size,
                                     64)
        self.last_dense_layer = nn.Linear(64, self.output_size)
        self.relu = nn.ReLU()

    def flatten_model_input(self, model_input):
        flatten_input = super().flatten_model_input(model_input)
        param, _ = model_input
        is_single_parameter = True if param.dim() == 2 else False
        return flatten_input.unsqueeze(0 if is_single_parameter else 1)

    def forward(self, model_input):
        flatten_input = super().flatten_model_input(model_input)
        param, _ = model_input
        is_single_parameter = True if param.dim() == 2 else False

        if is_single_parameter:
            h_0 = Variable(torch.zeros(self.num_layers, self.hidden_size)).to(param.device)
            c_0 = Variable(torch.zeros(self.num_layers, self.hidden_size)).to(param.device)
        else:
            h_0 = Variable(torch.zeros(self.num_layers, flatten_input.size(0), self.hidden_size)).to(param.device)
            c_0 = Variable(torch.zeros(self.num_layers, flatten_input.size(0), self.hidden_size)).to(param.device)
        # Propagate input through LSTM
        output, _ = self.lstm(flatten_input, (h_0, c_0))  # lstm with input, hidden, and internal state
        out = output.squeeze(1)  # reshaping the data for Dense layer next
        out = self.relu(out)
        out = self.dense_layer(out)
        out = self.relu(out)
        out = self.last_dense_layer(out)
        network_output = NormalizeAtan2WeightsLayer(self.num_joints)(out)

        # if is_single_parameter:
        #     network_output = network_output.squeeze(0)

        return self.collect_distributions(is_single_parameter, network_output, param)