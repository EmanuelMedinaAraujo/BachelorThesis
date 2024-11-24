from torch import nn

from analyticalRL.networks.distributions.two_peak_distributions.lstm.two_peak_norm_dist_lstm_base_network import \
    TwoPeakNormalLstmDistrNetworkBase
from analyticalRL.networks.distributions.two_peak_distributions.two_peak_norm_dist_network_base import \
    NormalizeWeightsLayer


class TwoPeakNormalLstmDistrNetwork(TwoPeakNormalLstmDistrNetworkBase):

    def __init__(self, num_joints, num_layer, layer_sizes, logger, error_tolerance, hidden_size, lstm_layers):
        super().__init__(num_joints, num_layer, layer_sizes, logger, error_tolerance)
        self.hidden_size = hidden_size  # number of features in hidden state
        self.num_layers = lstm_layers  # number of stacked LSTM layers

        self.input_size = num_joints * 3 + 2
        self.lstm_output_size = num_joints * 8
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, proj_size=self.lstm_output_size, batch_first=True)

    def forward(self, model_input):
        flatten_input = super().flatten_model_input(model_input)
        param, _ = model_input
        is_single_parameter = True if param.dim() == 2 else False

        # Propagate input through LSTM
        output = super().forward_in_lstm(flatten_input,
                                         is_single_parameter)  # lstm with input, hidden, and internal state
        out = output.squeeze(1)  # reshaping the data for Dense layer next
        network_output = NormalizeWeightsLayer(self.num_joints)(out)

        if is_single_parameter:
            network_output = network_output.squeeze(0)
        return super().collect_distributions(is_single_parameter, network_output)
