from networks.analyticalRL.networks.distributions.two_peak_distributions.two_peak_norm_dist_network_base import \
    TwoPeakNormalDistrNetworkBase
from torch import nn
import torch


class NormalizeWeightsLayer(nn.Module):
    def __init__(self, num_joints):
        super(NormalizeWeightsLayer, self).__init__()
        self.num_joints = num_joints

    def forward(self, x):
        is_single_parameter = True if x.dim() == 1 else False

        if is_single_parameter:
            structured_input_view = x.view(-1, self.num_joints, 4)
            # Do Softmax on weights (every fourth row) of the input, because they should sum up to 1
            softmax_weights = torch.softmax(structured_input_view[:, :, 3], dim=1)
            return torch.cat([structured_input_view[:, :, :3], softmax_weights.unsqueeze(-1)], dim=-1).flatten()
        else:
            structured_input_view = x.view(-1, self.num_joints, 2, 4)
            # Do Softmax on weights (every fourth row) of the input, because they should sum up to 1
            softmax_weights = torch.softmax(structured_input_view[:, :, :, 3], dim=2)
            return torch.cat([structured_input_view[:, :, :, :3], softmax_weights.unsqueeze(-1)], dim=-1).flatten(
                start_dim=1)

class TwoPeakNormalDistrNetwork(TwoPeakNormalDistrNetworkBase):

    def __init__(self, num_joints, num_layer, layer_sizes, logger, error_tolerance):
        super().__init__(num_joints, num_layer, layer_sizes, logger, error_tolerance)

    def create_layer_stack_list(self, layer_sizes, num_joints, num_layer, output_per_joint):
        stack_list = super().create_layer_stack_list(layer_sizes, num_joints, num_layer, output_per_joint)
        stack_list.pop()
        stack_list.append(nn.Linear(layer_sizes[-1], (num_joints * output_per_joint)+1))
        stack_list.append(NormalizeWeightsLayer(num_joints))
        return stack_list

    def forward(self, model_input):
        network_output = super().forward(model_input)
        param, _ = model_input
        is_single_parameter = True if param.dim() == 2 else False

        return super().collect_distributions(is_single_parameter, network_output)
