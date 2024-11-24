from networks.analyticalRL.networks.distributions.two_peak_distributions.two_peak_norm_dist_network_base import \
    TwoPeakNormalDistrNetworkBase, NormalizeWeightsLayer


class TwoPeakNormalDistrNetwork(TwoPeakNormalDistrNetworkBase):

    def __init__(self, num_joints, num_layer, layer_sizes, logger, error_tolerance):
        super().__init__(num_joints, num_layer, layer_sizes, logger, error_tolerance)

    def create_layer_stack_list(self, layer_sizes, num_joints, num_layer, output_per_joint):
        stack_list = super().create_layer_stack_list(layer_sizes, num_joints, num_layer, output_per_joint)
        stack_list.append(NormalizeWeightsLayer(num_joints))
        return stack_list

    def forward(self, model_input):
        network_output = super().forward(model_input)
        param, _ = model_input
        is_single_parameter = True if param.dim() == 2 else False

        return super().collect_distributions(is_single_parameter, network_output)
