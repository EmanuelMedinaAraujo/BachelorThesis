import torch
from torch import nn, Tensor

from networks.analyticalRL.networks.distributions.two_peak_distributions.lstm.two_peak_norm_dist_lstm_base_network import \
    TwoPeakNormalLstmDistrNetworkBase
from networks.analyticalRL.networks.kinematics_network_base_class import concatenate_distributions


class NormalizeSelectionWeightsLayer(nn.Module):
    def __init__(self, num_joints):
        super(NormalizeSelectionWeightsLayer, self).__init__()
        self.num_joints = num_joints

    def forward(self, x):
        is_single_parameter = True if x.dim() == 1 else False

        if is_single_parameter:
            # Get the first entry and apply sigmoid to it
            first_entry = x[0]
            normalized_weight = torch.sigmoid(first_entry)
            return torch.cat([normalized_weight.unsqueeze(0), x[1:]])
        else:
            first_entries = x[:, 0]
            normalized_weight = torch.sigmoid(first_entries)
            return torch.cat([normalized_weight.unsqueeze(dim=-1), x[:, 1:]], dim=1)


class TwoPeakNormalLstmDistrNetwork(TwoPeakNormalLstmDistrNetworkBase):

    def __init__(self, num_joints, num_layer, layer_sizes, logger, error_tolerance, hidden_size, lstm_layers):
        super().__init__(num_joints, num_layer, layer_sizes, logger, error_tolerance)
        self.hidden_size = hidden_size  # number of features in hidden state
        self.num_layers = lstm_layers  # number of stacked LSTM layers

        self.input_size = num_joints * 3 + 2
        self.lstm_output_size = (num_joints * 6) + 1
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
        network_output = NormalizeSelectionWeightsLayer(self.num_joints)(out)

        if is_single_parameter:
            network_output = network_output.squeeze(0)
        return super().collect_distributions(is_single_parameter, network_output)

    # noinspection DuplicatedCode
    @staticmethod
    def map_six_parameters_ranges(parameter1, parameter2, parameter3, parameter4, parameter5, parameter6):
        # Use atan2 to calculate angle
        mu1 = torch.atan2(parameter1, parameter2)
        # Map sigma to positive values from [0,1] to [0,1]
        sigma1 = parameter3 * 0.5
        sigma1 = sigma1.clamp(min=1e-6)

        mu2 = torch.atan2(parameter4, parameter5)
        sigma2 = parameter6 * 0.5
        sigma2 = sigma2.clamp(min=1e-6)

        # Keep weights unchanged
        # Ensure the parameters have to correct shape
        mu1 = mu1.unsqueeze(-1) if mu1.dim() == 1 else mu1
        sigma1 = sigma1.unsqueeze(-1) if sigma1.dim() == 1 else sigma1
        mu2 = mu2.unsqueeze(-1) if mu2.dim() == 1 else mu2
        sigma2 = sigma2.unsqueeze(-1) if sigma2.dim() == 1 else sigma2

        return mu1, sigma1, mu2, sigma2

    @staticmethod
    def sample_component(mu1, mu2, sigma1, sigma2, component_selection):
        # Use the chosen component to select mu and sigma
        mu = mu1 * (component_selection == 0).float() + mu2 * (component_selection == 1).float()
        sigma = sigma1 * (component_selection == 0).float() + sigma2 * (component_selection == 1).float()
        return mu.squeeze(), sigma.squeeze()

    def extract_loss_variable_from_parameters(self, mu1, sigma1, mu2, sigma2, component_selection):
        mu, sigma = self.sample_component(mu1, mu2, sigma1, sigma2, component_selection)
        with torch.no_grad():
            noise = torch.randn(mu.size()).to(mu1.device)

        # Reparameterized sampling
        samples = mu + sigma * noise
        return samples

    def loss_fn(self, param, pred: Tensor, goal, ground_truth):
        is_single_parameter = True if param.dim() == 2 else False

        # Get weight and generate which component to use randomly
        if is_single_parameter:
            weight = pred[0, 0]
            component_selection = torch.tensor([weight > 0.5]).to(param.device)
        else:
            weight = pred[:, 0, 0]
            component_selection = torch.tensor(weight > 0.5).to(param.device)

        all_loss_variables = None
        for joint_number in range(self.num_joints):
            mu1, sigma1, mu2, sigma2 = self.extract_four_parameters(is_single_parameter,
                                                                    joint_number, pred)

            loss_variable = self.extract_loss_variable_from_parameters(mu1, sigma1, mu2, sigma2,
                                                                       component_selection).unsqueeze(-1)
            if all_loss_variables is None:
                all_loss_variables = loss_variable
            else:
                if is_single_parameter:
                    all_loss_variables = torch.cat([all_loss_variables, loss_variable]).to(param.device)
                else:
                    all_loss_variables = torch.cat([all_loss_variables, loss_variable], dim=1).to(param.device)
        return self.calculate_batch_loss(all_loss_variables, goal, param)

    @staticmethod
    def extract_four_parameters(is_single_parameter, joint_number, pred):
        if is_single_parameter:
            distribution_params = pred[joint_number]
            parameter1 = torch.tensor([distribution_params[1]]).to(pred.device)
            parameter2 = torch.tensor([distribution_params[2]]).to(pred.device)
            parameter3 = torch.tensor([distribution_params[3]]).to(pred.device)
            parameter4 = torch.tensor([distribution_params[4]]).to(pred.device)
        else:
            distribution_params = pred[:, joint_number]
            parameter1 = distribution_params[:, 1]
            parameter2 = distribution_params[:, 2]
            parameter3 = distribution_params[:, 3]
            parameter4 = distribution_params[:, 4]
        return parameter1, parameter2, parameter3, parameter4

    def collect_distributions(self, is_single_parameter, network_output):
        all_distributions = None
        if is_single_parameter:
            weight = network_output[0]
        else:
            weight = network_output[:, 0]
        for joint_number in range(self.num_joints):
            index = 1 + 6 * joint_number

            if is_single_parameter:
                parameter1 = network_output[index]
                parameter2 = network_output[index + 1]
                parameter3 = network_output[index + 2]
                parameter4 = network_output[index + 3]
                parameter5 = network_output[index + 4]
                parameter6 = network_output[index + 5]
            else:
                parameter1 = network_output[:, index]
                parameter2 = network_output[:, index + 1]
                parameter3 = network_output[:, index + 2]
                parameter4 = network_output[:, index + 3]
                parameter5 = network_output[:, index + 4]
                parameter6 = network_output[:, index + 5]

            (mu1, sigma1, mu2, sigma2) = self.map_six_parameters_ranges(parameter1,
                                                                        parameter2,
                                                                        parameter3,
                                                                        parameter4,
                                                                        parameter5,
                                                                        parameter6)

            if is_single_parameter:
                distribution = torch.cat([weight.unsqueeze(-1), mu1.unsqueeze(-1), sigma1.unsqueeze(-1),
                                          mu2.unsqueeze(-1), sigma2.unsqueeze(-1)
                                          ])
            else:
                distribution = torch.cat(
                    [weight.unsqueeze(-1), mu1, sigma1, mu2, sigma2],
                    dim=-1)

            all_distributions = concatenate_distributions(all_distributions, distribution, is_single_parameter)
        return all_distributions

