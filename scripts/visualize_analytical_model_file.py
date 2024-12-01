import hydra
from hydra.core.config_store import ConfigStore
from stable_baselines3.common.utils import set_random_seed

from conf.conf_dataclasses.config import TrainConfig
from networks.analyticalRL.analytical_training_and_tests import visualize_analytical_model
from networks.analyticalRL.networks.distributions.one_peak_distributions.beta_rsample_dist_network import *  # noqa
from networks.analyticalRL.networks.distributions.one_peak_distributions.normal_distributions.ground_truth_loss_network import *  # noqa
from networks.analyticalRL.networks.distributions.one_peak_distributions.normal_distributions.lstm_rsample_network import *  # noqa
from networks.analyticalRL.networks.distributions.one_peak_distributions.normal_distributions.manual_reparam_network import *  # noqa
from networks.analyticalRL.networks.distributions.one_peak_distributions.normal_distributions.mu_distance_loss_network import *  # noqa
from networks.analyticalRL.networks.distributions.one_peak_distributions.normal_distributions.rsample_network import *  # noqa
from networks.analyticalRL.networks.distributions.two_peak_distributions.lstm.two_peak_norm_dist_lstm_network import *  # noqa
from networks.analyticalRL.networks.distributions.two_peak_distributions.lstm.two_peak_norm_dist_lstm_network_variant import *  # noqa
from networks.analyticalRL.networks.distributions.two_peak_distributions.two_peak_norm_dist_network import *  # noqa
from networks.analyticalRL.networks.simple_kinematics_network import *  # noqa

device = "cuda" if torch.cuda.is_available() else "cpu"
parameter_convention = "DH"
min_link_len = 0.3
max_link_len = 0.5

# Path from repository root
# model_file_path = "comparison_results/model_save_files/benchmark/NormalDistrManualReparameterizationNetwork/dof2/NormalDistrManualReparameterizationNetwork_2024-11-27_05-51-12_model.pth"
# model_file_path = "comparison_results/model_save_files/benchmark/NormalDistrManualReparameterizationNetwork/dof3/NormalDistrManualReparameterizationNetwork_2024-11-27_06-53-31_model.pth"
# model_file_path = "comparison_results/model_save_files/benchmark/SimpleKinematicsNetwork/dof3/SimpleKinematicsNetwork_2024-11-26_15-21-58_model.pth"
model_file_path = "comparison_results/model_save_files/benchmark/TwoPeakNormalLstmDistrNetwork/dof2/TwoPeakNormalLstmDistrNetwork_2024-11-28_05-30-53_model.pth"

plot_all_in_one = False
save_plot = False
show_plot = True

cs = ConfigStore.instance()
cs.store(name="train_conf", node=TrainConfig)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def test_model_from_file(train_config: TrainConfig):
    if "dof2" in model_file_path:
        num_of_joints = 2
    elif "dof3" in model_file_path:
        num_of_joints = 3
    else:
        raise ValueError("Number of joints not found in model_file_path")

    set_random_seed(0 if num_of_joints == 2 else 42)  # Have a unique test set for every trial

    visualization_params = [torch.tensor([[0.0, 0.5, 0.0],
                                          [0.0, 0.5, 0.0],
                                          [0.0, 0.5, 0.0]
                                          ], device='cuda:0')] if num_of_joints == 3 else [torch.tensor([[0.0, 0.4, 0.0],
                                                                                                         [0.0, 0.45, 0.0]],
                                                                                                        device='cuda:0')]
    visualization_goals = [torch.tensor([-0.75, 1.], device='cuda:0')] if num_of_joints == 3 else [
        torch.tensor([0.6, 0.5], device='cuda:0')]
    visualization_ground_truth = [torch.tensor([[4.3548],
                                                [3.2458]], device='cuda:0')]

    # Get output type by getting file name until first underscore
    output_type = model_file_path.split("/")[-1].split("_")[0]
    # Set output type in train_config
    train_config.hyperparams.analytical.output_type = output_type
    train_config.number_of_joints = num_of_joints
    train_config.vis.num_problems_to_visualize = 1
    train_config.vis.save_to_file = save_plot
    train_config.vis.show_plot = show_plot
    train_config.vis.analytical.plot_all_in_one = plot_all_in_one
    train_config.vis.max_legend_length = 0
    train_config.vis.analytical.default_line_width = 1.
    train_config.vis.analytical.default_line_transparency = 1. if "SimpleKinematicsNetwork" in model_file_path else 0.1

    # Set configs here if wanted

    # Load model with modelclass
    loaded_model = torch.load(model_file_path, weights_only=False)
    loaded_model.eval()
    # Create cfg for logger
    logger = GeneralLogger(train_config, False, True, True)

    visualize_analytical_model(train_config, device, 0, logger, loaded_model, visualization_goals,
                               visualization_ground_truth, [], visualization_params)


if __name__ == "__main__":
    test_model_from_file()
