import hydra
from hydra.core.config_store import ConfigStore
from stable_baselines3.common.utils import set_random_seed

from conf.conf_dataclasses.config import TrainConfig
from data_generation.parameter_dataset import CustomParameterDataset
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

# Path from repository root
model_file_path = "model_saves_files_perf_direct/dof3/SimpleKinematicsNetwork_2024-11-24_16-15-32_model.pth"
num_of_joints = 3
num_of_visualizations = 2
plot_all_in_one = False
save_plot = False
show_plot = True

device = "cuda" if torch.cuda.is_available() else "cpu"
parameter_convention = "DH"
min_link_len = 0.3
max_link_len = 0.5

cs = ConfigStore.instance()
cs.store(name="train_conf", node=TrainConfig)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def test_model_from_file(train_config: TrainConfig):
    set_random_seed(0)  # Have a unique test set for every trial
    # Create Test Set
    test_dataset = CustomParameterDataset(
        length=num_of_visualizations,
        device_to_use=device,
        num_of_joints=num_of_joints,
        parameter_convention=parameter_convention,
        min_link_len=min_link_len,
        max_link_len=max_link_len,
    )

    visualization_params = []
    visualization_goals = []
    visualization_ground_truth = []
    for x in range(num_of_visualizations):
        param, goal, ground_truth = test_dataset.__getitem__(x)
        visualization_params.append(param)
        visualization_goals.append(goal)
        visualization_ground_truth.append(ground_truth)

    # Get output type by getting file name until first underscore
    output_type = model_file_path.split("/")[-1].split("_")[0]
    # Set output type in train_config
    train_config.hyperparams.analytical.output_type = output_type
    train_config.number_of_joints = num_of_joints
    train_config.vis.num_problems_to_visualize = num_of_visualizations
    train_config.vis.save_to_file = save_plot
    train_config.vis.show_plot = show_plot
    train_config.vis.analytical.plot_all_in_one = plot_all_in_one
    # Set configs here if wanted

    # Load model with modelclass
    loaded_model = torch.load(model_file_path, weights_only=False)
    loaded_model.eval()
    # Create cfg for logger
    logger = GeneralLogger(None, False, True, True),

    visualize_analytical_model(train_config, device, 0, logger, loaded_model, visualization_goals,
                               visualization_ground_truth, [], visualization_params)


if __name__ == "__main__":
    test_model_from_file()
