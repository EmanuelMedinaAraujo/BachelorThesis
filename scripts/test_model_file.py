from stable_baselines3.common.utils import set_random_seed

from data_generation.parameter_dataset import CustomParameterDataset
from networks.analyticalRL.analytical_training_and_tests import test_model
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

model_file_path = "model_saves_files_perf_direct/dof2/SimpleKinematicsNetwork_2024-11-24_15-36-09_model.pth"
modelclass = SimpleKinematicsNetwork
test_set_length = 10000
device = "cuda" if torch.cuda.is_available() else "cpu"
num_of_joints = 2
parameter_convention = "DH"
min_link_len = 0.3
max_link_len = 0.5

problems_to_visualize = 2


def test_model_from_file():
    set_random_seed(0)  # Have a unique test set for every trial
    # Create Test Set
    test_dataset = CustomParameterDataset(
        length=test_set_length,
        device_to_use=device,
        num_of_joints=num_of_joints,
        parameter_convention=parameter_convention,
        min_link_len=min_link_len,
        max_link_len=max_link_len,
    )

    # Load model with modelclass
    loadedModel = torch.load(model_file_path, weights_only=False)
    loadedModel.eval()
    # Create cfg for logger

    test_model(
        test_dataset=test_dataset,
        model=loadedModel,
        logger=GeneralLogger(None, False, True, True),
        num_epoch=0,
    )

if __name__ == "__main__":
    test_model_from_file()