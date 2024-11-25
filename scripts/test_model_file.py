import os

from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm

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
from networks.analyticalRL.networks.simple_kinematics_network import SimpleKinematicsNetwork  # noqa

model_save_folder_path = "model_saves_files_perf_direct/dof2/"
number_of_joints = 2

test_set_length = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
parameter_convention = "DH"
min_link_len = 0.3
max_link_len = 0.5


def test_models_in_folder(num_of_joints=number_of_joints,
                          model_folder_path=model_save_folder_path) -> list[tuple[str, int, float]]:
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

    # List all model files in the folder
    model_files = [f for f in os.listdir(model_folder_path) if f.endswith(".pth")]

    loss_and_acc_list = []
    # Test each model file and show progress bar
    for model_file in tqdm(model_files, desc="Testing models"):
        model_file_path = os.path.join(model_folder_path, model_file)

        # Load model with modelclass
        loaded_model = torch.load(model_file_path, weights_only=False)
        loaded_model.eval()

        loss, acc = test_model(
            test_dataset=test_dataset,
            model=loaded_model,
            logger=GeneralLogger(None, False, True, True),
            num_epoch=0,
        )
        loss_and_acc_list.append((model_file, loss, acc))
    return loss_and_acc_list

if __name__ == "__main__":
    test_models_in_folder()
