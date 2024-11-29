import time

from stable_baselines3.common.utils import set_random_seed

from data_generation.goal_generator import generate_achievable_goal
from data_generation.parameter_generator import ParameterGeneratorForPlanarRobot
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

device = "cuda" if torch.cuda.is_available() else "cpu"
parameter_convention = "DH"
min_link_len = 0.3
max_link_len = 0.5

model_save_file_path = "comparison_results/model_save_files/benchmark/TwoPeakNormalLstmVariantDistrNetwork/dof3/TwoPeakNormalLstmVariantDistrNetwork_2024-11-28_06-18-30_model.pth"
number_of_joints = 3

repeats = 1000
batched_size = 100


def test_models_in_folder(num_of_joints=number_of_joints,
                          num_repeat=repeats,
                          batch_size=batched_size):
    set_random_seed(0)  # Have a unique test set for every trial

    generator = ParameterGeneratorForPlanarRobot(
        batch_size=1,
        device=device,
        tensor_type=torch.float32,
        num_joints=num_of_joints,
        parameter_convention=parameter_convention,
        min_len=min_link_len,
        max_len=max_link_len,
    )
    single_parameter = generator.get_random_dh_parameters()
    goal_single, _ = generate_achievable_goal(single_parameter, device)

    # Create batched input
    generator.batch_size = batch_size
    dh_parameters_batched = generator.get_random_dh_parameters()
    goal_batched, _ = generate_achievable_goal(dh_parameters_batched, device)

    # Test each model file and show progress bar
    with torch.no_grad():
        # Load model with modelclass
        loaded_model = torch.load(model_save_file_path, weights_only=False)
        loaded_model.eval()

        # Load model into memory by running it once in single and batched mode
        loaded_model((single_parameter, goal_single))
        loaded_model((dh_parameters_batched, goal_batched))

        single_runtimes = []
        batched_runtimes = []

        for i in range(num_repeat):
            start_time = time.perf_counter()
            loaded_model((single_parameter, goal_single))
            end_time = time.perf_counter()
            single_runtimes.append(end_time - start_time)

            # Try batched input
            start_time = time.perf_counter()
            loaded_model((dh_parameters_batched, goal_batched))
            end_time = time.perf_counter()
            batched_runtimes.append(end_time - start_time)
        # print in microseconds
        print(f"Single Runtime: {sum(single_runtimes) / len(single_runtimes) * 1e6} µs")
        print(f"Batched Runtime: {sum(batched_runtimes) / len(batched_runtimes) * 1e6} µs")


if __name__ == "__main__":
    test_models_in_folder()
