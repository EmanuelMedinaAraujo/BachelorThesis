import os
import sys

import yaml
from tqdm import tqdm

from benchmark_script import execute_main_script
from test_model_file import test_models_in_folder

# List of output types to iterate over
analytical_direct = {
    "num_hidden_layer": 3,
    "hidden_layer_sizes": [2048, 256, 2048],
    "learning_rate": 0.001492270739565321,
    "batch_size": 512,
    "problems_per_epoch": 7168,
    "optimizer": "Adam",
    "output_type": "SimpleKinematicsNetwork"
}
# One peak distribution
one_peak_dist = {
    "num_hidden_layer": 3,
    "hidden_layer_sizes": [128, 512, 2048],
    "learning_rate": 0.00043306334967391496,
    "batch_size": 16,
    "problems_per_epoch": 112,
    "optimizer": "Adam",
    "output_type": "NormalDistrManualReparameterizationNetwork"
}
# Beta
beta = {
    "num_hidden_layer": 4,
    "hidden_layer_sizes": [16, 16, 1024, 128],
    "learning_rate": 0.0004133444288382607,
    "batch_size": 32,
    "problems_per_epoch": 288,
    "optimizer": "Adam",
    "output_type": "BetaDistrRSampleMeanNetwork"
}
# One peak lstm
one_peak_lstm = {
    "learning_rate": 0.004623192404444301,
    "batch_size": 256,
    "problems_per_epoch": 4608,
    "optimizer": "Adam",
    "lstm_hidden_size": 1024,
    "lstm_num_layers": 4,
    "output_type": "NormalDistrRandomSampleLSTMDistNetwork",

    # dummy parameter needed for initialization
    "num_hidden_layer": 0,
    "hidden_layer_sizes": []
}
# Two peak distribution
two_peak = {
    "num_hidden_layer": 2,
    "hidden_layer_sizes": [2048, 32],
    "learning_rate": 0.008179331918920956,
    "batch_size": 256,
    "problems_per_epoch": 3328,
    "optimizer": "Adam",
    "output_type": "TwoPeakNormalDistrNetwork"
}
# Two peak LSTM variant distribution
two_peak_lstm = {
    "learning_rate": 0.0027334146724357724,
    "batch_size": 512,
    "problems_per_epoch": 1536,
    "lstm_hidden_size": 256,
    "lstm_num_layers": 3,
    "num_hidden_layer": 4,
    "hidden_layer_sizes": [4, 128, 8, 128],
    "optimizer": "Adam",
    "output_type": "TwoPeakNormalLstmVariantDistrNetwork"
}

##########Alternatives############
# One peak distribution (mudistance)
one_peak_dist_mudistance = {
    "num_hidden_layer": 4,
    "hidden_layer_sizes": [1024, 128, 64, 256],
    "learning_rate": 0.0002920265071713353,
    "batch_size": 64,
    "problems_per_epoch": 1024,
    "optimizer": "Adam",
    "output_type": "NormalDistrMuDistanceNetworkBase"
}
# One peak distribution (torch reparam)
one_peak_dist_torch_reparam = {
    "num_hidden_layer": 2,
    "hidden_layer_sizes": [1042, 64],
    "learning_rate": 0.00012210729721590343,
    "batch_size": 256,
    "problems_per_epoch": 8,
    "optimizer": "Adam",
    "output_type": "NormalDistrRandomSampleDistNetwork"
}
# One peak distribution (Ground Truth)
one_peak_dist_ground_truth = {
    "num_hidden_layer": 20,
    "hidden_layer_sizes": [16, 32, 512, 2048, 256, 2048, 8, 256, 8, 1024, 2, 2, 16, 256, 512, 16, 4, 256, 2048, 65535],
    "learning_rate": 0.0004062026352168701,
    "batch_size": 128,
    "optimizer": "Adam",
    "output_type": "NormalDistrGroundTruthLossNetwork"
}
# Two peak LSTM (Non-Variant)
two_peak_lstm_non_variant = {
    "learning_rate": 0.0009395146226113611,
    "batch_size": 128,
    "problems_per_epoch": 1792,
    "optimizer": "Adam",
    "output_type": "TwoPeakNormalLstmDistrNetwork",
    "lstm_hidden_size": 256,
    "lstm_num_layers": 1,

    # dummy parameter needed for initialization
    "num_hidden_layer": 0,
    "hidden_layer_sizes": []
}
##################################

# List to store all configurations
configurations = [
    # analytical_direct,
    one_peak_dist,
    # one_peak_lstm,
    # beta,
    # two_peak,
    two_peak_lstm,

    # Alternatives
    # one_peak_dist_mudistance,
    # one_peak_dist_torch_reparam,
    # one_peak_dist_ground_truth,
    # two_peak_lstm_non_variant
]

num_joints_to_test = [2,3]
test_length = 10000
save_folder_path_prefix = 'outputs/model_save_files/benchmark'

# Path to the configuration file
hyperparameters_config_file_path = 'conf/hyperparams/hyperparams.yaml'
config_file_path = 'conf/config.yaml'

number_of_repeats = 5

def run_benchmark_script():
    script_folder = os.path.dirname(os.path.abspath(__file__))
    benchmark_script = os.path.join(script_folder, "benchmark_script.py")

    if not os.path.isfile(benchmark_script):
        print("Error: 'benchmark_script.py' not found in the current folder.")
        return []

    return execute_main_script(number_of_repeats)


def run_various_configurations():
    # Check if folders might already exist and ask for deletion confirmation
    for num_joint in num_joints_to_test:
        for run_configuration in configurations:
            folder_path = save_folder_path_prefix + '/' + run_configuration['output_type'] + '/dof' + str(num_joint)
            if os.path.exists(folder_path):
                print(f"Folder {folder_path} already exists. Do you want to delete it? (y/n[abort])")
                answer = input()
                if answer == 'y':
                    for file in os.listdir(folder_path):
                        os.remove(os.path.join(folder_path, file))
                else:
                    print("Exiting...")
                    return

    summary_map = {}
    for num_joint in num_joints_to_test:
        print(f"Updating number_of_joints to {num_joint} and starting benchmarks...")
        update_number_of_joints(num_joint)
        for run_configuration in tqdm(configurations, desc="Processing output types",file=sys.stdout):
            # skip if run_configuration['output_type'] is NormalDistrMuDistanceNetworkBase, NormalDistrRandomSampleDistNetwork or NormalDistrGroundTruthLossNetwork
            if num_joint == 3 and run_configuration['output_type'] in ['NormalDistrMuDistanceNetworkBase', 'NormalDistrRandomSampleDistNetwork', 'NormalDistrGroundTruthLossNetwork']:
                # TODO remove this if statement when the models are trained
                print(f"Skipping {run_configuration['output_type']} for 3 joints as it is already recorded.")
                continue
            print(f"Running benchmark for output type: {run_configuration['output_type']} and number of joints: {num_joint}")
            folder_path = save_folder_path_prefix + '/' + run_configuration['output_type'] + '/dof' + str(num_joint)
            update_conf_file(run_configuration, folder_path)
            
            runtimes = run_benchmark_script()
            # Reverse runtimes list
            runtimes = runtimes[::-1]

            loss_acc_file_list = test_models_in_folder(model_folder_path=folder_path, num_of_joints=num_joint, test_set_length=test_length)
            # Combine entries of loss_acc_file_list with runtimes since both have the same length
            summary_map[(run_configuration['output_type'], num_joint)] = [(a, b, c, d) for (a, b, c), d in zip(loss_acc_file_list, runtimes)]
            
    # Print summary of comparison_results
    print("\nSummary of Results:")
    for (output_type, num_joint), results in summary_map.items():
        print(f"\nOutput Type: {output_type}, Number of Joints: {num_joint}")
        for i, (model_file, loss, acc, runtime) in enumerate(results, 1):
            print(f"\tLoss: {loss:.4f}, Accuracy: {acc:.4f}, Runtime: {runtime:.4f} seconds, Model File: {model_file}")
        print(f"\tLosses: {[loss for _, loss, _, _ in results]}, Mean Loss: {sum([loss for _, loss, _, _ in results]) / len(results):.4f}")
        print(f"\tAccuracies: {[acc for _, _, acc, _ in results]}, Mean Accuracy: {sum([acc for _, _, acc, _ in results]) / len(results):.4f}")
        results = results[::-1]
        print(f"\tRuntimes: {[runtime for _, _, _, runtime in results]}, Mean Runtime: {sum([runtime for _, _, _, runtime in results]) / len(results):.4f} seconds")

def update_conf_file(run_configuration, folder_path):
    with open(hyperparameters_config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    for key, value in run_configuration.items():
        config['analytical'][key] = value
    with open(hyperparameters_config_file_path, 'w') as file:
        yaml.safe_dump(config, file)

    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    config['model_save_dir'] = folder_path
    with open(config_file_path, 'w') as file:
        yaml.safe_dump(config, file)


def update_number_of_joints(number_of_joints):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    config['number_of_joints'] = number_of_joints

    with open(config_file_path, 'w') as file:
        yaml.safe_dump(config, file)


if __name__ == "__main__":
    run_various_configurations()
