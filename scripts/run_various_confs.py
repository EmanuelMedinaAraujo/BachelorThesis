import os

import yaml
from tqdm import tqdm

from scripts.benchmark_script import execute_main_script
from scripts.test_model_file import test_models_in_folder

# List of output types to iterate over
output_types = [
    "SimpleKinematicsNetwork",
    "BetaDistrRSampleMeanNetwork",
]

num_joints_to_test = [2,3]
save_folder_path_prefix = 'outputs/model_save_files/benchmark'

# Path to the configuration file
hyperparameters_config_file_path = 'conf/hyperparams/hyperparams.yaml'
config_file_path = 'conf/config.yaml'

number_of_repeats = 2

def run_benchmark_script():
    script_folder = os.path.dirname(os.path.abspath(__file__))
    benchmark_script = os.path.join(script_folder, "benchmark_script.py")

    if not os.path.isfile(benchmark_script):
        print("Error: 'benchmark_script.py' not found in the current folder.")
        return []

    return execute_main_script(number_of_repeats)


def run_various_configurations():
    summary_map = {}
    for num_joint in num_joints_to_test:
        print(f"\nUpdating number_of_joints to {num_joint} and starting benchmarks...")
        update_number_of_joints(num_joint)
        for output_type in tqdm(output_types, desc="Processing output types"):
            print(f"Running benchmark for output type: {output_type} and number of joints: {num_joint}")
            folder_path = save_folder_path_prefix + '/' + output_type + '/dof' + str(num_joint)
            if os.path.exists(folder_path):
                # Remove existing contents of the folder
                for file in os.listdir(folder_path):
                    os.remove(os.path.join(folder_path, file))
            update_conf_file(output_type, folder_path)
            
            runtimes = run_benchmark_script()
            # Reverse runtimes list
            runtimes = runtimes[::-1]

            loss_acc_file_list = test_models_in_folder(model_folder_path=folder_path, num_of_joints=num_joint)
            # Combine entries of loss_acc_file_list with runtimes since both have the same length
            summary_map[(output_type, num_joint)] = [(a, b, c, d) for (a, b, c), d in zip(loss_acc_file_list, runtimes)]
            
    # Print summary of results
    print("\nSummary of Results:")
    for (output_type, num_joint), results in summary_map.items():
        print(f"\nOutput Type: {output_type}, Number of Joints: {num_joint}")
        for i, (model_file, loss, acc, runtime) in enumerate(results, 1):
            print(f"\tLoss: {loss:.4f}, Accuracy: {acc:.4f}, Runtime: {runtime:.4f} seconds, Model File: {model_file}")
        # Print only loss from results as a list
        print(f"\tLosses: {[loss for _, loss, _, _ in results]}, Mean Loss: {sum([loss for _, loss, _, _ in results]) / len(results):.4f}")
        # Print only runtimes from results as a list
        print(f"\tRuntimes: {results[-1][3]}, Mean Runtime: {sum([runtime for _, _, _, runtime in results]) / len(results):.4f} seconds")
        # Print only accuracies from results as a list
        print(f"\tAccuracies: {[acc for _, _, acc, _ in results]}, Mean Accuracy: {sum([acc for _, _, acc, _ in results]) / len(results):.4f}")

def update_conf_file(output_type, folder_path):
    with open(hyperparameters_config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    config['analytical']['output_type'] = output_type
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
