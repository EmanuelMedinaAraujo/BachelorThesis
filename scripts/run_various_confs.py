import os

import yaml
from tqdm import tqdm

from scripts.benchmark_script import execute_main_script

# List of output types to iterate over
output_types = [
    "SimpleKinematicsNetwork"
]

first_number_of_joints = 2
updated_number_of_joints = 3

save_folder_path = 'outputs/model_save_files_perf_'

# Path to the configuration file
hyperparameters_config_file_path = 'conf/hyperparams/hyperparams.yaml'
config_file_path = 'conf/config.yaml'


def run_benchmark_script():
    script_folder = os.path.dirname(os.path.abspath(__file__))
    benchmark_script = os.path.join(script_folder, "benchmark_script.py")

    if not os.path.isfile(benchmark_script):
        print("Error: 'benchmark_script.py' not found in the current folder.")
        return []

    return execute_main_script()


def run_various_configurations():
    runtimes_map = {}
    num_joint = first_number_of_joints
    update_number_of_joints(num_joint)
    for output_type in tqdm(output_types, desc="Processing output types"):
        print(f"\nRunning benchmark for output type: {output_type}")
        update_conf_file(output_type, num_joint, save_folder_path)
        runtimes = run_benchmark_script()
        runtimes_map[(output_type, num_joint)] = runtimes

    print("\nUpdating number_of_joints to 3 and restarting benchmarks...")
    num_joint = updated_number_of_joints
    update_number_of_joints(num_joint)

    for output_type in tqdm(output_types, desc="Processing output types with 3 joints"):
        print(f"\nRunning benchmark for output type: {output_type}")
        update_conf_file(output_type, num_joint, save_folder_path)
        runtimes = run_benchmark_script()
        runtimes_map[(output_type, num_joint)] = runtimes


def update_conf_file(output_type, number_of_joints, folder_path_prefix):
    with open(hyperparameters_config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    config['analytical']['output_type'] = output_type
    config['model_save_dir']= folder_path_prefix+output_type+'/dof'+str(number_of_joints)

    with open(hyperparameters_config_file_path, 'w') as file:
        yaml.safe_dump(config, file)


def update_number_of_joints(number_of_joints):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    config['number_of_joints'] = number_of_joints

    with open(config_file_path, 'w') as file:
        yaml.safe_dump(config, file)


if __name__ == "__main__":
    run_various_configurations()
