import os
import subprocess
import yaml
import sys
from tqdm import tqdm

# List of output types to iterate over
output_types = [
    "SimpleKinematicsNetwork"
]
new_number_of_joints = 3

# Path to the configuration file
hyperparameters_config_file_path = 'conf/hyperparams/hyperparams.yaml'
config_file_path = 'conf/config.yaml'

log_file_path = 'benchmark_log.txt'


def update_output_type(output_type):
    with open(hyperparameters_config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    config['analytical']['output_type'] = output_type

    with open(hyperparameters_config_file_path, 'w') as file:
        yaml.safe_dump(config, file)


def update_number_of_joints(number_of_joints):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    config['number_of_joints'] = number_of_joints

    with open(config_file_path, 'w') as file:
        yaml.safe_dump(config, file)


def run_benchmark_script():
    script_folder = os.path.dirname(os.path.abspath(__file__))
    benchmark_script = os.path.join(script_folder, "benchmark_script.py")

    if not os.path.isfile(benchmark_script):
        print("Error: 'benchmark_script.py' not found in the current folder.")
        return

    try:
        with open(log_file_path, 'a') as log_file:
            process = subprocess.Popen(["python3", benchmark_script], stdout=sys.stdout, stderr=sys.stderr)
            process.wait()
    except subprocess.CalledProcessError as e:
        print(f"Execution failed: {e}")


def main():
    for output_type in tqdm(output_types, desc="Processing output types"):
        print(f"\nRunning benchmark for output type: {output_type}")
        update_output_type(output_type)
        run_benchmark_script()

    print("\nUpdating number_of_joints to 3 and restarting benchmarks...")
    update_number_of_joints(new_number_of_joints)

    for output_type in tqdm(output_types, desc="Processing output types with 3 joints"):
        print(f"\nRunning benchmark for output type: {output_type}")
        update_output_type(output_type)
        run_benchmark_script()


if __name__ == "__main__":
    main()