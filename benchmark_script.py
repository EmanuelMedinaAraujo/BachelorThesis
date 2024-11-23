import os
import subprocess
import time

number_of_repeats = 2


def execute_main_script():
    # Get the path to main.py in the same folder as this script
    script_folder = os.path.dirname(os.path.abspath(__file__))
    main_file = os.path.join(script_folder, "main.py")

    if not os.path.isfile(main_file):
        print("Error: 'main.py' not found in the current folder.")
        return

    runtimes = []
    for i in range(number_of_repeats):
        start_time = time.perf_counter()
        try:
            subprocess.run(["python3", main_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Execution failed during run {i + 1}: {e}")
        end_time = time.perf_counter()

        runtime = end_time - start_time
        runtimes.append(runtime)

    mean_runtime = sum(runtimes) / len(runtimes) if runtimes else 0
    print("\nExecution Summary:")
    for i, runtime in enumerate(runtimes, 1):
        print(f"Run {i}: {runtime:.4f} seconds")
    print(f"Mean Runtime: {mean_runtime:.4f} seconds")


if __name__ == "__main__":
    execute_main_script()
