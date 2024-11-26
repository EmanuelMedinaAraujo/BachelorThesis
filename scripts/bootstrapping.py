import numpy as np
from scipy.stats import bootstrap


def calc_bootsstrap_runtimes():
    dof2_list = [252.3367774319995, 248.48100662900015, 247.26871795999978, 246.11029099900043, 247.08818190199963]
    dof3_list = [304.34515529499913, 302.02081129399994, 305.8172302900002, 302.0463187739988, 300.0565301379993]

    # Example data: runtimes for 2 DOF and 3 DOF
    runtime_2dof = np.array(dof2_list)
    runtime_3dof = np.array(dof3_list)

    # Bootstrapping the mean runtime for 2 DOF
    result_2dof = bootstrap((runtime_2dof,), np.mean, confidence_level=0.95, n_resamples=10000)
    # Bootstrapping the mean runtime for 3 DOF
    result_3dof = bootstrap((runtime_3dof,), np.mean, confidence_level=0.95, n_resamples=10000)

    ci_interval2 = tuple(map(convert_seconds, result_2dof.confidence_interval))
    print("2 DOF:", convert_seconds(runtime_2dof.mean()), f"({ci_interval2[0]}, {ci_interval2[1]})")

    ci_interval3 = tuple(map(convert_seconds, result_3dof.confidence_interval))
    print("3 DOF:", convert_seconds(runtime_3dof.mean()), f"({ci_interval3[0]}, {ci_interval3[1]})")

def calc_bootsstrap_losses():
    loss_list_2dof = [0.012120698657458706, 0.012109289352946508, 0.01208666185123293, 0.012091975383381942, 0.012110781074671832]
    loss_list_3dof =  [0.010375037568945845, 0.010363861933567387, 0.010357808310736436, 0.01037748655596588, 0.010352261379240007]
    losses_2dof = np.array(loss_list_2dof)
    losses_3dof = np.array(loss_list_3dof)

    result_2dof = bootstrap((losses_2dof,), np.mean, confidence_level=0.95, n_resamples=10000)
    result_3dof = bootstrap((losses_3dof,), np.mean, confidence_level=0.95, n_resamples=10000)

    ci_interval2 = tuple(map(lambda x: f"{x:.6f}", result_2dof.confidence_interval))
    ci_interval3 = tuple(map(lambda x: f"{x:.6f}", result_3dof.confidence_interval))
    print("2 DOF:", f"{losses_2dof.mean():.6f}", f"({ci_interval2[0]}, {ci_interval2[1]})")
    print("3 DOF:", f"{losses_3dof.mean():.6f}", f"({ci_interval3[0]}, {ci_interval3[1]})")

def calc_bootsstrap_accuracies():
    acc_list_2dof =  [46.28, 46.28, 46.42, 46.6, 46.34]
    acc_list_3dof =  [60.97, 60.74, 60.99, 61.11, 61.11]
    acc_2dof = np.array(acc_list_2dof)
    acc_3dof = np.array(acc_list_3dof)

    result_2dof = bootstrap((acc_2dof,), np.mean, confidence_level=0.95, n_resamples=10000)
    result_3dof = bootstrap((acc_3dof,), np.mean, confidence_level=0.95, n_resamples=10000)

    ci_interval2 = tuple(map(lambda x: f"{x:.2f}", result_2dof.confidence_interval))
    ci_interval3 = tuple(map(lambda x: f"{x:.2f}", result_3dof.confidence_interval))
    print("2 DOF:", f"{acc_2dof.mean():.2f}", f"({ci_interval2[0]}, {ci_interval2[1]})")
    print("3 DOF:", f"{acc_3dof.mean():.2f}", f"({ci_interval3[0]}, {ci_interval3[1]})")


def convert_seconds(seconds):
    # Calculate hours, minutes, and seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    if hours == 0:
        return f"{minutes:.0f}m {remaining_seconds:.0f}s"

    # Format the result as "hh:mm:ss"
    return f"{hours:.0f}h {minutes:.0f}m {remaining_seconds:.0f}s"


if __name__ == "__main__":
    calc_bootsstrap_accuracies()
