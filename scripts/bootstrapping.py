import numpy as np
from scipy.stats import bootstrap


def calc_bootsstrap_runtimes(dof2_list, dof3_list=None):
    # dof2_list = [381.63827980299993, 378.5175722220001, 377.1377383529998, 377.2592813629999, 375.89905818800025]
    # dof3_list = [453.84056038000017, 454.61482982100006, 449.8340852269994, 450.11251080199963, 449.8690109399995]

    # Example data: runtimes for 2 DOF and 3 DOF
    runtime_2dof = np.array(dof2_list)
    if dof3_list is not None:
        runtime_3dof = np.array(dof3_list)

    # Bootstrapping the mean runtime for 2 DOF
    result_2dof, result_3dof = bootstap_two_lists(dof2_list, dof3_list, 2)
    ci_interval2 = tuple(map(convert_seconds, result_2dof.confidence_interval))
    print(convert_seconds(runtime_2dof.mean()), f"({ci_interval2[0]}, {ci_interval2[1]})")

    if dof3_list is not None:
        ci_interval3 = tuple(map(convert_seconds, result_3dof.confidence_interval))
        print(convert_seconds(runtime_3dof.mean()), f"({ci_interval3[0]}, {ci_interval3[1]})")


def calc_bootsstrap_losses(loss_list_2dof, loss_list_3dof=None):
    # loss_list_2dof = [0.004390149802332598, 0.0043901365686146165, 0.004390156978155937, 0.004390155097571915, 0.004390138595167809]
    # loss_list_3dof = [0.00848811249920618, 0.008488229663586026, 0.008488159015898783, 0.00848805751567561, 0.00848816524681024]
    bootstap_two_lists(loss_list_2dof, loss_list_3dof, 2)


def calc_bootsstrap_accuracies(acc_list_2dof, acc_list_3dof=None):
    # acc_list_2dof = [97.5, 97.5, 97.5, 97.5, 97.5]
    # acc_list_3dof = [75.6, 75.59, 75.59, 75.59, 75.6]
    bootstap_two_lists(acc_list_2dof, acc_list_3dof, 2)


def bootstap_two_lists(loss_list_2dof, loss_list_3dof, after_decimal):
    losses_2dof = np.array(loss_list_2dof)
    result_2dof = bootstrap((losses_2dof,), np.mean, confidence_level=0.95, n_resamples=10000)
    ci_interval2 = tuple(map(lambda x: f"{x:.{after_decimal}f}", result_2dof.confidence_interval))
    print(f"{losses_2dof.mean():.{after_decimal}f}", f"({ci_interval2[0]}, {ci_interval2[1]})")

    if loss_list_3dof is not None:
        losses_3dof = np.array(loss_list_3dof)
        result_3dof = bootstrap((losses_3dof,), np.mean, confidence_level=0.95, n_resamples=10000)
        ci_interval3 = tuple(map(lambda x: f"{x:.{after_decimal}f}", result_3dof.confidence_interval))
        print(f"{losses_3dof.mean():.{after_decimal}f}", f"({ci_interval3[0]}, {ci_interval3[1]})")
        return result_2dof, result_3dof
    return result_2dof, None

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
    dof2_list= [302.9289, 300.2082, 304.0576, 300.2754, 302.7790]
    dof3_list = [784.9076, 778.0595, 787.7320, 789.4031, 795.0810]

    # Uncomment when using losses
    #dof2_list=[loss * 1000 for loss in dof2_list]
    #dof3_list=[loss * 1000 for loss in dof3_list]

    calc_bootsstrap_runtimes(dof2_list,dof3_list)

