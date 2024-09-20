from datetime import datetime

import torch

from analyticalRL.kinematics_network import KinematicsNetwork
from conf.config import TrainConfig

import matplotlib.pyplot as plt
import os

from vis.planar_robot_vis import set_plot_settings, plot_planar_robot


def visualize_analytical_problem(
        model: KinematicsNetwork,
        param,
        goal,
        param_history,
        cfg: TrainConfig,
        logger=None,
        current_step=None,
):
    """
    Visualize a single robot arm with the given parameters and goal.
    """
    model.eval()
    pred = model((param, goal))

    vis_params = cfg.vis
    # Concatenate the predicted theta values to the parameter
    updated_param = torch.cat((param, pred.unsqueeze(1)), dim=-1)
    param_history.append(updated_param)

    # Check if max history length is reached
    if len(param_history) > vis_params.analytical.max_history_length:
        param_history.pop(0)

    tensor_to_pass = (
        updated_param if not vis_params.analytical.plot_all_in_one else torch.stack(param_history)
    )

    visualize_analytical_planar_robot(
        parameter=tensor_to_pass,
        goal=goal,
        save_to_file=vis_params.save_to_file,
        default_line_transparency=vis_params.analytical.default_line_transparency,
        default_line_width=vis_params.analytical.default_line_width,
        use_gradual_transparency=vis_params.analytical.use_gradual_transparency,
        show_plot=vis_params.show_plot,
        show_joints=vis_params.show_joints,
        show_end_effectors=vis_params.show_end_effectors,
        show_joint_label=vis_params.show_joint_label,
        show_distance=vis_params.show_distance_in_legend,
        max_legend_length=vis_params.max_legend_length,
        logger=logger if cfg.logging.wandb.log_visualization_plots else None,
        current_step=current_step,
        show_legend=vis_params.show_legend
    )

    model.train()

def visualize_analytical_planar_robot(
        parameter,
        default_line_transparency=1.,
        default_line_width=1.,
        max_legend_length = 10,
        use_gradual_transparency=False,
        goal=None,
        save_to_file=False,
        show_joints=False,
        show_joint_label=True,
        show_end_effectors=False,
        show_plot=True,
        robot_label_note="",
        show_distance=False,
        logger=None,
        current_step=None,
        show_legend=True
):
    """
    Visualize one or multiple planar robot arms based on the given parameters using matplotlib.
    Args:
        parameter: The parameters of the robot arm(s).
        default_line_transparency: The default transparency of the robot arm links.
        default_line_width: The default line width of the robot arm links.
        use_gradual_transparency: If True, the transparency of the robot arm links will be gradually increased.
        goal: The goal of the robot arm(s).
        save_to_file: If True, the plot will be saved to a file.
        show_end_effectors: If True, the end effector will be shown in the plot.
        show_joints: If True, the joints will be plotted.
        show_joint_label: If True, the joint labels will be shown in the legend.
        show_plot: If True, a plot will be opened
        robot_label_note: A string to add to each robot label.
        show_distance: If True, the distance to the goal will be shown in the legend.
        max_legend_length: The maximum length of the legend.
        logger: The logger to use for plot custom_logging.
        current_step: The current step of the training.
        show_legend: If True, the legend will be shown.
    """
    ax, multiple_robots = set_plot_settings(parameter)

    # Check whether multiple robot arms were given or only one
    if multiple_robots:
        # Multiple Robots arm were given as parameter
        for i in range(parameter.shape[0]):
            if use_gradual_transparency:
                # Calculate the transparency based on the number of robot arms
                transparency_steps = 1 / parameter.shape[0]
                transparency = 0.1 + transparency_steps * (i+1)
                if transparency > 1:
                    transparency = 1
            else:
                transparency = default_line_transparency
            # Plot the planar robot
            plot_planar_robot(
                ax=ax,
                parameter=parameter[i],
                default_line_width=default_line_width,
                transparency=transparency,
                show_joint_label=show_joint_label,
                show_joints=show_joints,
                show_end_effectors=show_end_effectors,
                robot_label_note=str(i)+robot_label_note,
                show_distance=show_distance,
                goal=goal,
            )
    else:
        # One Robot arm was given as parameter
        plot_planar_robot(
            ax=ax,
            parameter=parameter,
            default_line_width=default_line_width,
            transparency=default_line_transparency,
            show_joint_label=show_joint_label,
            show_joints=show_joints,
            show_end_effectors=show_end_effectors,
            robot_label_note=robot_label_note,
            show_distance=show_distance,
            goal=goal
        )

    # Plot the goal of the robot
    if goal is not None:
        x, y = goal[0].item(), goal[1].item()
        ax.plot(x, y, "-x", label=f"Robot Goal [{x:>0.1f},{y:>0.1f}]")

    if show_legend:
        # Only show first max_legend_length entries
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > max_legend_length:
            labels = [labels.pop()]+ labels[-max_legend_length-1:]
            handles = [handles.pop()] + handles[-max_legend_length-1:]
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )

    plt.tight_layout()

    if logger is not None:
        logger.log_plot(plt, current_step)

    if save_to_file:
        # Get day and time for the filename
        day_time = str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        path = os.path.join("../plots", f"rbt_plt_{day_time}.png")
        plt.savefig(path)

    if show_plot:
        plt.show()
    plt.close()
