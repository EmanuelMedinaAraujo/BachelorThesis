from datetime import datetime

import torch

import seaborn as sns
from analyticalRL.kinematics_network import KinematicsNetwork
from conf.config import TrainConfig

import math

import matplotlib.pyplot as plt
import numpy as np
import os

from util.forward_kinematics import calculate_distances


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


def plot_analytical_planar_robot(
        ax,
        parameter,
        default_line_width,
        transparency=1.0,
        show_distance = False,
        show_joint_label=True,
        show_joints=True,
        show_end_effectors=False,
        robot_label_note="",
        goal=None,
):
    """
    Plot a planar robot arm based on the given parameters.
    Args:
        ax: The axis to plot the robot arm on.
        parameter: The parameters of the robot arm. The parameters are expected to be in the format
            [alpha, a, d, theta] with alpha and d being 0.
        default_line_width: The default line width of the robot arm links.
        show_joint_label: If True, the joint labels will be shown in the legend
        show_joints: If True, the joints will be plotted.
        show_end_effectors: If True, the end effector will be shown in the plot.
        robot_label_note: A string to add to the robot label.
        show_distance: If True, the distance to the goal will be shown in the legend.
        goal: The goal of the robot arm. Is needed to calculate the distance to the goal.
        transparency: The transparency of the robot arm links.
    Returns: The maximum length of the robot arm.
    """
    start_coordinates = np.array([0, 0])
    end_coordinates = np.array([0, 0])
    total_angle = 0.0

    # Plot the robot links
    color = 'r'
    for i in range(parameter.shape[0]):
        link_length = parameter[i, 1].item()
        link_angle = parameter[i, 3].item()
        total_angle += link_angle

        end_coordinates = start_coordinates + np.array(
            [link_length * math.cos(total_angle), link_length * math.sin(total_angle)]
        )

        # Plot single link
        (link_line,) = ax.plot(
            [start_coordinates[0], end_coordinates[0]],
            [start_coordinates[1], end_coordinates[1]],
            lw=default_line_width,
            alpha=transparency,
            color=color
        )

        # Add a label only once for each robot arm, since this is only executed for the first link
        if i == 0:
            distance_label = ""
            if show_distance and goal is not None:
                distance = calculate_distances(parameter, goal)
                distance_label = f"({distance:.2f})"
            link_line.set_label(
                f"Robot Arm {robot_label_note}" + distance_label
            )

        # Plot joint
        if show_joints:
            joint_label = (
                f"$\\theta$={np.rad2deg(link_angle):.1f}\N{DEGREE SIGN}, L={link_length:.2f}"
                if show_joint_label
                else None
            )
            ax.plot(
                start_coordinates[0],
                start_coordinates[1],
                "-o",
                label=joint_label,
                markersize=1,
                color=color,
                alpha=transparency,
            )
        start_coordinates = end_coordinates

    if show_end_effectors:
        # Plot end effector
        ax.plot(
            end_coordinates[0],
            end_coordinates[1],
            "-o",
            markersize=1,
            color=color,
            alpha=transparency,
        )

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
                transparency = 0.1 + transparency_steps * i+1
                if transparency > 1:
                    transparency = 1
            else:
                transparency = default_line_transparency
            # Plot the planar robot
            plot_analytical_planar_robot(
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
        plot_analytical_planar_robot(
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
        # Save the plot to a file temporarily
        plt.savefig("temp_plot.png")
        logger.log_plot("temp_plot.png", current_step)
        # Delete the temporary file
        os.remove("temp_plot.png")

    if save_to_file:
        # Get day and time for the filename
        day_time = str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        path = os.path.join("../plots", f"rbt_plt_{day_time}.png")
        plt.savefig(path)

    if show_plot:
        plt.show()


def set_plot_settings(parameter):
    # General settings
    sns.set(style="whitegrid")
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.axhline(0, color="black", lw=1.5)
    ax.axvline(0, color="black", lw=1.5)
    # Calculate the maximum length of the robot arm from the given parameters
    max_length = 0
    multiple_robots = len(parameter.size()) == 3
    if multiple_robots:
        for i in range(parameter.shape[1]):
            max_length += parameter[0, i, 1].item()
    else:
        for i in range(parameter.shape[0]):
            max_length += parameter[i, 1].item()
    # Set the limits of the plot
    ax.set_xlim(-max_length, max_length)
    ax.set_ylim(-max_length, max_length)
    return ax, multiple_robots
