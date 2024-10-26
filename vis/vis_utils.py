import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from util.forward_kinematics import calculate_euclidean_distances


def set_plot_settings(parameter):
    # General settings
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.axhline(0, color="black", lw=1.5)
    ax.axvline(0, color="black", lw=1.5)
    max_length, multiple_robots = compute_max_robot_length(parameter)
    # Set the limits of the plot
    ax.set_xlim(-max_length, max_length)
    ax.set_ylim(-max_length, max_length)
    return ax, multiple_robots


def plot_goal_and_configure_legend(ax, goal, max_legend_length, show_legend):
    if goal is not None:
        x, y = goal[0].item(), goal[1].item()
        ax.plot(x, y, "-x", label=f"Robot Goal [{x:>0.1f},{y:>0.1f}]")
    if show_legend:
        # Only show first max_legend_length entries
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > max_legend_length:
            labels = [labels.pop()] + labels[-max_legend_length - 1:]
            handles = [handles.pop()] + handles[-max_legend_length - 1:]
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
    plt.tight_layout()


def compute_max_robot_length(parameter):
    # Calculate the maximum length of the robot arm from the given parameters
    max_length = 0
    multiple_robots = len(parameter.size()) == 3
    if multiple_robots:
        for i in range(parameter.shape[1]):
            max_length += parameter[0, i, 1].item()
    else:
        for i in range(parameter.shape[0]):
            max_length += parameter[i, 1].item()
    return max_length, multiple_robots


def plot_line(ax, color, default_line_width, link_length, start_coordinates, total_angle,
              transparency):
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
    return end_coordinates, link_line


def plot_distribution_single_link(
        ax,
        link_length,
        angle,
        default_line_width,
        device,
        mark_as_best_end_effector=False,
        transparency=1.0,
        show_distance=False,
        goal=None,
        start_point=np.array([0, 0]),
        start_angle=0.0,
        color='b',
):
    total_angle = start_angle + angle

    end_coordinates, link_line = plot_line(ax, color, default_line_width, link_length,
                                           start_point, total_angle, transparency)

    # Add a label only once for each robot arm, since this is only executed for the first link
    if show_distance:
        distance_label = ""
        if show_distance and goal is not None:
            end_coordinates = torch.tensor(end_coordinates).to(device)
            goal = goal.clone().detach().to(device)
            distance = calculate_euclidean_distances(end_coordinates, goal)
            distance_label = f"({distance:.2f})"
        probability_label = f"[{transparency:.1f}]"
        link_line.set_label(
            f"Robot Arm" + distance_label + probability_label
        )

    if mark_as_best_end_effector:
        # Plot end effector
        ax.plot(
            end_coordinates[0].item(),
            end_coordinates[1].item(),
            "-o",
            markersize=3,
            color=color,
            alpha=0.5,
        )


def finish_and_close_plot(ax, chart_index, current_step, goal, logger, max_legend_length, save_to_file, show_legend,
                          show_plot):
    plot_goal_and_configure_legend(ax, goal, max_legend_length, show_legend)

    if logger is not None:
        logger.log_image(plt, current_step, "chart" + str(chart_index))
    if save_to_file:
        save_plot("outputs/vis/plots/", "chart" + str(chart_index))
    if show_plot:
        plt.show()
    plt.close()


def save_plot(parent_folder, filename):
    # Get day and time for the filename
    day_time = str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    if parent_folder[-1] != "/":
        parent_folder += "/"
    if not os.path.isdir(parent_folder):
        os.makedirs(parent_folder)
    plt.savefig(parent_folder + filename + "_" + day_time + ".png")
