import math

import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_planar_robot(parameter, default_line_transparency, default_line_width, frame_size_scalar, device='cpu',
                           use_color_per_robot=False, goal=None, link_accuracy=None, standard_size=False,
                           save_to_file=False, show_joint_label=True, show_plot=True, robot_label_note=""):
    multiple_robots = len(parameter.size()) == 3
    if link_accuracy is None:
        if multiple_robots:
            # Create a tensor of length parameter.size(dim=0) with default_line_transparency as values
            link_accuracy = torch.full([parameter.size(dim=0)], default_line_transparency).to(device)
        else:
            link_accuracy = torch.full([1], default_line_transparency).to(device)
    else:
        assert parameter.shape[0] == link_accuracy.shape[
            0], "The amount of robots arm and the amount of probabilities is not equal"

    # General settings
    fig, ax = plt.subplots()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.axhline(0, color='black', lw=1.5)
    ax.axvline(0, color='black', lw=1.5)

    if goal is not None:
        x, y = goal[0].item(), goal[1].item()
        ax.plot(x, y, '-x', label=f"Robot Goal [{x:>0.1f},{y:>0.1f}]")

    max_length = 0

    # Check whether multiple robot arms were given or only one
    if multiple_robots:
        # Multiple Robots arm were given as parameter
        for i in range(parameter.shape[0]):
            # Plot the planar robot
            arm_length = plot_planar_robot(ax, parameter[i], link_accuracy[i].item(), default_line_width,
                                           use_color_per_robot, robot_num=i + 1, show_joint_label=show_joint_label,
                                           robot_label_note=robot_label_note)
            max_length = max(arm_length, max_length)
    else:
        # One Robot arm was given as parameter
        max_length = plot_planar_robot(ax, parameter, link_accuracy.item(), default_line_width,
                                       show_joint_label=show_joint_label)

    # Set the plot size
    set_plot_limits(ax, max_length, standard_size, frame_size_scalar)

    if standard_size:
        ax.legend()
    else:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
    if save_to_file:
        plt.savefig('')
    if show_plot:
        plt.show()


def plot_planar_robot(ax, parameter, link_accuracy, default_line_width, use_color_per_robot=False, robot_num=1,
                      show_joint_label=True, robot_label_note=""):
    start_coordinates = np.array([0, 0])
    max_length = 0.0
    total_angle = 0.0

    color = None
    for i in range(parameter.shape[0]):
        link_length = parameter[i, 1].item()
        max_length += link_length

        link_angle = parameter[i, 3].item()
        total_angle += link_angle

        end_coordinates = start_coordinates + np.array(
            [link_length * math.cos(total_angle), link_length * math.sin(total_angle)])

        # Plot link
        link_line, = ax.plot([start_coordinates[0], end_coordinates[0]], [start_coordinates[1], end_coordinates[1]],
                             color=color if use_color_per_robot else 'r', lw=default_line_width, alpha=link_accuracy)
        if color is None:
            color = link_line.get_color()

            link_line.set_label(f"Robot Arm {robot_num} " + robot_label_note)
        joint_label = f'$\\theta$={np.rad2deg(link_angle):.1f}\N{DEGREE SIGN}, L={link_length:.2f}' if show_joint_label else None
        ax.plot(start_coordinates[0], start_coordinates[1], '-o', label=joint_label)
        start_coordinates = end_coordinates
    return max_length


def set_plot_limits(ax, max_length, standard_size, frame_size_scalar):
    if standard_size:
        # Ensure that the entire robot is visible by setting the limits as the maximum length of the robot
        # Show the same size for every arm position
        max_length *= frame_size_scalar
        ax.set_xlim(-max_length, max_length)
        ax.set_ylim(-max_length, max_length)
    else:
        # Only show relevant part with a robot arm
        x_max_limit = abs(max(ax.get_xlim(), key=abs)) * frame_size_scalar
        ax.set_xlim(-x_max_limit, x_max_limit)
