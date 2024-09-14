import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from Util.forward_kinematics import calculate_distances


def visualize_planar_robot(parameter, default_line_transparency, default_line_width, frame_size_scalar,
                           max_legend_length, use_gradual_transparency=False, device='cpu',
                           use_color_per_robot=False, goal=None, link_accuracy=None, standard_size=False,
                           save_to_file=False, show_joints=False, show_joint_label=True, show_end_effectors=False, show_plot=True, robot_label_note="",
                           show_distance=False, logger=None):
    """
    Visualize one or multiple planar robot arms based on the given parameters using matplotlib.
    Args:
        parameter: The parameters of the robot arm(s).
        default_line_transparency: The default transparency of the robot arm links.
        default_line_width: The default line width of the robot arm links.
        frame_size_scalar: The scalar to multiply the frame size with.
        use_gradual_transparency: If True, the transparency of the robot arm links will be gradually increased.
        device: The device to use for torch operations.
        use_color_per_robot: If True, the robot arm links will be colored based on the robot number.
        goal: The goal of the robot arm(s).
        link_accuracy: The transparency of the robot arm links. If None, the default_line_transparency will be used.
        standard_size: If True, the plot size will be set to the maximum theoretical length of the robot arm.
        save_to_file: If True, the plot will be saved to a file.
        show_end_effectors: If True, the end effector will be shown in the plot.
        show_joints: If True, the joints will be plotted.
        show_joint_label: If True, the joint labels will be shown in the legend.
        show_plot: If True, a plot will be opened
        robot_label_note: A string to add to each robot label.
        show_distance: If True, the distance to the goal will be shown in the legend.
        max_legend_length: The maximum length of the legend.
        logger: The logger to use for plot logging.
    """
    multiple_robots = len(parameter.size()) == 3
    if link_accuracy is None:
        if multiple_robots:
            # Create a tensor of length parameter.size(dim=0) with default_line_transparency as values
            link_accuracy = torch.full([parameter.size(dim=0)], default_line_transparency).to(device)
        else:
            link_accuracy = torch.full([1], default_line_transparency).to(device)
    else:
        assert parameter.shape[0] == link_accuracy.shape[
            0], "The amount of robots arms and the amount of probabilities is not equal"

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
            arm_length = plot_planar_robot(ax=ax,
                                           parameter=parameter[i],
                                           link_accuracy=link_accuracy[i].item(),
                                           default_line_width=default_line_width,
                                           use_color_per_robot=use_color_per_robot,
                                           robot_num=(i + 1, parameter.shape[0]),
                                           show_joint_label=show_joint_label,
                                           show_joints=show_joints,
                                           show_end_effectors=show_end_effectors,
                                           robot_label_note=robot_label_note,
                                           use_gradual_transparency=use_gradual_transparency,
                                           show_distance=show_distance,
                                           goal=goal)
            max_length = max(arm_length, max_length)
    else:
        # One Robot arm was given as parameter
        max_length = plot_planar_robot(ax=ax,
                                       parameter=parameter,
                                       link_accuracy=link_accuracy.item(),
                                       default_line_width=default_line_width,
                                       show_joint_label=show_joint_label,
                                       show_joints=show_joints,
                                        show_end_effectors=show_end_effectors,
                                       show_distance=show_distance,
                                       goal=goal)

    # Set the plot size
    set_plot_limits(ax, max_length, standard_size, frame_size_scalar)

    # Only show first 20 entries and '...' if there are more
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:max_legend_length],
              labels[:max_legend_length] + (['...'] if len(labels) > max_legend_length else []), loc='upper left',
              bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if logger is not None:
        # Save the plot to a file temporarily
        plt.savefig('temp_plot.png')
        logger.log_plot('temp_plot.png')
        # Delete the temporary file
        os.remove('temp_plot.png')
    if save_to_file:
        plt.savefig('')
    if show_plot:
        plt.show()


def plot_planar_robot(ax, parameter, link_accuracy,
                      default_line_width,
                      show_distance,
                      use_gradual_transparency=False,
                      use_color_per_robot=False,
                      robot_num=(1, 1),
                      show_joint_label=True,
                      show_joints=True,
                      show_end_effectors=False,
                      robot_label_note="",
                      goal=None):
    """
    Plot a planar robot arm based on the given parameters.
    Args:
        ax: The axis to plot the robot arm on.
        parameter: The parameters of the robot arm. The parameters are expected to be in the format
            [alpha, a, d, theta] with alpha and d being 0.
        link_accuracy: The transparency of the robot arm links.
        default_line_width: The default line width of the robot arm links.
        use_gradual_transparency: If True, the transparency of the robot arm links will be gradually increased.
        use_color_per_robot: If True, the robot arm links will be colored based on the robot number.
        robot_num: The number of the robot arm and the total number of robot arms.
        show_joint_label: If True, the joint labels will be shown in the legend
        show_joints: If True, the joints will be plotted.
        show_end_effectors: If True, the end effector will be shown in the plot.
        robot_label_note: A string to add to the robot label.
        show_distance: If True, the distance to the goal will be shown in the legend.
        goal: The goal of the robot arm. Is needed to calculate the distance to the goal.
    Returns: The maximum length of the robot arm.
    """
    start_coordinates = np.array([0, 0])
    end_coordinates = np.array([0, 0])
    max_length = 0.0
    total_angle = 0.0

    transparency = link_accuracy
    if use_gradual_transparency:
        transparency_steps = 0.9 / robot_num[1]
        transparency = 0.1 + transparency_steps * robot_num[0]
        if transparency > 1:
            transparency = 1

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
                             color=color if use_color_per_robot else 'r', lw=default_line_width, alpha=transparency)
        # Set the color for the next link to the color of the current link
        if color is None:
            color = link_line.get_color()
            # Add a label only once for each robot arm, since this is only executed for the first link
            distance_label = ""
            if show_distance and goal is not None:
                distance = calculate_distances(parameter, goal)
                distance_label = f"({distance:.2f})"
            link_line.set_label(f"Robot Arm {robot_num[0]}" + distance_label + " " + robot_label_note)
        joint_label = f'$\\theta$={np.rad2deg(link_angle):.1f}\N{DEGREE SIGN}, L={link_length:.2f}' if show_joint_label else None
        # Plot joint
        if show_joints:
            ax.plot(start_coordinates[0], start_coordinates[1], '-o', label=joint_label, markersize=1, color=color,
                    alpha=transparency)
        start_coordinates = end_coordinates

    if show_end_effectors:
        # Plot end effector
        ax.plot(end_coordinates[0], end_coordinates[1], '-o', markersize=1,
                color=color, alpha=transparency)
    return max_length


def set_plot_limits(ax, max_length, standard_size, frame_size_scalar):
    """
    Set the plot limits for the given axis. The limits are set based on the length of the robot arm.
    If standard_size is True, the limits are set to the maximum theoretical length of the robot arm.
    If standard_size is False, the limits are set in a way that only the robot arm is visible.
    """
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
