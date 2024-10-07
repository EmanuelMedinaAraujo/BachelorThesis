from datetime import datetime

import numpy as np
import torch
from torch.utils.hipify.hipify_python import InputError

from analyticalRL.kinematics_network import KinematicsNetwork
from analyticalRL.kinematics_network_base import KinematicsNetworkBase
from conf.config import TrainConfig

import matplotlib.pyplot as plt
import os

from vis.planar_robot_vis import set_plot_settings, plot_planar_robot, plot_single_link


def plot_distribution(parameter, link_angles, ground_truth, link_probabilities, chart_index, goal, default_line_width,
                      save_to_file, show_plot, max_legend_length, logger, current_step, show_legend, device):
    ax, _ = set_plot_settings(parameter)

    # Plot the ground truth
    if ground_truth is not None:
        # Update parameter with ground truth
        updated_param = torch.cat((parameter, ground_truth), dim=-1)
        plot_planar_robot(
            ax=ax,
            parameter=updated_param,
            default_line_width=default_line_width,
            transparency=0.5,
            show_joint_label=True,
            show_joints=False,
            show_end_effectors=False,
            show_distance=False,
            goal=goal,
            robot_label_note="Ground Truth"
        )

    is_last_link = False
    start_point = [0, 0]
    start_angle = 0
    for joint_number in range(link_angles.__len__()):
        if joint_number == link_angles.__len__() - 1:
            is_last_link = True
        for link_num in range(link_angles[joint_number].__len__()):
            link_prob = link_probabilities[joint_number][link_num]
            draw_end_effector = False
            if is_last_link and link_prob == max(link_probabilities[joint_number]):
                draw_end_effector = True
            plot_single_link(ax=ax,
                             link_length=parameter[joint_number, 1].item(),
                             angle=link_angles[joint_number][link_num],
                             default_line_width=default_line_width,
                             transparency=link_probabilities[joint_number][link_num],
                             show_distance=is_last_link,
                             draw_best_end_effector=draw_end_effector,
                             goal=goal,
                             start_point=start_point,
                             start_angle=start_angle,
                             color="blue" if is_last_link else 'g',
                             device=device)
        # Get index of maximum probability in link_probabilities[joint_number]
        max_prob_index = link_probabilities[joint_number].index(max(link_probabilities[joint_number]))
        start_angle += link_angles[joint_number][max_prob_index]
        start_point = [start_point[0] + parameter[joint_number, 1].item() * np.cos(start_angle),
                       start_point[1] + parameter[joint_number, 1].item() * np.sin(start_angle)]

    # Plot the goal of the robot
    if goal is not None:
        x, y = goal[0].item(), goal[1].item()
        ax.plot(x, y, "-x", label=f"Robot Goal [{x:>0.1f},{y:>0.1f}]")

    if show_legend:
        # Only show last max_legend_length entries
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

    if logger is not None:
        logger.log_image(plt, current_step, "chart" + str(chart_index))

    if save_to_file:
        # Get day and time for the filename
        day_time = str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        path = os.path.join("../plots", f"rbt_plt_{day_time}.png")
        plt.savefig(path)

    if show_plot:
        plt.show()
    plt.close()


def visualize_analytical_distribution(model: KinematicsNetworkBase, param, ground_truth, goal, cfg: TrainConfig, device,
                                      logger=None,
                                      current_step=None, chart_index=1):
    model.eval()
    pred = model((param, goal))

    link_angles = [[] for _ in range(cfg.number_of_joints)]
    link_probabilities = [[] for _ in range(cfg.number_of_joints)]
    for joint_number in range(cfg.number_of_joints):
        index = 2 * joint_number
        mu_output = pred[index]
        sigma_output = pred[index + 1]

        # Map sigma to positive values from [0,1] to [1,2]
        sigma = sigma_output + 1

        # Map mu from [0,1] to [-pi,pi]
        mu = ((mu_output * 2) - 1) * np.pi

        link_angles[joint_number].extend(np.linspace(mu.item() - sigma.item(), mu.item() + sigma.item(),
                                                     num=cfg.vis.analytical.distribution_samples))

        normal_dist = torch.distributions.Normal(mu, sigma)
        for point in link_angles[joint_number]:
            expected_truth_prob = torch.exp(normal_dist.log_prob(torch.tensor(point).to(device))).item()
            if expected_truth_prob > 1:
                expected_truth_prob = 1
                #raise InputError("Expected truth probability is greater than 1")
            if expected_truth_prob < 0.1:
                expected_truth_prob = 0.1
            link_probabilities[joint_number].append(expected_truth_prob)

    vis_params = cfg.vis

    plot_distribution(
        parameter=param,
        link_angles=link_angles,
        link_probabilities=link_probabilities,
        chart_index=chart_index,
        goal=goal,
        ground_truth=ground_truth,
        default_line_width=vis_params.analytical.default_line_width,
        save_to_file=vis_params.save_to_file,
        show_plot=vis_params.show_plot,
        max_legend_length=vis_params.max_legend_length,
        logger=logger if cfg.logging.wandb.log_visualization_plots else None,
        current_step=current_step,
        show_legend=vis_params.show_legend,
        device=device
    )

    model.train()


def visualize_analytical_problem(model: KinematicsNetwork, param, goal, param_history, cfg: TrainConfig, logger=None,
                                 current_step=None, chart_index=1):
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
        chart_index=chart_index,
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
        chart_index,
        default_line_transparency=1.,
        default_line_width=1.,
        max_legend_length=10,
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
        chart_index: The index of the chart. Used for logging in wandb.
    """
    ax, multiple_robots = set_plot_settings(parameter)

    # Check whether multiple robot arms were given or only one
    if multiple_robots:
        # Multiple Robots arm were given as parameter
        for i in range(parameter.shape[0]):
            if use_gradual_transparency:
                # Calculate the transparency based on the number of robot arms
                transparency_steps = 1 / parameter.shape[0]
                transparency = 0.1 + transparency_steps * (i + 1)
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
                robot_label_note=str(i) + robot_label_note,
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
            labels = [labels.pop()] + labels[-max_legend_length - 1:]
            handles = [handles.pop()] + handles[-max_legend_length - 1:]
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )

    plt.tight_layout()

    if logger is not None:
        logger.log_image(plt, current_step, "chart" + str(chart_index))

    if save_to_file:
        # Get day and time for the filename
        day_time = str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        path = os.path.join("../plots", f"rbt_plt_{day_time}.png")
        plt.savefig(path)

    if show_plot:
        plt.show()
    plt.close()
