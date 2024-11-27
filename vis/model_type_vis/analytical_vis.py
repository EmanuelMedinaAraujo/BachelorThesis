import numpy as np
import torch

from networks.analyticalRL.networks.distributions.one_peak_distributions.three_output_param_dist_network_base import \
    ThreeOutputParameterDistrNetworkBase
from networks.analyticalRL.networks.distributions.two_peak_distributions.two_peak_norm_dist_network_base import \
    TwoPeakNormalDistrNetworkBase
from networks.analyticalRL.networks.simple_kinematics_network import SimpleKinematicsNetwork
from conf.conf_dataclasses.config import TrainConfig
from vis.planar_robot_vis import plot_planar_robot, create_eef_heatmap
from vis.vis_utils import set_plot_settings, plot_distribution_single_link, finish_and_close_plot


def plot_distribution(parameter, link_angles, ground_truth, link_probabilities, chart_index, goal, default_line_width,
                      save_to_file, show_plot, max_legend_length, logger, current_step, show_legend, device,
                      do_heat_map):
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

    all_start_angles = np.zeros(1)
    all_start_points = np.zeros((1, 2))

    start_angle_max = 0
    start_point_max = [0, 0]

    for joint_number in range(link_angles.__len__()):
        if joint_number == link_angles.__len__() - 1:
            is_last_link = True
        tmp_all_start_angles = []
        tmp_all_start_points = []
        for link_num in range(link_angles[joint_number].__len__()):
            for i in range(all_start_angles.__len__()):
                current_total_angle = all_start_angles[i] + link_angles[joint_number][link_num]
                tmp_all_start_angles.append(current_total_angle)
                tmp_all_start_points.append(all_start_points[i] + np.array(
                    [parameter[joint_number, 1].item() * np.cos(current_total_angle),
                     parameter[joint_number, 1].item() * np.sin(current_total_angle)]
                ))

            link_prob = link_probabilities[joint_number][link_num]
            draw_end_effector = False
            if is_last_link and link_prob == max(link_probabilities[joint_number]):
                draw_end_effector = True
            if link_prob < 0.1:
                link_prob = 0.1
            if link_prob > 1.:
                link_prob = 1.
            plot_distribution_single_link(ax=ax,
                                          link_length=parameter[joint_number, 1].item(),
                                          angle=link_angles[joint_number][link_num],
                                          default_line_width=default_line_width,
                                          transparency=link_prob,
                                          show_distance=is_last_link,
                                          mark_as_best_end_effector=draw_end_effector,
                                          goal=goal,
                                          start_point=start_point_max,
                                          start_angle=start_angle_max,
                                          # color="blue" if is_last_link else 'g',
                                          color="blue",
                                          device=device)
        # Get index of maximum probability in link_probabilities[joint_number]
        max_prob_index = link_probabilities[joint_number].index(max(link_probabilities[joint_number]))
        start_angle_max += link_angles[joint_number][max_prob_index]
        start_point_max = [start_point_max[0] + parameter[joint_number, 1].item() * np.cos(start_angle_max),
                           start_point_max[1] + parameter[joint_number, 1].item() * np.sin(start_angle_max)]

        all_start_angles = np.array(tmp_all_start_angles)
        all_start_points = np.array(tmp_all_start_points)

    # Plot the goal of the robot, configure the legend, log, save, open and close the plot
    finish_and_close_plot(ax, chart_index, current_step, goal, logger, max_legend_length, save_to_file, show_legend,
                          show_plot)

    if do_heat_map:
        create_eef_heatmap(all_start_points, goal, logger, current_step, show_plot, save_to_file, parameter,
                           chart_index)


def visualize_analytical_distribution(model, param, ground_truth, goal, cfg: TrainConfig,
                                      device,
                                      logger=None,
                                      current_step=None, chart_index=1):
    model.eval()
    pred = model((param, goal))

    link_angles = [[] for _ in range(cfg.number_of_joints)]
    link_probabilities = [[] for _ in range(cfg.number_of_joints)]
    for joint_number in range(cfg.number_of_joints):
        distribution_params = pred[joint_number]
        parameter_1 = distribution_params[0].unsqueeze(-1)
        parameter_2 = distribution_params[1].unsqueeze(-1)

        if cfg.hyperparams.analytical.output_type == "BetaDist":
            epsilon = 1e-6
            parameter_1 = parameter_1 + epsilon
            parameter_2 = parameter_2 + epsilon
            dist = torch.distributions.Beta(parameter_1, parameter_2)

            angles = dist.sample(torch.Size([cfg.vis.analytical.distribution_samples]))
            expected_truth_prob = torch.exp(dist.log_prob(angles))
            link_probabilities[joint_number].extend(expected_truth_prob.squeeze().tolist())
            # Map angles since beta distribution is between 0 and 1
            angle = (2 * angles - 1) * np.pi
            link_angles[joint_number].extend(angle.squeeze().tolist())

            # Add mean to visualization with probability of 1 for easier comparison
            link_probabilities[joint_number].append(1)
            angle = (2 * dist.mean.item() - 1) * np.pi
            link_angles[joint_number].append(angle)
        elif isinstance(model, TwoPeakNormalDistrNetworkBase):
            parameter_3 = distribution_params[2].unsqueeze(-1)
            parameter_4 = distribution_params[3].unsqueeze(-1)
            parameter_5 = distribution_params[4].unsqueeze(-1)
            parameter_6 = distribution_params[5].unsqueeze(-1)

            (mu1, sigma1, weight1, mu2, sigma2, weight2) = (parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6)

            mu, sigma = model.sample_component(mu1, mu2, sigma1, sigma2, weight1, weight2, cfg.vis.analytical.distribution_samples)

            # Sample standard normal noise
            noise = torch.randn(torch.Size([cfg.vis.analytical.distribution_samples])).to(device)

            # Reparameterized sampling
            angles = mu + (sigma * noise)

            link_angles[joint_number].extend(angles.tolist())

            # Calculate probabilities using mixture of normal distributions
            expected_truth_prob = weight1 * torch.exp(torch.distributions.Normal(mu1, sigma1).log_prob(angles)) + \
                                  weight2 * torch.exp(torch.distributions.Normal(mu2, sigma2).log_prob(angles))

            link_probabilities[joint_number].extend(expected_truth_prob.squeeze().tolist())
        else:
            dist = torch.distributions.Normal(loc=parameter_1, scale=parameter_2)

            angles = dist.sample(torch.Size([cfg.vis.analytical.distribution_samples]))
            link_angles[joint_number].extend(angles.squeeze().tolist())
            expected_truth_prob = torch.exp(dist.log_prob(angles))
            link_probabilities[joint_number].extend(expected_truth_prob.squeeze().tolist())

            # Add mean to visualization with probability of 1 for easier comparison
            link_probabilities[joint_number].append(1)
            link_angles[joint_number].append(dist.mean.item())

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
        device=device,
        do_heat_map=vis_params.analytical.do_heat_map
    )

    model.train()


def visualize_analytical_problem(model: SimpleKinematicsNetwork, param, goal, param_history, cfg: TrainConfig,
                                 logger=None,
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

    # Plot the goal of the robot, configure the legend, log, save, open and close the plot
    finish_and_close_plot(ax, chart_index, current_step, goal, logger, max_legend_length, save_to_file, show_legend,
                          show_plot)
