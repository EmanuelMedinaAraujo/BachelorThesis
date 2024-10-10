import numpy as np
import torch

from conf.conf_dataclasses.config import TrainConfig
from util.forward_kinematics import calculate_angles_from_network_output
from vis.planar_robot_vis import visualize_planar_robot


def visualize_stb3_problem(
        model,
        param,
        goal,
        device,
        param_history,
        cfg: TrainConfig,
        logger=None,
        current_step=None,
        chart_index=1
):
    """
    Visualize a single robot arm with the given parameters and goal.
    """
    with torch.no_grad():
        observation = torch.concat([param.flatten(), goal]).detach().cpu().numpy()

        if cfg.hyperparams.stb3.use_recurrent_policy:
            # cell and hidden state of the LSTM
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            pred, _ = model.predict(
                observation, state=lstm_states, episode_start=episode_starts
            )
        else:
            pred, _ = model.predict(observation)
        pred = torch.tensor(pred).to(device)
        all_angles = calculate_angles_from_network_output(pred, cfg.number_of_joints, param.device)
        predictions = [all_angles]

        if cfg.vis.stb3.visualize_distribution:
            for _ in range(cfg.vis.stb3.num_distribution_samples - 1):
                if cfg.hyperparams.stb3.use_recurrent_policy:
                    pred, _ = model.predict(
                        observation, state=lstm_states, episode_start=episode_starts
                    )
                else:
                    pred, _ = model.predict(observation)
                pred = torch.tensor(pred).to(device)
                all_angles = calculate_angles_from_network_output(pred, cfg.number_of_joints, param.device)
                predictions.append(all_angles)

        vis_params = cfg.vis.stb3


        predictions = torch.stack(predictions)
        link_accuracy = torch.tensor([cfg.vis.stb3.default_line_transparency] * predictions.shape[0] )

        predictions_tensor = predictions.unsqueeze(-1)
        repeated_params = param.expand(predictions_tensor.shape[0], 2, 3)
        updated_param = torch.cat([repeated_params, predictions_tensor], dim=-1)
        param_history.append(updated_param)

        tensor_to_pass = torch.stack(
            param_history).squeeze(dim=0) if not vis_params.visualize_distribution else updated_param

        visualize_planar_robot(
            parameter=tensor_to_pass,
            goal=goal,
            save_to_file=cfg.vis.save_to_file,
            default_line_transparency=vis_params.default_line_transparency,
            default_line_width=vis_params.default_line_width,
            show_plot=cfg.vis.show_plot,
            show_joints=cfg.vis.show_joints,
            show_end_effectors=cfg.vis.show_end_effectors,
            show_joint_label=cfg.vis.show_joint_label,
            show_distance=cfg.vis.show_distance_in_legend,
            link_accuracy=link_accuracy.tolist() if vis_params.visualize_distribution else None,
            max_legend_length=cfg.vis.max_legend_length,
            logger=logger if cfg.logging.wandb.log_visualization_plots else None,
            current_step=current_step,
            do_heat_map=vis_params.do_heat_map,
            chart_index=chart_index
        )
