import numpy as np
import torch

from Visualization.planar_robot_vis import visualize_planar_robot


def visualize_problem(model, param, goal, device, param_history, hyperparams, logger=None):
    """
    Visualize a single robot arm with the given parameters and goal.
    """
    if not hyperparams.use_stable_baselines3:
        model.eval()

    with torch.no_grad():
        if hyperparams.use_stable_baselines3:
            env = model.get_env()
            # Get goal and parameter from model environment
            old_goal = env.env_method("get_wrapper_attr", "goal")
            old_param = env.env_method("get_wrapper_attr", "parameter")

            # Set visualization goal and parameter to training_env
            env.env_method("set_goal", goal)
            env.env_method("set_parameter", param)

            observation = torch.concat([param.flatten(), goal]).detach().cpu().numpy()

            if hyperparams.use_recurrent_policy:
                # cell and hidden state of the LSTM
                lstm_states = None
                episode_starts = np.ones((1,), dtype=bool)
                pred, _ = model.predict(observation, state=lstm_states, episode_start=episode_starts)
            else:
                pred, _ = model.predict(observation)
            pred = torch.tensor(pred).to(device)

            if hyperparams.visualization.visualize_distribution:
                predictions = [pred]
                for _ in range(hyperparams.visualization.num_distribution_samples - 1):
                    if hyperparams.use_recurrent_policy:
                        pred, _ = model.predict(observation, state=lstm_states, episode_start=episode_starts)
                    else:
                        pred, _ = model.predict(observation)
                    pred = torch.tensor(pred).to(device)
                    predictions.append(pred)

            # Reset goal and parameter
            env.env_method("set_goal", old_goal[0])
            env.env_method("set_parameter", old_param[0])

        else:
            pred = model((param, goal))

        vis_params = hyperparams.visualization
        # Concatenate the predicted theta values to the parameter
        if vis_params.visualize_distribution:

            # Check if the predictions are equal within a certain error margin and replace them with the same value
            for i in range(vis_params.num_distribution_samples):
                error_margin = vis_params.distribution_sample_error
                for j in range(i,vis_params.num_distribution_samples):
                    # Check if the predictions are equal within a certain error margin
                    if torch.all(torch.abs(predictions[i] - predictions[j]) < error_margin):
                        predictions[j] = predictions[i]

            predictions, link_accuracy = torch.unique(torch.stack(predictions), dim=0,
                                               return_counts=True)
            link_accuracy = link_accuracy / vis_params.num_distribution_samples
            if len(torch.unique(link_accuracy)) == 1:
                # If all link accuracies are the same, set all to the default value
                link_accuracy = torch.full([len(link_accuracy)], vis_params.default_line_transparency).to(device)

            if vis_params.scale_distribution_probabilities:
                scaling_summand = (1 - torch.max(link_accuracy))/2
                link_accuracy = link_accuracy + scaling_summand

            # Sort link_accuracy in descending order and predictions accordingly
            link_accuracy, indices = torch.sort(link_accuracy, descending=True)
            predictions = predictions[indices]

            predictions_tensor = predictions.unsqueeze(-1)
            repeated_params = param.expand(predictions_tensor.shape[0], 2, 3)
            updated_param = torch.cat([repeated_params, predictions_tensor], dim=-1)
            param_history.append(updated_param)
        else:
            updated_param = torch.cat((param, pred.unsqueeze(1)), dim=-1)
            param_history.append(updated_param)

        # Check if max history length is reached
        if len(param_history) > vis_params.max_history_length:
            param_history.pop(0)

        tensor_to_pass = updated_param if not vis_params.plot_all_in_one else torch.stack(param_history)

        visualize_planar_robot(parameter=tensor_to_pass,
                               goal=goal,
                               device=device,
                               standard_size=vis_params.standard_size,
                               save_to_file=vis_params.save_to_file,
                               default_line_transparency=vis_params.default_line_transparency,
                               frame_size_scalar=vis_params.frame_size_scalar,
                               default_line_width=vis_params.default_line_width,
                               use_color_per_robot=vis_params.use_color_per_robot,
                               use_gradual_transparency=vis_params.use_gradual_transparency,
                               show_plot=vis_params.show_plot,
                               show_joint_label=vis_params.show_joint_label,
                               show_distance=vis_params.show_distance_in_legend,
                               is_distribution=vis_params.visualize_distribution,
                               link_accuracy=link_accuracy if vis_params.visualize_distribution else None,
                               max_legend_length=vis_params.max_legend_length,
                               logger=logger if hyperparams.log_visualization_plots else None)

        if not hyperparams.use_stable_baselines3:
            model.train()
