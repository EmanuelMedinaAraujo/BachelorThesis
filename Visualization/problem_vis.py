import numpy as np
import torch

from Visualization.planar_robot_vis import visualize_planar_robot


def visualize_problem(model, param, goal, device, param_history, hyperparams):
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

            pred, _ = model.predict(np.array([1000]).astype(np.float32))
            pred = torch.tensor(pred).to(device)

            # Reset goal and parameter
            env.env_method("set_goal", old_goal[0])
            env.env_method("set_parameter", old_param[0])

        else:
            pred = model((param, goal))

        # Concatenate the predicted theta values to the parameter
        updated_param = torch.cat((param, pred.unsqueeze(1)), dim=-1)
        param_history.append(updated_param)

        vis_params = hyperparams.visualization
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
                               show_distance=vis_params.show_distance_in_legend)

        if not hyperparams.use_stable_baselines3:
            model.train()
