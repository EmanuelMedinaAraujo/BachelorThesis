import sys

import hydra
import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.data import DataLoader
from tqdm import tqdm

from AnalyticalRL.kinematics_network import KinematicsNetwork
from AnalyticalRL.kinematics_network_testing import test_loop
from AnalyticalRL.kinematics_network_training import train_loop
from AnalyticalRL.parameter_dataset import CustomParameterDataset
from Logging.loggger import Logger
from PPO.kinematics_environment import KinematicsEnvironment
from PPO.logger_callback import LoggerCallback
from Visualization.planar_robot_vis import visualize_planar_robot


def visualize_problem(model, param, goal, device, param_history, hyperparams):
    """
    Visualize a single robot arm with the given parameters and goal.
    """
    if not hyperparams.use_stable_baselines3:
        model.eval()

    with torch.no_grad():
        if hyperparams.use_stable_baselines3:
            kin_env = KinematicsEnvironment(device=device, parameter=param, goal_coordinates=goal,
                                            num_joints=hyperparams.number_of_joints)
            env = make_vec_env(lambda: kin_env, n_envs=1)
            obs = env.reset()
            pred, _ = model.predict(obs)
            pred = torch.tensor(pred).to(device)
        else:
            pred = model((param, goal))

        # Add a dimension to include theta values
        new_theta_values = pred[0].unsqueeze(-1)
        updated_param = torch.cat((param[0], new_theta_values), dim=-1)
        param_history.append(updated_param)

        vis_params = hyperparams.visualization
        tensor_to_pass = updated_param if not vis_params.plot_all_in_one else torch.stack(param_history, dim=0)

        visualize_planar_robot(parameter=tensor_to_pass, goal=goal[0], device=device,
                               standard_size=vis_params.standard_size,
                               save_to_file=vis_params.save_to_file,
                               default_line_transparency=vis_params.default_line_transparency,
                               frame_size_scalar=vis_params.frame_size_scalar,
                               default_line_width=vis_params.default_line_width,
                               use_color_per_robot=vis_params.use_color_per_robot,
                               use_gradual_transparency=vis_params.use_gradual_transparency,
                               show_plot=vis_params.show_plot, show_joint_label=vis_params.show_joint_label)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_and_test_model(cfg: DictConfig):
    """
    Train and test the model with the given hydra configuration.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    hyperparams = cfg.hyperparams
    logger = Logger(log_in_wandb=hyperparams.log_in_wandb,
                    log_in_console=hyperparams.log_in_console, cfg=cfg)

    logger.log_used_device(device=device)

    train_dataloader = DataLoader(CustomParameterDataset(length=hyperparams.dataset_length,
                                                         device_to_use=device,
                                                         num_of_joints=hyperparams.number_of_joints,
                                                         parameter_convention=hyperparams.parameter_convention,
                                                         min_link_len=hyperparams.min_link_length,
                                                         max_link_len=hyperparams.max_link_length),
                                  hyperparams.batch_size, shuffle=True)
    test_dataloader = DataLoader(CustomParameterDataset(length=hyperparams.dataset_length,
                                                        device_to_use=device,
                                                        num_of_joints=hyperparams.number_of_joints,
                                                        parameter_convention=hyperparams.parameter_convention,
                                                        min_link_len=hyperparams.min_link_length,
                                                        max_link_len=hyperparams.max_link_length),
                                 hyperparams.batch_size, shuffle=True)

    visualization_param, visualization_goal = next(iter(test_dataloader))
    visualization_history = []

    if hyperparams.use_stable_baselines3:
        do_stable_baselines3_learning(device, hyperparams, logger, train_dataloader, visualization_history,
                                      visualization_goal, visualization_param)
    else:
        do_analytical_learning(device, hyperparams, logger, test_dataloader, train_dataloader, visualization_history,
                               visualization_goal, visualization_param)
    tqdm.write("Done!")


def do_stable_baselines3_learning(device, hyperparams, logger, train_dataloader, visualization_history,
                                  visualization_goal, visualization_param):
    env = make_vec_env(lambda: make_environment(device, train_dataloader, hyperparams.number_of_joints),
                       n_envs=hyperparams.n_envs)
    # Define the model
    model = PPO(policy=hyperparams.policy, env=env, batch_size=hyperparams.batch_size,
                learning_rate=hyperparams.learning_rate, n_epochs=hyperparams.epochs,
                n_steps=hyperparams.n_steps, verbose=hyperparams.log_verbosity)

    # Train the model
    logger_callback = LoggerCallback(logger=logger, visualization_history=visualization_history,
                                     goal_to_vis=visualization_goal, param_to_vis=visualization_param,
                                     verbose=hyperparams.log_verbosity,
                                     hyperparams=hyperparams, visualization_call=visualize_problem, device=device)

    model.learn(total_timesteps=hyperparams.total_timesteps, callback=logger_callback,
                progress_bar=True)


def do_analytical_learning(device, hyperparams, logger, test_dataloader, train_dataloader, visualization_history,
                           visualization_goal, visualization_param):
    model = KinematicsNetwork(num_joints=hyperparams.number_of_joints, num_layer=hyperparams.num_layer,
                              layer_sizes=hyperparams.layer_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.learning_rate)
    for epoch_num in tqdm(range(hyperparams.epochs), colour='green', ncols=80, file=sys.stdout):
        train_loop(dataloader=train_dataloader,
                   model=model,
                   optimizer=optimizer,
                   device=device,
                   logger=logger,
                   epoch_num=epoch_num,
                   error_tolerance=hyperparams.tolerable_accuracy_error)
        if (epoch_num + 1) % hyperparams.testing_interval == 0:
            test_loop(dataloader=test_dataloader,
                      model=model,
                      device=device,
                      logger=logger,
                      tolerable_accuracy_error=hyperparams.tolerable_accuracy_error)
        vis_params = hyperparams.visualization
        # Visualize the same problem every interval epochs
        if vis_params.do_visualization and epoch_num % vis_params.interval == 0:
            visualize_problem(model=model, device=device, param=visualization_param, goal=visualization_goal,
                              param_history=visualization_history,
                              hyperparams=hyperparams)


def make_environment(device, dataloader, num_joints):
    parameters, goals = next(iter(dataloader))
    return KinematicsEnvironment(device, parameters[0], goal_coordinates=goals[0], num_joints=num_joints)


if __name__ == "__main__":
    train_and_test_model()
