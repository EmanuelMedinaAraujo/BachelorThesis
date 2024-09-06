import sys

import hydra
import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from AnalyticalRL.kinematics_network import KinematicsNetwork
from AnalyticalRL.kinematics_network_testing import test_loop
from AnalyticalRL.kinematics_network_training import train_loop
from DataGeneration.parameter_dataset import CustomParameterDataset
from DataGeneration.parameter_generator import ParameterGeneratorForPlanarRobot
from Logging.custom_loggger import GeneralLogger
from Logging.logger_callback import LoggerCallback
from PPO.kinematics_environment import KinematicsEnvironment
from Visualization.planar_robot_vis import visualize_planar_robot


def visualize_problem(model, param, goal, device, param_history, hyperparams):
    """
    Visualize a single robot arm with the given parameters and goal.
    """
    if not hyperparams.use_stable_baselines3:
        model.eval()

    with torch.no_grad():
        if hyperparams.use_stable_baselines3:
            data_list = [(param, goal)]
            kin_env = KinematicsEnvironment(device=device, dataloader=data_list,
                                            num_joints=hyperparams.number_of_joints,
                                            tolerable_accuracy_error=hyperparams.tolerable_accuracy_error)
            env = make_vec_env(lambda: kin_env, n_envs=1)
            obs = env.reset()
            pred, _ = model.predict(obs)
            pred = torch.tensor(pred).to(device)
        else:
            pred = model((param, goal))

        # Concatenate the predicted theta values to the parameter
        updated_param = torch.cat((param, pred.unsqueeze(1)), dim=-1)
        param_history.append(updated_param)

        vis_params = hyperparams.visualization
        tensor_to_pass = updated_param if not vis_params.plot_all_in_one else torch.stack(param_history)

        visualize_planar_robot(parameter=tensor_to_pass, goal=goal, device=device,
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

    set_random_seed(hyperparams.random_seed)

    logger = GeneralLogger(log_in_wandb=hyperparams.log_in_wandb,
                           log_in_console=hyperparams.log_in_console, cfg=cfg)
    logger.log_used_device(device=device)

    train_dataloader = DataLoader(CustomParameterDataset(length=hyperparams.problems_per_epoch,
                                                         device_to_use=device,
                                                         num_of_joints=hyperparams.number_of_joints,
                                                         parameter_convention=hyperparams.parameter_convention,
                                                         min_link_len=hyperparams.min_link_length,
                                                         max_link_len=hyperparams.max_link_length),
                                  hyperparams.batch_size, shuffle=True)
    test_dataset = CustomParameterDataset(length=hyperparams.problems_per_epoch, device_to_use=device,
                                          num_of_joints=hyperparams.number_of_joints,
                                          parameter_convention=hyperparams.parameter_convention,
                                          min_link_len=hyperparams.min_link_length,
                                          max_link_len=hyperparams.max_link_length)

    visualization_param, visualization_goal = test_dataset.__getitem__(0)
    visualization_history = []

    if hyperparams.use_stable_baselines3:
        do_stable_baselines3_learning(device, hyperparams, logger, train_dataloader, test_dataset,
                                      visualization_history,
                                      visualization_goal, visualization_param)
    else:
        do_analytical_learning(device, hyperparams, logger, test_dataset, visualization_history,
                               visualization_goal, visualization_param)
    tqdm.write("Done!")


def do_stable_baselines3_learning(device, hyperparams, logger, train_dataloader, test_dataloader, visualization_history,
                                  visualization_goal, visualization_param):
    env = make_vec_env(lambda: make_environment(device, train_dataloader, hyperparams.number_of_joints,
                                                hyperparams.tolerable_accuracy_error),
                       n_envs=hyperparams.n_envs)
    # Define the model
    model = PPO(policy=hyperparams.policy, env=env, batch_size=hyperparams.batch_size,
                learning_rate=hyperparams.learning_rate, n_epochs=hyperparams.epochs,
                n_steps=hyperparams.n_steps, verbose=hyperparams.log_verbosity)

    logger_callback = LoggerCallback(logger=logger, visualization_history=visualization_history,
                                     goal_to_vis=visualization_goal, param_to_vis=visualization_param,
                                     verbose=hyperparams.log_verbosity, test_dataloader=test_dataloader,
                                     hyperparams=hyperparams, visualization_call=visualize_problem, device=device,
                                     num_joints=hyperparams.number_of_joints,
                                     tolerable_accuracy_error=hyperparams.tolerable_accuracy_error)
    # Train the model
    model.learn(total_timesteps=hyperparams.total_timesteps, callback=logger_callback,
                progress_bar=True)


def do_analytical_learning(device, hyperparams, logger, test_dataset, visualization_history,
                           visualization_goal, visualization_param):
    model = KinematicsNetwork(num_joints=hyperparams.number_of_joints, num_layer=hyperparams.num_layer,
                              layer_sizes=hyperparams.layer_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.learning_rate)

    # Create Problem Generator
    problem_generator = ParameterGeneratorForPlanarRobot(batch_size=hyperparams.batch_size,
                                                         device=device,
                                                         tensor_type=torch.float32,
                                                         num_joints=hyperparams.number_of_joints,
                                                         parameter_convention=hyperparams.parameter_convention,
                                                         min_len=hyperparams.min_link_length,
                                                         max_len=hyperparams.max_link_length)

    for epoch_num in tqdm(range(hyperparams.epochs), colour='green', ncols=80, file=sys.stdout):
        train_loop(model=model,
                   optimizer=optimizer,
                   problem_generator=problem_generator,
                   problems_per_epoch=hyperparams.problems_per_epoch,
                   batch_size=hyperparams.batch_size,
                   device=device,
                   logger=logger,
                   epoch_num=epoch_num,
                   error_tolerance=hyperparams.tolerable_accuracy_error)

        # Test the model every hyperparams.testing_interval epochs
        if (epoch_num + 1) % hyperparams.testing_interval == 0:
            test_loop(test_dataset=test_dataset,
                      model=model,
                      device=device,
                      logger=logger,
                      tolerable_accuracy_error=hyperparams.tolerable_accuracy_error)

        # Visualize the same problem every hyperparams.visualization.interval epochs
        if hyperparams.visualization.do_visualization and epoch_num % hyperparams.visualization.interval == 0:
            visualize_problem(model=model, device=device, param=visualization_param, goal=visualization_goal,
                              param_history=visualization_history,
                              hyperparams=hyperparams)


def make_environment(device, dataloader, num_joints, tolerable_accuracy_error):
    return KinematicsEnvironment(device, dataloader, num_joints=num_joints,
                                 tolerable_accuracy_error=tolerable_accuracy_error)


if __name__ == "__main__":
    train_and_test_model()
