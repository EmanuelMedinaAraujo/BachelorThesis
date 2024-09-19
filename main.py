import sys

import hydra
import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib import RecurrentPPO

from tqdm import tqdm

from AnalyticalRL.kinematics_network import KinematicsNetwork
from AnalyticalRL.kinematics_network_testing import test_loop
from AnalyticalRL.kinematics_network_training import train_loop
from DataGeneration.parameter_dataset import CustomParameterDataset
from DataGeneration.parameter_generator import ParameterGeneratorForPlanarRobot
from Logging.custom_loggger import GeneralLogger
from Logging.logger_callback import LoggerCallback
from PPO.kinematics_environment import KinematicsEnvironment
from Visualization.problem_vis import visualize_problem


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

    # Print total timestep from config
    print(f"Total Timesteps: {cfg.hyperparams.total_timesteps}")

    tensor_type = torch.float32

    hyperparams = cfg.hyperparams

    set_random_seed(hyperparams.random_seed)

    logger = GeneralLogger(log_in_wandb=hyperparams.log_in_wandb,
                           log_in_console=hyperparams.log_in_console,
                           cfg=cfg)
    logger.log_used_device(device=device)

    test_dataset = CustomParameterDataset(length=hyperparams.number_of_test_problems,
                                          device_to_use=device,
                                          num_of_joints=hyperparams.number_of_joints,
                                          parameter_convention=hyperparams.parameter_convention,
                                          min_link_len=hyperparams.min_link_length,
                                          max_link_len=hyperparams.max_link_length)

    visualization_param, visualization_goal = test_dataset.__getitem__(0)
    visualization_history = []

    if hyperparams.use_stable_baselines3:
        do_stable_baselines3_learning(device, hyperparams, logger, test_dataset,
                                      visualization_history, visualization_goal, visualization_param, tensor_type)
    else:
        do_analytical_learning(device, hyperparams, logger, test_dataset, visualization_history,
                               visualization_goal, visualization_param, tensor_type)
    logger.finish_logging()
    tqdm.write("Done!")


def do_stable_baselines3_learning(device, hyperparams, logger, test_dataloader, visualization_history,
                                  visualization_goal, visualization_param, tensor_type):
    env = make_vec_env(lambda: make_environment(device, hyperparams, tensor_type), n_envs=hyperparams.n_envs)
    # check_env(make_environment(device, hyperparams, tensor_type), warn=True)

    # Define the model
    model = PPO(policy=hyperparams.non_recurrent_policy,
                env=env,
                batch_size=hyperparams.batch_size,
                learning_rate=hyperparams.learning_rate,
                n_epochs=hyperparams.epochs,
                n_steps=hyperparams.n_steps,
                verbose=hyperparams.log_verbosity,
                gamma=hyperparams.gamma,
                ent_coef=hyperparams.ent_coef,
                seed=hyperparams.random_seed,
                policy_kwargs=dict(log_std_init=hyperparams.log_std_init)  # Set initial standard deviation
                )

    if hyperparams.use_recurrent_policy:
        model = RecurrentPPO(policy=hyperparams.recurrent_policy,
                             env=env,
                             batch_size=hyperparams.batch_size,
                             learning_rate=hyperparams.learning_rate,
                             n_epochs=hyperparams.epochs,
                             n_steps=hyperparams.n_steps,
                             verbose=hyperparams.log_verbosity,
                             gamma=hyperparams.gamma,
                             ent_coef=hyperparams.ent_coef,
                             seed=hyperparams.random_seed,
                             policy_kwargs=dict(log_std_init=hyperparams.log_std_init)  # Set initial standard deviation
                             )

    logger_callback = LoggerCallback(logger=logger,
                                     visualization_history=visualization_history,
                                     goal_to_vis=visualization_goal,
                                     param_to_vis=visualization_param,
                                     verbose=hyperparams.log_verbosity,
                                     test_dataloader=test_dataloader,
                                     hyperparams=hyperparams,
                                     device=device,
                                     tolerable_accuracy_error=hyperparams.tolerable_accuracy_error)
    # Train the model
    model.learn(total_timesteps=hyperparams.total_timesteps, callback=logger_callback,
                progress_bar=True)


def do_analytical_learning(device, hyperparams, logger, test_dataset, visualization_history,
                           visualization_goal, visualization_param, tensor_type):
    model = KinematicsNetwork(num_joints=hyperparams.number_of_joints,
                              num_layer=hyperparams.num_layer,
                              layer_sizes=hyperparams.layer_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.learning_rate)

    # Create Problem Generator
    problem_generator = ParameterGeneratorForPlanarRobot(batch_size=hyperparams.batch_size,
                                                         device=device,
                                                         tensor_type=tensor_type,
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
            visualize_problem(model=model, param=visualization_param, goal=visualization_goal, device=device,
                              param_history=visualization_history, hyperparams=hyperparams)


def make_environment(device, hyperparams, tensor_type):
    return KinematicsEnvironment(device, hyperparams=hyperparams, tensor_type=tensor_type)


if __name__ == "__main__":
    train_and_test_model()
