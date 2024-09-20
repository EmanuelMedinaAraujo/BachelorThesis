import sys

import hydra
from hydra.core.config_store import ConfigStore
import torch
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib import RecurrentPPO

from tqdm import tqdm

from analyticalRL.kinematics_network import KinematicsNetwork
from analyticalRL.kinematics_network_testing import test_loop
from analyticalRL.kinematics_network_training import train_loop
from data_generation.parameter_dataset import CustomParameterDataset
from data_generation.parameter_generator import ParameterGeneratorForPlanarRobot
from custom_logging.custom_loggger import GeneralLogger
from custom_logging.logger_callback import LoggerCallback
from stb3.kinematics_environment import KinematicsEnvironment
from vis.analytical_vis import visualize_analytical_problem
from conf.config import TrainConfig

cs = ConfigStore.instance()
cs.store(name="train_conf", node=TrainConfig)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def train_and_test_model(train_config: TrainConfig):
    """
    Train and test the model with the given hydra configuration.
    """
    device = (
        "cuda"+str(train_config.server_postfix)
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    tensor_type = torch.float32

    set_random_seed(train_config.random_seed)

    logger = GeneralLogger(cfg=train_config)
    logger.log_used_device(device=device)

    test_dataset = CustomParameterDataset(
        length=train_config.number_of_test_problems,
        device_to_use=device,
        num_of_joints=train_config.number_of_joints,
        parameter_convention=train_config.parameter_convention,
        min_link_len=train_config.min_link_length,
        max_link_len=train_config.max_link_length,
    )

    visualization_param, visualization_goal = test_dataset.__getitem__(0)
    visualization_history = []

    if train_config.use_stb3:
        do_stable_baselines3_learning(
            device=device,
            cfg=train_config,
            logger=logger,
            test_dataset=test_dataset,
            visualization_history=visualization_history,
            visualization_goal=visualization_goal,
            visualization_param=visualization_param,
            tensor_type=tensor_type,
        )
    else:
        do_analytical_learning(
            device=device,
            cfg=train_config,
            logger=logger,
            test_dataset=test_dataset,
            visualization_history=visualization_history,
            visualization_goal=visualization_goal,
            visualization_param=visualization_param,
            tensor_type=tensor_type,
        )
    logger.finish_logging()
    tqdm.write("Done!")


def do_stable_baselines3_learning(
    device,
    cfg: TrainConfig,
    logger,
    test_dataset,
    visualization_history,
    visualization_goal,
    visualization_param,
    tensor_type,
):
    # check_env(make_environment(device, hyperparams, tensor_type), warn=True)
    env = make_vec_env(
        lambda: make_environment(device, cfg, tensor_type),
        n_envs=cfg.hyperparams.stb3.n_envs,
    )

    # Define the model
    model = PPO(
        policy=cfg.hyperparams.stb3.non_recurrent_policy,
        env=env,
        batch_size=cfg.hyperparams.stb3.batch_size,
        learning_rate=cfg.hyperparams.stb3.learning_rate,
        n_epochs=cfg.hyperparams.stb3.epochs,
        n_steps=cfg.hyperparams.stb3.n_steps,
        gamma=cfg.hyperparams.stb3.gamma,
        ent_coef=cfg.hyperparams.stb3.ent_coef,
        seed=cfg.random_seed,
        policy_kwargs=dict(
            log_std_init=cfg.hyperparams.stb3.log_std_init
        ),  # Set initial standard deviation
    )

    if cfg.hyperparams.stb3.use_recurrent_policy:
        model = RecurrentPPO(
            policy=cfg.hyperparams.stb3.recurrent_policy,
            env=env,
            batch_size=cfg.hyperparams.stb3.batch_size,
            learning_rate=cfg.hyperparams.stb3.learning_rate,
            n_epochs=cfg.hyperparams.stb3.epochs,
            n_steps=cfg.hyperparams.stb3.n_steps,
            gamma=cfg.hyperparams.stb3.gamma,
            ent_coef=cfg.hyperparams.stb3.ent_coef,
            seed=cfg.random_seed,
            policy_kwargs=dict(log_std_init=cfg.hyperparams.stb3.log_std_init),
            # Set initial standard deviation
        )

    logger_callback = LoggerCallback(
        logger=logger,
        visualization_history=visualization_history,
        goal_to_vis=visualization_goal,
        param_to_vis=visualization_param,
        test_dataloader=test_dataset,
        cfg=cfg,
        device=device,
        tolerable_accuracy_error=cfg.tolerable_accuracy_error,
    )
    if cfg.logging.wandb.log_in_wandb:
        logger_callback = CallbackList(
            [
                logger_callback,
                WandbCallback(
                    gradient_save_freq=cfg.logging.wandb.wand_callback_logging_freq
                ),
            ]
        )

    # Train the model
    model.learn(
        total_timesteps=cfg.hyperparams.stb3.total_timesteps,
        callback=logger_callback,
        progress_bar=True,
    )


def do_analytical_learning(
    device,
    cfg: TrainConfig,
    logger,
    test_dataset,
    visualization_history,
    visualization_goal,
    visualization_param,
    tensor_type,
):
    model = KinematicsNetwork(
        num_joints=cfg.number_of_joints,
        num_layer=cfg.hyperparams.analytical.num_layer,
        layer_sizes=cfg.hyperparams.analytical.layer_sizes,
        logger=logger,
    ).to(device)
    logger.watch_model(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.hyperparams.analytical.learning_rate
    )

    # Create Problem Generator
    problem_generator = ParameterGeneratorForPlanarRobot(
        batch_size=cfg.hyperparams.analytical.batch_size,
        device=device,
        tensor_type=tensor_type,
        num_joints=cfg.number_of_joints,
        parameter_convention=cfg.parameter_convention,
        min_len=cfg.min_link_length,
        max_len=cfg.max_link_length,
    )

    for epoch_num in tqdm(
        range(cfg.hyperparams.analytical.epochs),
        colour="green",
        file=sys.stdout,
    ):
        train_loop(
            model=model,
            optimizer=optimizer,
            problem_generator=problem_generator,
            problems_per_epoch=cfg.hyperparams.analytical.problems_per_epoch,
            batch_size=cfg.hyperparams.analytical.batch_size,
            device=device,
            logger=logger,
            epoch_num=epoch_num,
            error_tolerance=cfg.tolerable_accuracy_error,
        )

        # Test the model every hyperparams.testing_interval epochs
        if epoch_num % cfg.hyperparams.analytical.testing_interval == 0:
            test_loop(
                test_dataset=test_dataset,
                model=model,
                device=device,
                logger=logger,
                tolerable_accuracy_error=cfg.tolerable_accuracy_error,
                epoche_num=epoch_num
            )

        # Visualize the same problem every hyperparams.visualization.interval epochs
        if cfg.do_vis and epoch_num % cfg.vis.analytical.interval == 0:
            visualize_analytical_problem(
                model=model,
                param=visualization_param,
                goal=visualization_goal,
                param_history=visualization_history,
                cfg=cfg,
                logger=logger,
                current_step=epoch_num,
            )


def make_environment(device, cfg, tensor_type):
    return KinematicsEnvironment(device, cfg=cfg, tensor_type=tensor_type)


if __name__ == "__main__":
    torch.set_num_threads(1)
    train_and_test_model()
