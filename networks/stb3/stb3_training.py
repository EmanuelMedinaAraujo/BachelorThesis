from datetime import datetime

import optuna
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from torch import nn

from conf.conf_dataclasses.config import TrainConfig
from custom_logging.callback_logger import LoggerCallback
from networks.stb3.kinematics_environment import KinematicsEnvironment


def do_stable_baselines3_learning(
        device,
        cfg: TrainConfig,
        logger,
        test_dataset,
        visualization_history,
        visualization_goals,
        visualization_params,
        tensor_type,
        trial: optuna.Trial = None
):
    # check_env(make_environment(device, cfg, tensor_type), warn=True)
    env = make_vec_env(
        lambda: make_environment(device, cfg, tensor_type),
        n_envs=cfg.hyperparams.stb3.n_envs
    )

    # Define the model
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[
        cfg.hyperparams.stb3.activation_fn_name]
    net_arch = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[cfg.hyperparams.stb3.net_arch_type]

    model = PPO(
        policy=cfg.hyperparams.stb3.non_recurrent_policy,
        env=env,
        batch_size=cfg.hyperparams.stb3.batch_size,
        learning_rate=cfg.hyperparams.stb3.learning_rate,
        n_epochs=cfg.hyperparams.stb3.epochs,
        n_steps=cfg.hyperparams.stb3.n_steps,
        gamma=cfg.hyperparams.stb3.gamma,
        ent_coef=cfg.hyperparams.stb3.ent_coef,
        gae_lambda=cfg.hyperparams.stb3.gae_lambda,
        clip_range=cfg.hyperparams.stb3.clip_range,
        normalize_advantage=cfg.hyperparams.stb3.norm_advantages,
        vf_coef=cfg.hyperparams.stb3.vf_coef,
        max_grad_norm=cfg.hyperparams.stb3.max_grad_norm,
        seed=cfg.random_seed,
        device=device,
        policy_kwargs=dict(
            log_std_init=cfg.hyperparams.stb3.log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=cfg.hyperparams.stb3.ortho_init,
            normalize_images=False
        ),
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
            gae_lambda=cfg.hyperparams.stb3.gae_lambda,
            clip_range=cfg.hyperparams.stb3.clip_range,
            normalize_advantage=cfg.hyperparams.stb3.norm_advantages,
            vf_coef=cfg.hyperparams.stb3.vf_coef,
            max_grad_norm=cfg.hyperparams.stb3.max_grad_norm,
            seed=cfg.random_seed,
            device=device,
            policy_kwargs=dict(
                log_std_init=cfg.hyperparams.stb3.log_std_init,
                net_arch=net_arch,
                activation_fn=activation_fn,
                ortho_init=cfg.hyperparams.stb3.ortho_init,
                normalize_images=False,
                enable_critic_lstm=cfg.hyperparams.stb3.enable_critic_lstm,
                lstm_hidden_size=cfg.hyperparams.stb3.lstm_hidden_size,
            ),
        )

    logger_callback = LoggerCallback(
        logger=logger,
        visualization_history=visualization_history,
        goals_to_vis=visualization_goals,
        params_to_vis=visualization_params,
        test_dataloader=test_dataset,
        cfg=cfg,
        device=device,
        tolerable_accuracy_error=cfg.tolerable_accuracy_error,
        trial=trial
    )

    # Train the model
    trained_model = model.learn(
        total_timesteps=cfg.hyperparams.stb3.total_timesteps,
        callback=logger_callback,
        progress_bar=True,
    )

    if cfg.save_trained_model:
        # Save Model
        date_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        recurrent_string = "Recurrent_" if cfg.hyperparams.stb3.use_recurrent_policy else ""
        path = cfg.model_save_dir + "/" + "PPO_" + recurrent_string + date_time_string
        model.save(path=path)
        logger.upload_model(path=path)

    return trained_model.logger.name_to_value["train/loss"]


def make_environment(device, cfg, tensor_type):
    return KinematicsEnvironment(device, cfg=cfg, tensor_type=tensor_type)
