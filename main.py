import sys
from typing import Callable, Union

import hydra
import optuna
import torch
from hydra.core.config_store import ConfigStore
from optuna import Study, TrialPruned
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from torch import nn as nn
from tqdm import tqdm
from wandb.integration.sb3 import WandbCallback

from analyticalRL.network_eval import test_loop, train_loop
from analyticalRL.networks.distributions.two_parameter_distributions.kinematics_network_reparam_beta_dist import \
    KinematicsNetworkBetaDist
from analyticalRL.networks.distributions.two_parameter_distributions.normal_distributions.kinematics_network_norm_dist import \
    KinematicsNetworkNormDist
from analyticalRL.networks.distributions.two_parameter_distributions.normal_distributions.kinematics_network_rand_sample import \
    KinematicsNetworkRandomSampleDist
from analyticalRL.networks.distributions.two_parameter_distributions.normal_distributions.kinematics_network_reparam_dist import \
    KinematicsNetworkReparamDist
from analyticalRL.networks.simple_kinematics_network import SimpleKinematicsNetwork
from conf.conf_dataclasses.config import TrainConfig
from custom_logging.custom_loggger import GeneralLogger
from custom_logging.logger_callback import LoggerCallback
from data_generation.parameter_dataset import CustomParameterDataset
from data_generation.parameter_generator import ParameterGeneratorForPlanarRobot
from stb3.kinematics_environment import KinematicsEnvironment
from vis.model_type_vis.analytical_vis import visualize_analytical_problem, visualize_analytical_distribution

cs = ConfigStore.instance()
cs.store(name="train_conf", node=TrainConfig)


def copy_cfg(cfg: TrainConfig) -> TrainConfig:
    return TrainConfig(
        hyperparams=cfg.hyperparams,
        logging=cfg.logging,
        vis=cfg.vis,
        optuna=cfg.optuna,
        random_seed=cfg.random_seed,
        use_stb3=cfg.use_stb3,
        do_vis=cfg.do_vis,
        use_optuna=cfg.use_optuna,
        server_postfix=cfg.server_postfix,
        torch_num_threads=cfg.torch_num_threads,
        number_of_joints=cfg.number_of_joints,
        tolerable_accuracy_error=cfg.tolerable_accuracy_error,
        parameter_convention=cfg.parameter_convention,
        min_link_length=cfg.min_link_length,
        max_link_length=cfg.max_link_length,
        number_of_test_problems=cfg.number_of_test_problems,
    )


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(train_config: TrainConfig):
    set_random_seed(train_config.random_seed)

    torch.set_num_threads(train_config.torch_num_threads)
    # th.autograd.set_detect_anomaly(True)

    if not train_config.use_optuna:
        train_and_test_model(copy_cfg(train_config))
        return

    # Use Optuna
    minimal_steps = train_config.optuna.min_num_steps
    num_processes = train_config.optuna.num_processes
    num_trials_per_process = train_config.optuna.trials_per_process
    try:
        optuna.delete_study(study_name='analytical', storage=f'sqlite:///analytical.db')
    except KeyError:
        pass
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=minimal_steps),
                                direction='maximize',
                                study_name='analytical',
                                storage=f'sqlite:///analytical.db',
                                )

    arguments = [(study, copy_cfg(train_config), num_trials_per_process) for _ in range(num_processes)]
    if num_processes == 1:
        _optimize(study, train_config, num_trials_per_process)
    else:
        torch.multiprocessing.set_start_method('spawn')
        with torch.multiprocessing.Pool(processes=num_processes) as p:
            p.starmap(_optimize, arguments)

    print_optuna_results(study)


def _optimize(study: Study, train_config, num_trials_per_process):
    study.optimize(lambda trial: _objective(train_config, trial), n_trials=num_trials_per_process,
                   catch=[ValueError, ZeroDivisionError, RuntimeError, TrialPruned])


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func


def _objective(defaults: TrainConfig, trial: optuna.Trial):
    """Copies the default config and adds the trial parameters to it."""
    cfg_copy = copy_cfg(defaults)
    lr = trial.suggest_float('learning_rate', 1e-5, 1, log=True)
    batch_size_exp = trial.suggest_int('batch_size', 4, 9)  # from 16 to 512
    batch_size = 2 ** batch_size_exp
    if defaults.use_stb3:
        n_envs_exp = trial.suggest_int('n_envs', 0, 4)  # from 1 to 256 in 4 base
        cfg_copy.hyperparams.stb3.n_envs = 4 ** n_envs_exp

        # Ranges from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py
        cfg_copy.hyperparams.stb3.batch_size = batch_size
        n_steps_exp = trial.suggest_int('n_steps', 1, 11)  # from 16 to 2048
        cfg_copy.hyperparams.stb3.n_steps = 2 ** n_steps_exp
        cfg_copy.hyperparams.stb3.learning_rate = lr
        cfg_copy.hyperparams.stb3.ent_coef = trial.suggest_float('ent_coef', 1e-8, 0.1)
        clip_range_factor = trial.suggest_int('clip_range', 1, 4)  # from [0.1, 0.2, 0.3, 0.4]
        cfg_copy.hyperparams.stb3.clip_range = 0.1 * clip_range_factor
        cfg_copy.hyperparams.stb3.epochs = trial.suggest_categorical("epochs", [1, 5, 10, 20])
        cfg_copy.hyperparams.stb3.gae_lambda = trial.suggest_categorical("gae_lambda",
                                                                         [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        cfg_copy.hyperparams.stb3.max_grad_norm = trial.suggest_categorical("max_grad_norm",
                                                                            [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
        cfg_copy.hyperparams.stb3.vf_coef = trial.suggest_float('vf_coef', 0., 1.)

        # new Parameters
        cfg_copy.hyperparams.stb3.net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
        cfg_copy.hyperparams.stb3.ortho_init = trial.suggest_categorical('ortho_init', [False, True])
        cfg_copy.hyperparams.stb3.activation_fn_name = trial.suggest_categorical("activation_fn",
                                                                                 ["tanh", "relu", "elu", "leaky_relu"])
        # TODO fix this
        # cfg_copy.hyperparams.stb3.lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
        # if cfg_copy.hyperparams.stb3.lr_schedule == "linear":
        #     cfg_copy.hyperparams.stb3.learning_rate = linear_schedule(cfg_copy.hyperparams.stb3.learning_rate)

        cfg_copy.hyperparams.stb3.log_std_init = trial.suggest_float('log_std_init', -1.6, 0)
        cfg_copy.hyperparams.stb3.norm_advantages = trial.suggest_categorical('norm_advantages', [True, False])

        cfg_copy.hyperparams.stb3.use_recurrent_policy = trial.suggest_categorical('use_recurrent_policy',
                                                                                   [True, False])
        if cfg_copy.hyperparams.stb3.use_recurrent_policy:
            cfg_copy.hyperparams.stb3.enable_critic_lstm = trial.suggest_categorical("enable_critic_lstm",
                                                                                     [False, True])
            lstm_hidden_size_exp = trial.suggest_int('batch_size', 4, 9)  # from 16 to 512
            cfg_copy.hyperparams.stb3.lstm_hidden_size = 2 ** lstm_hidden_size_exp

    else:
        cfg_copy.hyperparams.analytical.learning_rate = lr
        cfg_copy.hyperparams.analytical.batch_size = batch_size
        cfg_copy.hyperparams.analytical.num_hidden_layer = trial.suggest_int('num_hidden_layer', 1, 50)
        cfg_copy.hyperparams.analytical.hidden_layer_sizes = [trial.suggest_int(f'hidden_layer_sizes_{i}', 1, 2048) for
                                                              i in
                                                              range(cfg_copy.hyperparams.analytical.num_hidden_layer)]
        cfg_copy.hyperparams.analytical.problems_per_epoch = trial.suggest_int('problems_per_epoch', 1, 100000,
                                                                               log=True)
        cfg_copy.hyperparams.analytical.epochs = trial.suggest_int('epochs', 1, 1028)
        cfg_copy.hyperparams.analytical.testing_interval = trial.suggest_int('testing_interval', 1,
                                                                             cfg_copy.hyperparams.analytical.epochs)
        # noinspection SpellCheckingInspection
        cfg_copy.hyperparams.analytical.optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])

    return train_and_test_model(cfg_copy, trial)


def train_and_test_model(train_config: TrainConfig, trial: optuna.Trial = None):
    """
    Train and test the model with the given hydra configuration.
    """
    device = (
        "cuda" + str(train_config.server_postfix)
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    tensor_type = torch.float32

    test_dataset = CustomParameterDataset(
        length=train_config.number_of_test_problems,
        device_to_use=device,
        num_of_joints=train_config.number_of_joints,
        parameter_convention=train_config.parameter_convention,
        min_link_len=train_config.min_link_length,
        max_link_len=train_config.max_link_length,
    )

    visualization_params = []
    visualization_goals = []
    visualization_ground_truth = []
    for x in range(train_config.vis.num_problems_to_visualize):
        param, goal, ground_truth = test_dataset.__getitem__(x)
        visualization_params.append(param)
        visualization_goals.append(goal)
        visualization_ground_truth.append(ground_truth)

    visualization_history = []

    logger = None
    exit_code = 3
    try:
        logger = GeneralLogger(cfg=train_config)
        logger.log_used_device(device=device)
        if train_config.use_stb3:
            eval_score = do_stable_baselines3_learning(
                device=device,
                cfg=train_config,
                logger=logger,
                test_dataset=test_dataset,
                visualization_history=visualization_history,
                visualization_goals=visualization_goals,
                visualization_params=visualization_params,
                tensor_type=tensor_type,
                trial=trial
            )
        else:
            eval_score = do_analytical_learning(
                device=device,
                cfg=train_config,
                logger=logger,
                test_dataset=test_dataset,
                visualization_history=visualization_history,
                visualization_goals=visualization_goals,
                visualization_params=visualization_params,
                visualization_ground_truth=visualization_ground_truth,
                tensor_type=tensor_type,
                trial=trial
            )
    except TrialPruned as trial_pruned_exception:
        logger.finish_logging(1)
        raise trial_pruned_exception
    except (ValueError, ZeroDivisionError, RuntimeError) as e:
        logger.finish_logging(2)
        raise e
    else:
        exit_code = 0
    finally:
        logger.finish_logging(exit_code)
    return eval_score


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
    trained_model = model.learn(
        total_timesteps=cfg.hyperparams.stb3.total_timesteps,
        callback=logger_callback,
        progress_bar=True,
    )
    return trained_model.logger.name_to_value["train/loss"]


def do_analytical_learning(device, cfg: TrainConfig, logger, test_dataset, visualization_history, visualization_goals,
                           visualization_params, visualization_ground_truth, tensor_type, trial: optuna.Trial = None,
                           ):
    match cfg.hyperparams.analytical.output_type:
        case "Normal":
            model = SimpleKinematicsNetwork(
                num_joints=cfg.number_of_joints,
                num_layer=cfg.hyperparams.analytical.num_hidden_layer,
                layer_sizes=cfg.hyperparams.analytical.hidden_layer_sizes,
                logger=logger,
            ).to(device)
        case "NormDist":
            model = KinematicsNetworkNormDist(
                num_joints=cfg.number_of_joints,
                num_layer=cfg.hyperparams.analytical.num_hidden_layer,
                layer_sizes=cfg.hyperparams.analytical.hidden_layer_sizes,
                logger=logger,
            ).to(device)
        case "ReparameterizationDist":
            model = KinematicsNetworkReparamDist(
                num_joints=cfg.number_of_joints,
                num_layer=cfg.hyperparams.analytical.num_hidden_layer,
                layer_sizes=cfg.hyperparams.analytical.hidden_layer_sizes,
                logger=logger,
            ).to(device)
        case "RandomSampleDist":
            model = KinematicsNetworkRandomSampleDist(
                num_joints=cfg.number_of_joints,
                num_layer=cfg.hyperparams.analytical.num_hidden_layer,
                layer_sizes=cfg.hyperparams.analytical.hidden_layer_sizes,
                logger=logger,
            ).to(device)
        case "BetaDist":
            model = KinematicsNetworkBetaDist(
                num_joints=cfg.number_of_joints,
                num_layer=cfg.hyperparams.analytical.num_hidden_layer,
                layer_sizes=cfg.hyperparams.analytical.hidden_layer_sizes,
                logger=logger,
            ).to(device)
        case _:
            raise ValueError(
                f"Unknown output type: {cfg.hyperparams.analytical.output_type}. Please adjust the config.")

    logger.watch_model(model, cfg.logging.wandb.wand_callback_logging_freq)

    # Use optimizer specified in the config
    optimizer = getattr(torch.optim, cfg.hyperparams.analytical.optimizer)(
        model.parameters(), lr=cfg.hyperparams.analytical.learning_rate,  # maximize=True,
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
    last_mean_loss = None
    if cfg.do_vis:
        with torch.no_grad():
            for i in range(cfg.vis.num_problems_to_visualize):
                if cfg.hyperparams.analytical.output_type == "Normal":
                    visualize_analytical_problem(
                        model=model,
                        param=visualization_params[i],
                        goal=visualization_goals[i],
                        param_history=visualization_history,
                        cfg=cfg,
                        logger=logger,
                        current_step=0,
                        chart_index=i + 1,
                    )
                else:
                    visualize_analytical_distribution(
                        model=model,
                        param=visualization_params[i],
                        goal=visualization_goals[i],
                        ground_truth=visualization_ground_truth[i],
                        cfg=cfg,
                        logger=logger,
                        current_step=0,
                        chart_index=i + 1,
                        device=device,
                    )
    for epoch_num in tqdm(
            range(cfg.hyperparams.analytical.epochs),
            colour="green",
            file=sys.stdout
    ):
        # Keep track of the last mean loss for optuna
        last_mean_loss = train_loop(
            model=model,
            optimizer=optimizer,
            problem_generator=problem_generator,
            problems_per_epoch=cfg.hyperparams.analytical.problems_per_epoch,
            batch_size=cfg.hyperparams.analytical.batch_size,
            device=device,
            logger=logger,
            epoch_num=epoch_num,
            error_tolerance=cfg.tolerable_accuracy_error,
            is_normal_output=cfg.hyperparams.analytical.output_type == "Normal"
        )

        # Test the model every hyperparams.testing_interval epochs
        if epoch_num % cfg.hyperparams.analytical.testing_interval == 0:
            test_loop(
                test_dataset=test_dataset,
                model=model,
                logger=logger,
                tolerable_accuracy_error=cfg.tolerable_accuracy_error,
                num_epoch=epoch_num,
                is_normal_output=cfg.hyperparams.analytical.output_type == "Normal"
            )

        # Visualize the same problem every hyperparams.visualization.interval epochs
        if cfg.do_vis and epoch_num % cfg.vis.analytical.interval == 0 and epoch_num != 0:
            with torch.no_grad():
                for i in range(cfg.vis.num_problems_to_visualize):
                    if cfg.hyperparams.analytical.output_type == "Normal":
                        visualize_analytical_problem(
                            model=model,
                            param=visualization_params[i],
                            goal=visualization_goals[i],
                            param_history=visualization_history,
                            cfg=cfg,
                            logger=logger,
                            current_step=epoch_num,
                            chart_index=i + 1,
                        )
                    else:
                        visualize_analytical_distribution(
                            model=model,
                            param=visualization_params[i],
                            goal=visualization_goals[i],
                            ground_truth=visualization_ground_truth[i],
                            cfg=cfg,
                            logger=logger,
                            current_step=epoch_num,
                            chart_index=i + 1,
                            device=device,
                        )
        if trial is not None:
            trial.report(last_mean_loss, epoch_num)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return last_mean_loss


def print_optuna_results(study):
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def make_environment(device, cfg, tensor_type):
    return KinematicsEnvironment(device, cfg=cfg, tensor_type=tensor_type)


if __name__ == "__main__":
    main()
