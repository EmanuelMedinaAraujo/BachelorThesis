import sys

import hydra
import optuna
from hydra.core.config_store import ConfigStore
import torch
from optuna import Study
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib import RecurrentPPO

from tqdm import tqdm
import torch as th

from analyticalRL.networks.kinematics_network_normal import KinematicsNetwork
from analyticalRL.networks.kinematics_network_norm_dist import KinematicsNetworkNormDist
from analyticalRL.kinematics_network_testing import test_loop
from analyticalRL.kinematics_network_training import train_loop
from analyticalRL.networks.kinematics_network_rand_sample import KinematicsNetworkRandomSampleDist
from analyticalRL.networks.kinematics_network_reparam_beta_dist import KinematicsNetworkBetaDist
from analyticalRL.networks.kinematics_network_reparam_dist import KinematicsNetworkReparamDist
from data_generation.parameter_dataset import CustomParameterDataset
from data_generation.parameter_generator import ParameterGeneratorForPlanarRobot
from custom_logging.custom_loggger import GeneralLogger
from custom_logging.logger_callback import LoggerCallback
from stb3.kinematics_environment import KinematicsEnvironment
from vis.analytical_vis import visualize_analytical_problem, visualize_analytical_distribution
from conf.conf_dataclasses.config import TrainConfig

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
    th.autograd.set_detect_anomaly(True)

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
    study.optimize(lambda trial: _objective(train_config, trial), n_trials=num_trials_per_process)


def _objective(defaults: TrainConfig, trial: optuna.Trial):
    """Copies the default config and adds the trial parameters to it."""
    cfg_copy = copy_cfg(defaults)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size_exp = trial.suggest_int('batch_size', 4, 9)  # from 16 to 512
    batch_size = 2 ** batch_size_exp
    if defaults.use_stb3:
        cfg_copy.hyperparams.stb3.learning_rate = lr
        cfg_copy.hyperparams.stb3.batch_size = batch_size
        cfg_copy.hyperparams.stb3.use_recurrent_policy = trial.suggest_categorical('use_recurrent_policy',
                                                                                   [True, False])

        n_envs_exp = trial.suggest_int('n_envs', 0, 4)  # from 1 to 256 in 4 base
        cfg_copy.hyperparams.stb3.n_envs = 4 ** n_envs_exp

        n_steps_exp = trial.suggest_int('n_steps', 1, 11)  # from 16 to 512
        cfg_copy.hyperparams.stb3.n_steps = 2 ** n_steps_exp

        cfg_copy.hyperparams.stb3.epochs = trial.suggest_int('epochs', 1, 512, log=True)
        cfg_copy.hyperparams.stb3.ent_coef = trial.suggest_float('ent_coef', 0.0, 0.99)
        cfg_copy.hyperparams.stb3.log_std_init = trial.suggest_float('log_std_init', -2, 0)

        cfg_copy.hyperparams.stb3.gae_lambda = trial.suggest_float('gae_lambda', 0.0, 1.0)
        cfg_copy.hyperparams.stb3.clip_range = trial.suggest_float('clip_range', 0.1, 1.0)
        cfg_copy.hyperparams.stb3.norm_advantages = trial.suggest_categorical('norm_advantages', [True, False])
        cfg_copy.hyperparams.stb3.vf_coef = trial.suggest_float('vf_coef', 0.1, 1.5)
        cfg_copy.hyperparams.stb3.max_grad_norm = trial.suggest_float('max_grad_norm', 0.1, 1.0)
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

    visualization_params = []
    visualization_goals = []
    visualization_ground_truth = []
    for x in range(train_config.vis.num_problems_to_visualize):
        param, goal, ground_truth = test_dataset.__getitem__(x)
        visualization_params.append(param)
        visualization_goals.append(goal)
        visualization_ground_truth.append(ground_truth)

    visualization_history = []

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
    tqdm.write("Done!")
    logger.finish_logging()
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
        policy_kwargs=dict(
            log_std_init=cfg.hyperparams.stb3.log_std_init,
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
            policy_kwargs=dict(log_std_init=cfg.hyperparams.stb3.log_std_init,
                               normalize_images=False),
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
            model = KinematicsNetwork(
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

    logger.watch_model(model)

    # Use optimizer specified in the config
    optimizer = getattr(torch.optim, cfg.hyperparams.analytical.optimizer)(
        model.parameters(), lr=cfg.hyperparams.analytical.learning_rate,# maximize=True,
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
                epoche_num=epoch_num,
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
