from typing import Callable, Union

import optuna
import torch
from optuna import Study
from optuna import TrialPruned

from conf.conf_dataclasses.config import TrainConfig, copy_cfg
from networks.train_and_test_models import train_and_test_model


def run_optuna(train_config):
    minimal_steps = train_config.optuna.min_num_steps
    num_processes = train_config.optuna.num_processes
    num_trials_per_process = train_config.optuna.trials_per_process
    try:
        optuna.delete_study(study_name='distribution_optuna_2p_lstm',
                            storage=f'sqlite:///distribution_optuna_2p_lstm.db')
    except KeyError:
        pass
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=minimal_steps),
                                direction='maximize' if train_config.use_stb3 else 'minimize',
                                study_name='distribution_optuna_2p_lstm',
                                storage=f'sqlite:///distribution_optuna_2p_lstm.db',
                                )
    arguments = [(study, copy_cfg(train_config), num_trials_per_process) for _ in range(num_processes)]
    if num_processes == 1:
        _optimize(study, train_config, num_trials_per_process)
    else:
        torch.multiprocessing.set_start_method('spawn')
        with torch.multiprocessing.Pool(processes=num_processes) as p:
            p.starmap(_optimize, arguments)
    print_optuna_results(study)


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


def _optimize(study: Study, train_config: TrainConfig, num_trials_per_process: int):
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
        cfg_copy.hyperparams.analytical.problems_per_epoch = batch_size * trial.suggest_int('problems_per_epoch', 1, 20)

        # noinspection SpellCheckingInspection
        cfg_copy.hyperparams.analytical.optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
        cfg_copy.hyperparams.analytical.output_type = trial.suggest_categorical('output_type',
                                                                                ['TwoPeakNormalLstmDistrNetwork',
                                                                                 'TwoPeakNormalLstmVariantDistrNetwork'])

        # lstm hyperparameters
        if 'lstm' in cfg_copy.hyperparams.analytical.output_type.lower():
            cfg_copy.hyperparams.analytical.lstm_hidden_size = 2 ** trial.suggest_int('hidden_size', 5, 11)
            cfg_copy.hyperparams.analytical.lstm_num_layers = trial.suggest_int('lstm_layers', 1, 5)
            if cfg_copy.hyperparams.analytical.output_type == 'TwoPeakNormalLstmVariantDistrNetwork':
                cfg_copy.hyperparams.analytical.num_hidden_layer = trial.suggest_int('num_hidden_layer', 1, 5)
                cfg_copy.hyperparams.analytical.hidden_layer_sizes = [
                    2 ** trial.suggest_int(f'hidden_layer_sizes_{i}', 3, 9)
                    for
                    i in
                    range(cfg_copy.hyperparams.analytical.num_hidden_layer)]
        else:
            cfg_copy.hyperparams.analytical.num_hidden_layer = trial.suggest_int('num_hidden_layer', 1, 20)
            cfg_copy.hyperparams.analytical.hidden_layer_sizes = [
                2 ** trial.suggest_int(f'hidden_layer_sizes_{i}', 1, 11)
                for
                i in
                range(cfg_copy.hyperparams.analytical.num_hidden_layer)]

    return train_and_test_model(cfg_copy, trial)
