import hydra
import torch
from hydra.core.config_store import ConfigStore
from stable_baselines3.common.utils import set_random_seed

import optuna
from analyticalRL.analytical_training_and_tests import do_analytical_learning
from conf.conf_dataclasses.config import TrainConfig
from custom_logging.custom_loggger import GeneralLogger
from data_generation.parameter_dataset import CustomParameterDataset
from optuna import TrialPruned
from optuna.optuna import run_optuna
from stb3.stb3_training import do_stable_baselines3_learning

cs = ConfigStore.instance()
cs.store(name="train_conf", node=TrainConfig)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(train_config: TrainConfig):
    set_random_seed(train_config.random_seed)

    torch.set_num_threads(train_config.torch_num_threads)
    # th.autograd.set_detect_anomaly(True)

    if not train_config.use_optuna:
        train_and_test_model(copy_cfg(train_config))
        return

    # Use Optuna
    run_optuna(train_config)


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
    set_random_seed(0)  # Have a unique test set for every trial
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

    set_random_seed(train_config.random_seed)

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
        if logger is not None:
            logger.finish_logging(exit_code)
    return eval_score


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
        model_save_dir=cfg.model_save_dir,
        save_trained_model=cfg.save_trained_model
    )


if __name__ == "__main__":
    main()
