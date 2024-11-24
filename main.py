import hydra
import torch
from hydra.core.config_store import ConfigStore
from stable_baselines3.common.utils import set_random_seed

from conf.conf_dataclasses.config import TrainConfig, copy_cfg
from hyperparameter_optimization.optuna_utils import run_optuna
from networks.train_and_test_models import train_and_test_model

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


if __name__ == "__main__":
    main()
