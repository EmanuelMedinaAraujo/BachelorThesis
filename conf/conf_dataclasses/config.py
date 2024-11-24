from dataclasses import dataclass

from conf.conf_dataclasses.hyperparams_config import Hyperparams
from conf.conf_dataclasses.logging_config import Logging
from conf.conf_dataclasses.optuna_config import OptunaConfig
from conf.conf_dataclasses.vis_config import Visualization

@dataclass
class TrainConfig:
    hyperparams: Hyperparams
    logging: Logging
    vis: Visualization
    optuna: OptunaConfig
    random_seed: int
    save_trained_model: bool
    model_save_dir: str
    use_stb3: bool
    do_vis: bool
    use_optuna: bool
    server_postfix: str
    torch_num_threads: int
    # Problem Generation Parameter
    number_of_joints: int
    tolerable_accuracy_error: (
        float  # The tolerance in accuracy that is still regarded as correct
    )
    parameter_convention: str
    min_link_length: float
    max_link_length: float
    number_of_test_problems: int


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