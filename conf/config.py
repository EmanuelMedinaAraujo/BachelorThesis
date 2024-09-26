from dataclasses import dataclass

from conf.hyperparams.hyperparams_config import Hyperparams
from conf.logging.logging_config import Logging
from conf.vis.vis_config import Visualization

@dataclass
class OptunaConfig:
    min_num_steps: int
    num_processes: int
    trials_per_process: int


@dataclass
class TrainConfig:
    hyperparams: Hyperparams
    logging: Logging
    vis: Visualization
    optuna: OptunaConfig
    random_seed: int
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
