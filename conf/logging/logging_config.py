from dataclasses import dataclass


@dataclass
class WandBLogging:
    log_in_wandb: bool
    log_visualization_plots: bool
    wand_callback_logging_freq: int
    project_name: str


@dataclass
class Logging:
    wandb: WandBLogging
    log_in_console: bool
    log_architecture: bool
