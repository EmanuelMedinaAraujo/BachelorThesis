from dataclasses import dataclass


@dataclass
class OptunaConfig:
    min_num_steps: int
    num_processes: int
    trials_per_process: int
