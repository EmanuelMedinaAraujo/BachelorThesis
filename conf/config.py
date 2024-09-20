from dataclasses import dataclass


@dataclass
class AnalyticalHyperparams:
    num_layer: int
    layer_sizes: list
    problems_per_epoch: int
    learning_rate: float
    batch_size: int
    epochs: int
    testing_interval: int


@dataclass
class StB3Hyperparams:
    use_recurrent_policy: bool
    non_recurrent_policy: str
    recurrent_policy: str
    n_envs: int
    n_steps: int
    total_timesteps: int
    learning_rate: float
    batch_size: int
    epochs: int
    gamma: float
    ent_coef: float
    log_std_init: float
    testing_interval: int


@dataclass
class Hyperparams:
    analytical: AnalyticalHyperparams
    stb3: StB3Hyperparams


@dataclass
class AnalyticalVisualization:
    interval: int
    max_history_length: int
    default_line_transparency: float
    default_line_width: float
    use_gradual_transparency: bool
    plot_all_in_one: bool


@dataclass
class StB3Visualization:
    visualize_distribution: bool
    num_distribution_samples: int
    do_heat_map: bool
    interval: int
    default_line_transparency: float
    default_line_width: float


@dataclass
class Visualization:
    show_plot: bool
    show_joint_label: bool
    show_joints: bool
    show_end_effectors: bool
    save_to_file: bool
    show_distance_in_legend: bool
    show_legend: bool
    max_legend_length: int
    analytical: AnalyticalVisualization
    stb3: StB3Visualization


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


@dataclass
class TrainConfig:
    hyperparams: Hyperparams
    logging: Logging
    vis: Visualization
    random_seed: int
    use_stb3: bool
    do_vis: bool
    server_postfix: str
    # Problem Generation Parameter
    number_of_joints: int
    tolerable_accuracy_error: (
        float  # The tolerance in accuracy that is still regarded as correct
    )
    parameter_convention: str
    min_link_length: float
    max_link_length: float
    number_of_test_problems: int
