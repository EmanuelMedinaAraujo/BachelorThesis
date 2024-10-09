from dataclasses import dataclass


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
    gae_lambda: float
    clip_range: float
    norm_advantages: bool
    vf_coef: float
    max_grad_norm: float
    net_arch_type: str
    ortho_init: bool
    activation_fn_name: str
    lr_schedule: str
    enable_critic_lstm: bool
    lstm_hidden_size: int

@dataclass
class AnalyticalHyperparams:
    num_hidden_layer: int
    hidden_layer_sizes: list
    problems_per_epoch: int
    learning_rate: float
    batch_size: int
    epochs: int
    testing_interval: int
    optimizer: str
    output_type: str


@dataclass
class Hyperparams:
    analytical: AnalyticalHyperparams
    stb3: StB3Hyperparams
