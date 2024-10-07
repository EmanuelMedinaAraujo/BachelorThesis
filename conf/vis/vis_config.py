from dataclasses import dataclass

@dataclass
class AnalyticalVisualization:
    interval: int
    max_history_length: int
    default_line_transparency: float
    default_line_width: float
    use_gradual_transparency: bool
    plot_all_in_one: bool
    distribution_samples: int


@dataclass
class StB3Visualization:
    visualize_distribution: bool
    visualize_value_loss: bool
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
    num_problems_to_visualize: int
    analytical: AnalyticalVisualization
    stb3: StB3Visualization
