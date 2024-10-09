import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from conf.conf_dataclasses.config import TrainConfig
from data_generation.goal_generator import generate_achievable_goal
from data_generation.parameter_generator import ParameterGeneratorForPlanarRobot
from util.forward_kinematics import update_theta_values, calculate_parameter_goal_distances, calculate_angles_from_network_output
from vis.planar_robot_vis import visualize_planar_robot


class KinematicsEnvironment(gym.Env):
    def __init__(self, device, cfg: TrainConfig, tensor_type):
        super(KinematicsEnvironment, self).__init__()

        self.device = device
        self.tolerable_accuracy_error = cfg.tolerable_accuracy_error
        self.max_legend_length = cfg.vis.max_legend_length
        self.show_joints = cfg.vis.show_joints
        self.show_end_effector = cfg.vis.show_end_effectors

        self.num_joints = cfg.number_of_joints
        self.problem_generator = ParameterGeneratorForPlanarRobot(
            batch_size=1,
            device=device,
            tensor_type=tensor_type,
            num_joints=self.num_joints,
            parameter_convention=cfg.parameter_convention,
            min_len=cfg.min_link_length,
            max_len=cfg.max_link_length,
        )

        self.action_space = spaces.Box(
            low=-np.pi, high=np.pi, shape=(self.num_joints*2,), dtype=np.float32
        )

        # The observation space is the concatenation of the parameter and the goal
        observation_dimension = self.num_joints * 3 + 2
        # The max observation value is the max link length times the number of joints plus the goal
        max_observation_value = cfg.max_link_length * self.num_joints * 2
        self.observation_space = spaces.Box(
            low=-max_observation_value,
            high=max_observation_value,
            shape=(observation_dimension,),
            dtype=np.float32,
        )

        self.parameter = self.problem_generator.get_random_parameters()
        self.goal, _ = generate_achievable_goal(self.parameter, self.device)

    def reset(self, seed=None, options=None):
        self.parameter = self.problem_generator.get_random_parameters()
        self.goal, _ = generate_achievable_goal(self.parameter, self.device)

        return (
            torch.concat([self.parameter.flatten(), self.goal]).detach().cpu().numpy(),
            {},
        )

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.tensor(action).to(self.device)
        # Calculate angles from the network output
        all_angles = calculate_angles_from_network_output(action, self.num_joints, self.device)

        updated_parameter = update_theta_values(self.parameter, all_angles)
        distance = calculate_parameter_goal_distances(updated_parameter, self.goal).detach().item()
        reward = -distance
        done = True  # One-step episode
        success = distance <= self.tolerable_accuracy_error
        observation = (
            torch.concat([self.parameter.flatten(), self.goal]).detach().cpu().numpy()
        )
        return observation, reward, done, success, {}

    def render_action(self, action):
        if isinstance(action, np.ndarray):
            action = torch.tensor(action).to(self.device)
            # Calculate angles from the network output
        all_angles = calculate_angles_from_network_output(action, self.num_joints, self.device)

        updated_parameter = update_theta_values(self.parameter, all_angles)
        # Visualize the robot arm using the updated parameters and using visualize_planar_robot
        visualize_planar_robot(parameter=updated_parameter, default_line_transparency=1.0, default_line_width=1.5,
                               max_legend_length=self.max_legend_length, goal=self.goal, show_joints=self.show_joints,
                               show_end_effectors=self.show_end_effector, show_distance=True)

    def set_goal(self, new_goal):
        self.goal = new_goal

    def set_parameter(self, new_parameter):
        self.parameter = new_parameter

    def get_goal(self):
        return self.goal

    def get_parameter(self):
        return self.parameter