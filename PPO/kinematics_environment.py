import gymnasium as gym
import numpy as np
from gymnasium import spaces

from DataGeneration.goal_generator import generate_achievable_goal
from DataGeneration.parameter_generator import ParameterGeneratorForPlanarRobot
from Util.forward_kinematics import update_theta_values, calculate_distances
from Visualization.planar_robot_vis import visualize_planar_robot


class KinematicsEnvironment(gym.Env):
    def __init__(self, device, hyperparams, tensor_type):
        super(KinematicsEnvironment, self).__init__()

        self.device = device
        self.tolerable_accuracy_error = hyperparams.tolerable_accuracy_error
        self.num_joints = hyperparams.number_of_joints

        self.problem_generator = ParameterGeneratorForPlanarRobot(batch_size=1,
                                                                  device=device,
                                                                  tensor_type=tensor_type,
                                                                  num_joints=self.num_joints,
                                                                  parameter_convention=hyperparams.parameter_convention,
                                                                  min_len=hyperparams.min_link_length,
                                                                  max_len=hyperparams.max_link_length)

        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1000, dtype=np.float32)

        self.parameter = self.problem_generator.get_random_parameters()
        self.goal = generate_achievable_goal(self.parameter, self.device)

    def reset(self, seed=None, options=None):
        self.parameter = self.problem_generator.get_random_parameters()
        self.goal = generate_achievable_goal(self.parameter, self.device)
        return np.array([1000]).astype(np.float32), {}

    def step(self, action):
        updated_parameter = update_theta_values(self.parameter, action)
        distance = calculate_distances(updated_parameter, self.goal).detach().item()
        reward = -distance
        done = True  # One-step episode
        success = distance <= self.tolerable_accuracy_error
        observation = np.array([distance]).astype(np.float32)
        return observation, reward, done, success, {}

    def render_action(self, action):
        updated_parameter = update_theta_values(self.parameter, action)
        # Visualize the robot arm using the updated parameters and using visualize_planar_robot
        visualize_planar_robot(parameter=updated_parameter,
                               goal=self.goal,
                               default_line_transparency=1.,
                               default_line_width=1.5,
                               frame_size_scalar=1.1,
                               device=self.device,
                               standard_size=True,
                               show_distance=True)

    def set_goal(self, new_goal):
        self.goal = new_goal

    def set_parameter(self, new_parameter):
        self.parameter = new_parameter

    def get_goal(self):
        return self.goal

    def get_parameter(self):
        return self.parameter
