import gymnasium as gym
import numpy as np
from gymnasium import spaces

from Util.forward_kinematics import update_theta_values, calculate_distances
from Visualization.planar_robot_vis import visualize_planar_robot


class KinematicsEnvironment(gym.Env):
    def __init__(self, device, parameter, goal_coordinates, num_joints):
        super(KinematicsEnvironment, self).__init__()
        self.parameter = parameter
        self.goal = goal_coordinates
        self.num_joints = num_joints
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1000, dtype=np.float32)
        self.state = None
        self.device = device

    def reset(self, seed=None, options=None):
        return np.array([1000]).astype(np.float32), {}

    def step(self, action):
        updated_parameter = update_theta_values(self.parameter, action)
        reward = -calculate_distances(updated_parameter, self.goal).detach().item()
        done = True  # One-step episode
        return np.array([-reward]).astype(np.float32), reward, done, False, {}

    def render(self, mode='human'):
        visualize_planar_robot(self.parameter, goal=self.goal, default_line_transparency=1., default_line_width=1.5,
                               frame_size_scalar=1.1)

    def close(self):
        pass
