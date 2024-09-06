import gymnasium as gym
import numpy as np
from gymnasium import spaces

from Util.forward_kinematics import update_theta_values, calculate_distances
from Visualization.planar_robot_vis import visualize_planar_robot


class KinematicsEnvironment(gym.Env):
    def __init__(self, device, dataloader, num_joints,
                 tolerable_accuracy_error):
        super(KinematicsEnvironment, self).__init__()

        self.device = device
        self.tolerable_accuracy_error = tolerable_accuracy_error
        self.dataloader = dataloader

        self.num_joints = num_joints
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1000, dtype=np.float32)

        self.goals = None
        self.parameters = None
        self.reset()

    def reset(self, seed=None, options=None):
        self.parameters, self.goals = next(iter(self.dataloader))
        return np.array([1000]).astype(np.float32), {}

    def step(self, action):
        updated_parameter = update_theta_values(self.parameters[0], action)
        distance = calculate_distances(updated_parameter, self.goals[0]).detach().item()
        reward = -distance
        done = True  # One-step episode
        success = distance <= self.tolerable_accuracy_error
        observation = np.array([distance]).astype(np.float32)
        return observation, reward, done, success, {}

    def render_action(self, action):
        updated_parameter = update_theta_values(self.parameters[0], action)
        # Visualize the robot arm using the updated parameters and using visualize_planar_robot
        visualize_planar_robot(parameter=updated_parameter, goal=self.goals[0], default_line_transparency=1.,
                               default_line_width=1.5,
                               frame_size_scalar=1.1, device=self.device)

    def set_goal(self, new_goal):
        self.goal = new_goal

    def set_parameter(self, new_parameter):
        self.parameter = new_parameter

    def get_goal(self):
        return self.goal

    def get_parameter(self):
        return self.parameter
