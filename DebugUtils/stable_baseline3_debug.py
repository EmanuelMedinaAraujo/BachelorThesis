import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from PPO.kinematics_environment import KinematicsEnvironment
from PPO.logger_callback import LoggerCallback
from Util.forward_kinematics import calculate_distances, update_theta_values

DH_EXAMPLE = torch.tensor(
    [[0, 13.5994, 0],
     [0, 18.755, 0]
     ]
)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
solution = torch.tensor([-4.8549, -2.336])
robot_goal = torch.tensor([13.4757, -1.3202])
kin_env = KinematicsEnvironment(device, DH_EXAMPLE, goal_coordinates=robot_goal, num_joints=2)
check_env(kin_env)
env = make_vec_env(lambda: kin_env, n_envs=1)
# Define the model
model = PPO("MlpPolicy", env, verbose=2)

# Train the model
model.learn(total_timesteps=1,
            progress_bar=True)

# Save the model
# model.save("ppo_kinematics")
# model = PPO.load("ppo_kinematics")

obs = env.reset()
action, _states = model.predict(obs)
print(f"Predicted Angles: {action}, but the solution is: {solution}")
# Print the distance between the predicted and the actual angles
distance = calculate_distances(update_theta_values(DH_EXAMPLE, action[0]), robot_goal)
print(f"Distance: {distance}")

model.learn(total_timesteps=5000, callback=LoggerCallback(None, None, None, 1, ),
            progress_bar=True)
obs = env.reset()
action, _states = model.predict(obs)
print(f"Predicted Angles: {action}, but the solution is: {solution}")
# Print the distance between the predicted and the actual angles
distance = calculate_distances(update_theta_values(DH_EXAMPLE, action[0]), robot_goal)
print(f"Distance: {distance}")
