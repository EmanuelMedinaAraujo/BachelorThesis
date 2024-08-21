import torch
from torch.utils.data import DataLoader

from SimpleRL.parameter_dataset import CustomParameterDataset

"""
This script is used to rapidly debug the CustomParameterDataset class and the DataLoader class.
"""

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

dataloader = DataLoader(CustomParameterDataset(length=3, device_to_use=device), batch_size=64, shuffle=True)
train_param, train_goal = next(iter(dataloader))
dh_param = train_param[0]
goal_coordinate = train_goal[0]
print(f"dhParam: {dh_param}")
print(f"goal end effector: {goal_coordinate}")
