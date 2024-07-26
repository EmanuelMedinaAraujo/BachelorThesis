import os
import pandas as pd
from torch.utils.data import Dataset

from DataGeneration import DH_2L_Generator as dh_gen

class CustomParameterDataset(Dataset):
    def __init__(self, randomDataLength=10000, transform=None, target_transform=None):
        self.dh_parameters = []
        self.goal = []
        for i in range(randomDataLength):
            dh_param = dh_gen.generate_extended_planar_2_link_dh()
            self.dh_parameters.append(dh_param)
            goal = dh_gen.generate_achievable_goal(dh_param)
            self.goal.append(goal)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dh_parameters)

    def __getitem__(self, idx):
        param = self.dh_parameters[idx]
        goal = self.goal[idx]
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return param, goal