import torch
from torch.utils.data import Dataset
from DataGeneration import DH_2L_Generator as dhGen
from DataGeneration.ParameterGenerator import ParameterGenerator
from random import *


class CustomParameterDataset(Dataset):
    def __init__(self, length=10000, device_to_use = None):
        self.dh_parameters = []
        self.goal = []
        generator = ParameterGenerator(amount_parameters_to_generate = length, device = device_to_use)
        self.dh_parameters = generator.get_random_dh_parameters()
        for i in range(length):
            #dh_param = dhGen.generate_extended_planar_2_link_dh()
            #self.dh_parameters.append(dh_param)
            goal = generate_achievable_goal(self.dh_parameters[i])
            self.goal.append(goal)
        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.dh_parameters)

    def __getitem__(self, idx):
        param = self.dh_parameters[idx]
        goal = self.goal[idx]
        # if self.transform:
        #     input = self.transform(param, goal)
        # if self.target_transform:
        #     target = self.target_transform(target)
        return param, goal

def generate_achievable_goal( dh_parameter: torch.Tensor):
    # Sum of second column of DH parameters
    arm_length = int(dh_parameter[:,1].sum().item())
    random_reachable_length = randint(0, arm_length)
    return torch.tensor([0, random_reachable_length],dtype=torch.float32)