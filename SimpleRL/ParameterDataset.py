from torch.utils.data import Dataset

from DataGeneration import DH_2L_Generator as dhGen

class CustomParameterDataset(Dataset):
    def __init__(self, random_data_length=10000):
        self.dh_parameters = []
        self.goal = []
        for i in range(random_data_length):
            dh_param = dhGen.generate_extended_planar_2_link_dh()
            self.dh_parameters.append(dh_param)
            goal = dhGen.generate_achievable_goal(dh_param)
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