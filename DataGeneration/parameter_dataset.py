import torch
from torch.utils.data import Dataset

from DataGeneration.goal_generator import generate_achievable_goal
from DataGeneration.parameter_generator import ParameterGeneratorForPlanarRobot


class CustomParameterDataset(Dataset):
    """
    A torch dataset that generates random DH parameters and an achievable goal for a planar robot.
    The Parameters are generated using the ParameterGeneratorForPlanarRobot class.
    """

    def __init__(self, length=10000,
                 device_to_use=None,
                 num_of_joints=2,
                 parameter_convention='DH',
                 tensor_type=torch.float32,
                 min_link_len=0.1,
                 max_link_len=20.):
        self.dh_parameters = []
        self.goal = []
        self.length = length
        generator = ParameterGeneratorForPlanarRobot(batch_size=length,
                                                     device=device_to_use,
                                                     tensor_type=tensor_type,
                                                     num_joints=num_of_joints,
                                                     parameter_convention=parameter_convention,
                                                     min_len=min_link_len,
                                                     max_len=max_link_len
                                                     )
        self.dh_parameters = generator.get_random_dh_parameters()
        self.goal = generate_achievable_goal(self.dh_parameters, device_to_use)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.dh_parameters[idx], self.goal[idx]
