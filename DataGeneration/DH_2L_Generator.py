import torch
from random import *

# DH Parameter of 2 Link Planar Robot with extended arm (alpha, a, d, theta)
# DH_EXAMPLE = torch.tensor([
#     [0, 15, 0, 0],
#     [0, 10, 0, 0]
# ])

def generate_extended_planar_2_link_dh():
    dh_parameter = torch.rand(2,4, dtype=torch.float32)
    dh_parameter[:,0] = 0
    dh_parameter[:,1] = dh_parameter[:,1] * 10
    dh_parameter[:,2] = 0
    dh_parameter[:,3] = 0
    return dh_parameter

def generate_planar_2_link_dh():
    dh_parameter = torch.rand(2,4, dtype=torch.float32)
    return dh_parameter

def generate_achievable_goal( dh_parameter: torch.Tensor):
    # Sum of second column of DH parameters
    arm_length = int(dh_parameter[:,1].sum().item())
    random_reachable_length = randint(0, arm_length)
    return torch.tensor([0, random_reachable_length],dtype=torch.float32)