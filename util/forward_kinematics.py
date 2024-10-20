#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 19.06.24
from typing import Union

import numpy as np
import torch
from torch import Tensor

from util.batched import BatchTransform
from util.dh_conventions import dh_to_homogeneous


def forward_kinematics(
        joint_offsets: Union[torch.Tensor, BatchTransform], full: bool = False
) -> BatchTransform:
    """
    Computes the forward kinematics for a given sequence of joint offsets.

    Args:
        joint_offsets: A tensor or Transform3d object containing the joint offsets. The returned forward kinematics
            will be computed for the coordinate systems defined by these relative offsets, i.e., if the joint offsets
            are based on MDH parameters, the FK coordinate frames will be placed directly AFTER the "physical" joint,
            if the joint offsets are based on DH parameters, the FK coordinate frames will be placed directly BEFORE
            the "physical" joint.
        full: If True, the full forward kinematics will be computed, i.e., the output has the same shape as the input.
            If False, only the forward kinematics for the last joint will be returned and the output has one dimension
            less.

    Returns: A Transform3d object containing the forward kinematics for the given joint offsets.
    """
    if isinstance(joint_offsets, torch.Tensor):
        joint_offsets = BatchTransform(joint_offsets)

    if joint_offsets.b is None:
        raise ValueError(
            "Calling forward kinematics for a single joint is not supported."
        )

    nj = joint_offsets.b[-1]

    fk = BatchTransform(matrix=joint_offsets.get_matrix().clone())
    for jnt in range(1, nj):
        fk[..., jnt, :, :] = fk[..., jnt - 1, :, :] @ joint_offsets[..., jnt, :, :]

    if not full:
        return fk.eef
    return fk


def calculate_eef_positions(parameters):
    """
    Calculate the end effector positions of a robot arm given the parameters.
    To calculate the end effector positions, the forward kinematics is used.
    Supports batched input.
    """
    fk = forward_kinematics(dh_to_homogeneous(parameters))
    eef_positions = fk.get_matrix()[..., :2, 3]
    return eef_positions


def update_theta_values(
        parameters: torch.Tensor, new_theta_values: Union[torch.Tensor, np.ndarray]
) -> torch.Tensor:
    """
    Update the theta values of the parameters with the new theta values.
    Supports batched input.
    Supports parameters with theta values and without theta values.
    """
    parameter_dimension = (
        parameters.shape[2] if len(parameters.shape) == 3 else parameters.shape[1]
    )
    updated_parameters = parameters.clone()

    if isinstance(new_theta_values, np.ndarray):
        new_theta_values = torch.tensor(new_theta_values).to(parameters.device)

    if parameter_dimension == 4:
        # Update the theta values
        updated_parameters[:, :, 3] = new_theta_values
    elif parameter_dimension == 3:
        # Add a dimension to include theta values
        new_theta_values = new_theta_values.unsqueeze(-1)
        updated_parameters = torch.cat((parameters, new_theta_values), dim=-1)
    else:
        raise Exception(
            f"Received parameter have unsupported dimension. Expected was 3 or 4 but was {parameter_dimension}"
        )

    return updated_parameters


def calculate_parameter_goal_distances(
        param: torch.Tensor, goal: Union[torch.Tensor, np.ndarray]
) -> torch.Tensor:
    """
    Calculate the Euclidean distance between the end effector position and the goal position.
    Supports batched input.
    """
    eef_positions = calculate_eef_positions(param)
    if isinstance(goal, np.ndarray):
        goal = torch.tensor(goal).to(param.device)
    return calculate_euclidean_distances(eef_positions, goal)


def calculate_euclidean_distances(eef_positions, goal):
    """
    Calculate the Euclidean distance between the eef position and the goal position
    """
    squared_distances = torch.square(eef_positions - goal)
    sum_dim = 1 if len(squared_distances.shape) == 2 else 0
    distances = squared_distances.sum(dim=sum_dim).sqrt()
    return distances


def calculate_angles_from_network_output(action: Tensor, num_joints, device):
    all_angles = None
    for joint_number in range(num_joints):
        index = 2 * joint_number
        sin_x = action[index]
        cos_y = action[index + 1]
        angle = torch.atan2(sin_x, cos_y).unsqueeze(dim=-1)
        if all_angles is None:
            all_angles = angle
        else:
            all_angles = torch.cat([all_angles, angle]).to(device)
    return all_angles
