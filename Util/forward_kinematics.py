#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 19.06.24
from typing import Union
import torch
from Util.batched import BatchTransform
from Util.dh_conventions import dh_to_homogeneous


def forward_kinematics(joint_offsets: Union[torch.Tensor, BatchTransform], full: bool = False) -> BatchTransform:
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
        raise ValueError("Calling forward kinematics for a single joint is not supported.")

    nj = joint_offsets.b[-1]

    fk = BatchTransform(matrix=joint_offsets.get_matrix().clone())
    for jnt in range(1, nj):
        fk[..., jnt, :, :] = fk[..., jnt - 1, :, :] @ joint_offsets[..., jnt, :, :]

    if not full:
        return fk.eef
    return fk


def calculate_eef_positions(dh_param):
    fk = forward_kinematics(dh_to_homogeneous(dh_param))
    eef_positions = fk.get_matrix()[..., :2, 3]
    return eef_positions


def update_theta_values(parameters, new_theta_values):
    parameter_dimension = parameters.shape[2]
    updated_parameters = parameters.clone()

    if parameter_dimension == 4:
        updated_parameters[:, :, 3] = new_theta_values
    elif parameter_dimension == 3:
        # Add a dimension to include theta values
        new_theta_values = new_theta_values.unsqueeze(-1)
        updated_parameters = torch.cat((parameters, new_theta_values), dim=-1)
    else:
        raise Exception(
            f"Received parameter have unsupported dimension. Expected was 3 or 4 but was {parameter_dimension}")

    return updated_parameters
