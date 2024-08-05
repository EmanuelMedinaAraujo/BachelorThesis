#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 19.06.24
from typing import Union

import torch

from util.batched import BatchTransform


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


#Tests
# dh_param = torch.tensor( [[[ 0.0000, 13.2026,  0.0000, -0.3302],
#         [ 0.0000,  9.6462,  0.0000, -0.0560]],
#
#         [[ 0.0000, 13.,  0.0000, -0.0],
#         [ 0.0000,  9.,  0.0000, -0.0]]
#         ])
# forward_kinematics_matrix = dh_conventions.dh_to_homogeneous(dh_param)
#
# expected_result = [21.4253, -7.9135]
# fk_calc = forward_kinematics(forward_kinematics_matrix).get_matrix()
# end_effector = fk_calc[..., :2, 3]
# x, y = end_effector[0]
# print(f"x: {x}, y: {y}")
# x1, y1 = end_effector[1]
# print(f"x: {x1}, y: {y1}")
# print(fk_calc)


