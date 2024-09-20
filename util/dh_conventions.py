#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 19.06.24

import torch


def dh_to_homogeneous(dh_parameters: torch.Tensor) -> torch.Tensor:
    """
    Converts a set of DH parameters to a homogeneous transformation matrix.

    Follows Siciliano et al., Robotics: Modelling, Planning and Control, 2009, p. 63.
    :param dh_parameters: The DH parameters, ordered as alpha, a, d, theta.
    :return: The homogeneous transformation matrix.
    """
    a, ca, ct, d, sa, st, theta, zeros = extract_values(dh_parameters)

    return torch.stack(
        [
            torch.stack([ct, -st * ca, st * sa, a * ct], dim=-1),
            torch.stack([st, ct * ca, -ct * sa, a * st], dim=-1),
            torch.stack([zeros, sa, ca, d], dim=-1),
            torch.stack([zeros, zeros, zeros, torch.ones_like(theta)], dim=-1),
        ],
        dim=-2,
    )


def mdh_to_homogeneous(mdh_parameters: torch.Tensor) -> torch.Tensor:
    """
    Converts a set of MDH parameters to a homogeneous transformation matrix.

    Follows Craig, Introduction to Robotics, 2005, p. 75.
    :param mdh_parameters: The MDH parameters, ordered as alpha, a, d, theta.
    :return: The homogeneous transformation matrix.
    """
    a, ca, ct, d, sa, st, theta, zeros = extract_values(mdh_parameters)
    return torch.stack(
        [
            torch.stack([ct, -st, zeros, a], dim=-1),
            torch.stack([st * ca, ct * ca, -sa, -d * sa], dim=-1),
            torch.stack([st * sa, ct * sa, ca, d * ca], dim=-1),
            torch.stack([zeros, zeros, zeros, torch.ones_like(theta)], dim=-1),
        ],
        dim=-2,
    )


def extract_values(mdh_parameters):
    alpha = mdh_parameters[..., 0]
    a = mdh_parameters[..., 1]
    d = mdh_parameters[..., 2]
    theta = mdh_parameters[..., 3]
    st = torch.sin(theta)
    ct = torch.cos(theta)
    sa = torch.sin(alpha)
    ca = torch.cos(alpha)
    zeros = torch.zeros_like(theta)
    return a, ca, ct, d, sa, st, theta, zeros


def homogeneous_to_dh(t: torch.Tensor) -> torch.Tensor:
    """
    Converts a homogeneous transformation matrix to a set of DH parameters.

    Attention, this method is expensive due to an internal sanity check.
    Follows Siciliano et al., Robotics: Modelling, Planning and Control, 2009, p. 63.
    :param t: The homogeneous transformation matrix.
    :return: The DH parameters.
    """
    d = t[..., 2, 3]
    theta = torch.atan2(t[..., 1, 0], t[..., 0, 0])
    alpha = torch.atan2(t[..., 2, 1], t[..., 2, 2])
    a = torch.empty_like(d)
    use_cos = torch.isclose(torch.sin(alpha), torch.zeros_like(alpha))
    a[use_cos] = t[use_cos][:, 0, 3] / torch.cos(alpha[use_cos])
    a[~use_cos] = t[~use_cos][:, 1, 3] / torch.sin(theta[~use_cos])

    parameters = torch.stack([alpha, a, d, theta], dim=-1)
    if not torch.allclose(dh_to_homogeneous(parameters), t, atol=1e-3):
        raise ValueError("The given transformation is not DH.")

    return parameters


def homogeneous_to_mdh(t: torch.Tensor) -> torch.Tensor:
    """
    Converts a homogeneous transformation matrix to a set of MDH parameters.

    Attention, this method is expensive due to an internal sanity check.
    Follows Craig, Introduction to Robotics, 2005, p. 75.
    :param t: The homogeneous transformation matrix.
    :return: The MDH parameters.
    """
    a = t[..., 0, 3]
    theta = torch.atan2(-t[..., 0, 1], t[..., 0, 0])
    alpha = torch.atan2(-t[..., 1, 2], t[..., 2, 2])
    d = torch.empty_like(a)
    use_cos = torch.isclose(torch.sin(alpha), torch.zeros_like(alpha))
    d[~use_cos] = -t[~use_cos][:, 1, 3] / torch.sin(alpha[~use_cos])
    d[use_cos] = t[use_cos][:, 2, 3] / torch.cos(alpha[use_cos])

    parameters = torch.stack([alpha, a, d, theta], dim=-1)
    if not torch.allclose(mdh_to_homogeneous(parameters), t, atol=1e-3):
        raise ValueError("The given transformation is not MDH.")

    return parameters
