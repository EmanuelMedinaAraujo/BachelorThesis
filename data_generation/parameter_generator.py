from abc import ABC
from collections.abc import Iterator
from enum import Enum
from typing import Optional, Union

import torch


class ParameterConvention(Enum):
    """A parameter convention for a kinematic chain."""

    MDH = 1  # Modified Denavit-Hartenberg convention after Craig (not supported yet)
    DH = 2  # Denavit-Hartenberg convention


class ParameterGeneratorForPlanarRobot(Iterator, ABC):
    """
    A generator that produces random DH parameters for a planar robot.
    Entire batches of parameters (DH and MDH) can be generated.
    """

    def __init__(
        self,
        batch_size: int = 1,
        device: Optional[str] = None,
        tensor_type: torch.dtype = torch.float32,
        num_joints: int = 2,
        parameter_convention: Union[ParameterConvention, str] = "DH",
        min_len: float = 0.1,
        max_len: float = 20.0,
    ):
        """
        Initializes the problem generator.

        Args:
            batch_size: The batch size to use (number of problems to generate per call).
            device: the device to use for torch tensors
            num_joints: The number of joints of the robot.
            parameter_convention: Either 'DH' or 'MDH'. (As of now, only 'DH' is supported)
            min_len: Limits the minimum value of a
            max_len: Limits the maximum value of a
        """
        super().__init__()
        self.batch_size: int = batch_size
        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device: str = device
        self.tensor_type: torch.dtype = tensor_type
        if not isinstance(parameter_convention, ParameterConvention):
            parameter_convention = ParameterConvention[parameter_convention.upper()]
        if parameter_convention not in ParameterConvention:
            raise ValueError(f"Unknown parameter convention {parameter_convention}")
        if parameter_convention is ParameterConvention.MDH:
            raise NotImplementedError("MDH convention is not supported yet.")
        self.convention = parameter_convention
        self.num_joints: int = num_joints
        self.min_len: float = min_len
        self.max_len: float = max_len

    def __next__(self):
        """Piped through to __call__ without arguments."""
        return self()

    def get_random_parameters(self) -> torch.Tensor:
        """Returns random parameters following the default convention."""
        if self.convention is ParameterConvention.MDH:
            raise NotImplementedError("MDH convention is not supported yet.")
        elif self.convention is ParameterConvention.DH:
            return self.get_random_dh_parameters()
        raise ValueError(f"Unknown parameter convention {self.convention}")

    def get_random_dh_parameters(self) -> torch.Tensor:
        """
        Generates (b x num_joints x 3) random DH parameters for planar robots.
        For planar robots, the DH parameters are (alpha, a, d) with alpha = 0 and d = 0.
        """
        delta = self.max_len - self.min_len
        ttype, device = self.tensor_type, self.device

        # Set alpha to zero
        alpha = torch.zeros((self.batch_size, self.num_joints, 1), device=device).to(
            dtype=ttype
        )

        # Randomly set some a
        a = (
            self.min_len
            + torch.rand((self.batch_size, self.num_joints, 1)).to(
                device=device, dtype=ttype
            )
            * delta
        )

        # Set d to zero as we are currently only interested in planar robots
        d = torch.zeros((self.batch_size, self.num_joints, 1)).to(
            device=device, dtype=ttype
        )

        concatenated_parameters = torch.cat([alpha, a, d], dim=-1)
        return (
            concatenated_parameters[0]
            if self.batch_size == 1
            else concatenated_parameters
        )

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Implements the problem generation."""
        return self.get_random_parameters()
