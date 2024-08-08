from abc import ABC
from collections.abc import Iterator
import torch
from enum import Enum
from typing import Optional, Tuple, Union


# DH Parameter of 2 Link Planar Robot with extended arm (alpha, a, d, theta)
# DH_EXAMPLE = torch.tensor([
#     [0, 15, 0, 0],
#     [0, 10, 0, 0]
# ])

class ParameterConvention(Enum):
    """A parameter convention for a kinematic chain."""
    MDH = 1  # Modified Denavit-Hartenberg convention after Craig
    DH = 2  # Denavit-Hartenberg convention


class ParameterGenerator(Iterator, ABC):
    T = torch.Tensor

    def __init__(self,
                 batch_size: int = 1,
                 device: Optional[str] = None,
                 tensor_type: torch.dtype = torch.float32,
                 num_joints: int = 2,
                 parameter_convention: Union[ParameterConvention, str] = 'DH',
                 min_len: float = 0.1,
                 max_len: float = 20.,
                 ):
        """
        Initializes the problem generator.

        :param batch_size: The batch size to use (number of problems to generate per call).
        :param device: the device to use for torch tensors
        :param num_joints: The number of joints of the robot.
        :param parameter_convention: Either 'DH' or 'MDH'.
        :param min_len: Limits the absolute value of the a, d parameters.
        :param max_len: Limits the absolute value of the a, d parameters.
        """
        super().__init__()
        self.batch_size: int = batch_size
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device: str = device
        self.tensor_type: torch.dtype = tensor_type
        if not isinstance(parameter_convention, ParameterConvention):
            parameter_convention = ParameterConvention[parameter_convention.upper()]
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
            return self.get_random_mdh_parameters()
        elif self.convention is ParameterConvention.DH:
            return self.get_random_dh_parameters()
        raise ValueError(f"Unknown parameter convention {self.convention}")

    def get_random_dh_parameters(self) -> torch.Tensor:
        """
        Generates (b x num_joints x 3) random DH parameters for alpha, a, d.
        """
        delta = self.max_len - self.min_len
        ttype, device = self.tensor_type, self.device

        # Set alpha to zero
        alpha = torch.zeros((self.batch_size, self.num_joints, 1), device=device).to(dtype=ttype)

        # Randomly set some a
        a = self.min_len + torch.rand((self.batch_size, self.num_joints, 1)).to(device=device, dtype=ttype) * delta

        # Set d to zero as we are currently only interested in planar robots
        d = torch.zeros((self.batch_size, self.num_joints, 1)).to(device=device, dtype=ttype)

        return torch.cat([alpha, a, d], dim=-1)

    def get_random_mdh_parameters(self) -> torch.Tensor:
        """
        Generates (b x num_joints x 3) random MDH parameters for alpha, a, d.
        """
        delta = self.max_len - self.min_len
        ttype, device = self.tensor_type, self.device

        # Set alpha to zero
        alpha = torch.zeros((self.batch_size, self.num_joints, 1), device=device).to(dtype=ttype)

        # Randomly set some 'a' to zero
        a = self.min_len + torch.rand((self.batch_size, self.num_joints, 1)).to(device=device, dtype=ttype) * delta

        # Set d to zero as we are currently only interested in planar robots
        d = torch.zeros((self.batch_size, self.num_joints, 1)).to(device=device, dtype=ttype)

        return torch.cat([alpha, a, d], dim=-1)

    def __call__(self, *args, **kwargs) -> T:
        """Implements the problem generation."""
        return self.get_random_parameters()
