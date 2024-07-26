#!/usr/bin/env python3
# Author: Jonathan Külz
from typing import Optional

import torch

from .generator_base import ProblemGeneratorBase


class RandomCartesianGoals(ProblemGeneratorBase):
    """Generates random cartesian goals for a given robot."""

    T = torch.Tensor

    def __init__(self,
                 scale: float = 1.0,
                 b: int = 1,
                 device: Optional[str] = None,
                 tensor_type: torch.dtype = torch.float32,
                 ):
        """
        Initializes the problem generator.

        :param b: The batch size to use (number of problems to generate per call).
        :param scale: Limits the size of the workspace.
        """
        super().__init__(b, device, tensor_type)
        self.scale: float = scale

    def __call__(self, *args, **kwargs) -> T:
        """
        Generates random cartesian goals.

        :param args: Unused.
        :param kwargs: Unused.
        :return: A tensor of shape b x 3 with the cartesian goals.
        """
        return torch.rand((self.b, 3)).to(device=self.device, dtype=self.tensor_type) * self.scale
