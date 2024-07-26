#!/usr/bin/env python3
# Author: Jonathan Külz
from abc import abstractmethod, ABC
from collections.abc import Iterator
import torch
from typing import Optional


class ProblemGeneratorBase(Iterator, ABC):
    """Base class for problem generators."""

    T: type = None  # The data type of the problem

    def __init__(self,
                 b: int = 1,
                 device: Optional[str] = None,
                 tensor_type: torch.dtype = torch.float32,
                 ):
        """
        Initializes the problem generator.

        :param b: The batch size to use (number of problems to generate per call).
        """
        super().__init__()
        self.b: int = b
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device: str = device
        self.tensor_type: torch.dtype = tensor_type

    def __next__(self):
        """Piped through to __call__ without arguments."""
        return self()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> T:
        """Implements the problem generation."""

