#!/usr/bin/env python3
# Author: Jonathan Külz
from typing import Optional, Tuple

import torch

from DataGeneration.util.problem_generation.geometries import Sphere
from problem_generation import ProblemGeneratorBase
from transforms.batched import BatchTransform
from transforms.parameterized import DHLikeBase
from utilities.collisions import has_env_collisions
from utilities.structures import EnvironmentInfo


class ObstacleGenerator(ProblemGeneratorBase):
    """Creates obstacles in the environment."""

    T = EnvironmentInfo

    def __init__(self,
                 b: int = 1,
                 max_num_obstacles: int = 4,
                 fraction_tasks_with_obstacles: float = 1,
                 workspace_radius: float = 1.0,
                 sphere_radius: float = 0.1,
                 goal_safety_margin: float = 0.1,
                 device: str = None,
                 tensor_type: torch.dtype = torch.float32):
        """
        Initializes the obstacle generator.

        :param b: The batch size to use (number of problems to generate per call).
        """
        super().__init__(b, device, tensor_type)
        self.n: int = max_num_obstacles
        self.fraction_tasks_with_obstacles: float = fraction_tasks_with_obstacles
        self.safety_margin: float = goal_safety_margin
        self.sphere_radius: float = sphere_radius
        self.workspace_radius: float = workspace_radius

    def __call__(self,
                 avoid_goals: Optional[BatchTransform] = None,
                 avoid_robot: Optional[Tuple[DHLikeBase, BatchTransform, float]] = None,
                 **kwargs):
        """Implements the obstacle generation. Creates n obstacles, the same ones for all problems in the batch."""
        if self.n == 0:
            return EnvironmentInfo(Sphere(torch.zeros((1, 3), device=self.device, dtype=self.tensor_type),
                                          torch.zeros((1,), device=self.device, dtype=self.tensor_type)),
                                   torch.zeros(self.b, self.n, dtype=torch.bool, device=self.device),
                                   b=self.b)
        directions = torch.randn((self.n, 3), device=self.device, dtype=self.tensor_type)
        directions /= directions.norm(dim=-1, keepdim=True)
        p = directions * self.workspace_radius * torch.rand((self.n, 1), device=self.device, dtype=self.tensor_type)
        p[..., 2] = torch.abs(p[..., 2])
        obstacles = Sphere(p, torch.ones(self.n, device=self.device) * self.sphere_radius)

        if avoid_goals is not None:
            avoid = Sphere(avoid_goals, torch.ones(len(avoid_goals), dtype=avoid_goals.dtype, device=avoid_goals.device) * self.safety_margin)
            indices = torch.cartesian_prod(torch.arange(self.n, device=self.device),
                                           torch.arange(len(avoid_goals), device=self.device))
            distances = obstacles[indices[:, 0]].signed_distance(avoid[indices[:, 1]]).view(self.n, -1)
            mask = (distances > 0).transpose(0, 1)
        else:
            mask = torch.ones((self.b, self.n), dtype=torch.bool, device=self.device)

        obstacles = Sphere(obstacles.center, obstacles.radius)

        random_mask = torch.rand((self.b, 1), device=self.device) <= self.fraction_tasks_with_obstacles
        env = EnvironmentInfo(obstacles, torch.logical_and(mask, random_mask), self.b)
        if avoid_robot is not None:
            collisions = has_env_collisions(*avoid_robot, env=env)
            env.goal_mask[collisions[:, 0], collisions[:, 1]] = False
        return env
