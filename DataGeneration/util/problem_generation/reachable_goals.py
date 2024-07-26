#!/usr/bin/env python3
# Author: Jonathan Külz
from typing import Optional, Tuple, Union

from omegaconf import DictConfig
import torch

from DataGeneration.util.problem_generation.geometries import Geometry
from utilities.collisions import has_self_collisions
from utilities.goals import get_scalar_goals
from utilities.structures import NetworkInput
from robot.kinematics import forward_kinematics
from transforms.batched import BatchTransform
from transforms.parameterized import DHLikeBase, ParameterConvention, CONVENTION_IMPLEMENTATIONS
from .environment_obstacles import ObstacleGenerator
from .generator_base import ProblemGeneratorBase


class GoalGenerator(ProblemGeneratorBase):
    """Generates random MDH parameters and goals that are reachable with these parameters."""

    T = Tuple[Union[torch.Tensor, DHLikeBase], BatchTransform, Optional[Geometry]]
    p_a_zero = .5
    p_d_zero = .5

    def __init__(self,
                 b: int = 1,
                 goals_per_robot: int = 1,
                 num_joints: int = 6,
                 parameter_convention: Union[ParameterConvention, str] = 'MDH',
                 min_len: float = 0.,
                 max_len: float = 1.,
                 alpha_values: Tuple[float, ...] = (0, -torch.pi / 2, torch.pi / 2),
                 obstacle_generator: Optional[ObstacleGenerator] = None,
                 link_radius: float = 0.1,
                 oversampling: float = 2.5,
                 device: str = None,
                 tensor_type: torch.dtype = torch.float32,
                 graph_ik_constraints: bool = False,
                 ):
        """
        Initializes the problem generator.

        :param b: The batch size to use (number of problems to generate per call).
        :param goals_per_robot: The number of goals to generate per robot. (Think of it as 2nd batch dimension)
        :param num_joints: The number of joints of the robot.
        :param parameter_convention: The parameter convention to use. Either 'MDH' or 'DH'.
        :param min_len: Limits the absolute value of the a, d parameters.
        :param max_len: Limits the absolute value of the a, d parameters.
        :param alpha_values: A set of possible values for the alpha DH parameter to take on
        :param obstacle_generator: An obstacle generator to use for generating obstacles.
        :param link_radius: The radius of the robot links, relevant to create collision-free goals.
        :param oversampling: Instead of b, creates oversampling * b problems and selects only b. Oversampling larger
            than 1 is recommended to avoid multiple sampling processes in case of obstacle avoidance.
        :param graph_ik_constraints: If true, uses constraints from Graphical IK paper.
        """
        super().__init__(b, device, tensor_type)
        if not alpha_values[0] == 0:
            print("Warning: The first alpha value should be 0.")
        self.alphas = alpha_values
        if not isinstance(parameter_convention, ParameterConvention):
            parameter_convention = ParameterConvention[parameter_convention.upper()]
        self.convention = parameter_convention
        self.parameter_class = CONVENTION_IMPLEMENTATIONS[self.convention]
        self.default_offset: DHLikeBase = self.parameter_class(torch.zeros(
            (int(b * oversampling * goals_per_robot), 1, 4), device=device))
        self.link_radius: float = link_radius
        self.obstacle_generator: Optional[ObstacleGenerator] = obstacle_generator
        self.oversampling: float = oversampling
        self.gpr: int = goals_per_robot
        self.nj: int = num_joints
        self.min_len: float = min_len
        self.max_len: float = max_len
        self.graph_ik_constraints: bool = graph_ik_constraints

    def get_random_parameters(self, oversample: bool = False) -> torch.Tensor:
        """Returns random parameters following the default convention."""
        if self.convention is ParameterConvention.MDH:
            return self.get_random_mdh_parameters(oversample=oversample)
        elif self.convention is ParameterConvention.DH:
            return self.get_random_dh_parameters(oversample=oversample)
        raise ValueError(f"Unknown parameter convention {self.convention}")

    def get_random_dh_parameters(self, oversample: bool = False) -> torch.Tensor:
        """
        Generates (b x nj x 4) random DH parameters for alpha, a, d, theta.

        Differences to Limoyo et al, "Generative Inverse Kinematics":
        - They enforce alpha_0 != 0
        - They enforce a_0 == 0
        - They enforce a_{n-1} = 0
        - They enforce alpha_{n-1} = 0
        """
        #oversampling egal
        #b batchsize
        b = int(self.b * self.oversampling) if oversample else self.b
        delta = self.max_len - self.min_len
        ttype, device = self.tensor_type, self.device

        alpha = (torch.ones((b, self.nj, 1), device=device) * torch.tensor(self.alphas, device=device)).to(dtype=ttype)
        alpha_indices = torch.randint(0, len(self.alphas), (b, self.nj, 1)).to(device=device)
        alpha_indices[:, 0, :] = torch.randint(1, 3, (b, 1)).to(device=device, dtype=ttype)  # NEW: alpha_0 != 0
        alpha = torch.gather(alpha, dim=-1, index=alpha_indices)

        # Never allow more than two consecutive parallel joints
        consecutive_parallel = torch.logical_and(
            torch.logical_and(alpha[:, :self.nj - 2, :] == 0, alpha[:, 1:self.nj - 1, :] == 0),
            alpha[:, 2:self.nj, :] == 0)
        alpha[:, :-2, :][consecutive_parallel] = torch.tensor([torch.pi / 2]).to(device=device, dtype=ttype)

        # Randomly set some a to zero, but never for parallel joints
        a = self.min_len + torch.rand((b, self.nj, 1)).to(device=device, dtype=ttype) * delta
        a[torch.logical_and(alpha != 0, torch.rand((b, self.nj, 1)).to(device=device, dtype=ttype) < self.p_a_zero)] = 0

        # Randomly set some d to zero. Enforce this for parallel joints by convention.
        d = self.min_len + torch.rand((b, self.nj, 1)).to(device=device, dtype=ttype) * delta
        d[torch.logical_or(alpha == 0, torch.rand((b, self.nj, 1)).to(device=device, dtype=ttype) < self.p_d_zero)] = 0

        # Ignorieren
        if self.graph_ik_constraints:
            a[:, 0, :] = 0  # a_0 = 0
            a[:, -1, :] = 0  # a_{n-1} = 0
            alpha[:, -1, :] = 0  # alpha_{n-1} = 0
            a[alpha != 0] = 0  # set a to zero if alpha is not zero

        return torch.cat([alpha, a, d], dim=-1)

    def get_random_mdh_parameters(self, oversample: bool = False) -> torch.Tensor:
        """
        Generates (b x nj x 3) random MDH parameters for alpha, a, d.
        """
        b = self.b * oversample if oversample else self.b
        delta = self.max_len - self.min_len
        ttype, device = self.tensor_type, self.device
        alpha = (torch.ones((b, self.nj, 1), device=device) * torch.tensor(self.alphas, device=device)).to(dtype=ttype)
        alpha_indices = torch.randint(0, 3, (b, self.nj, 1)).to(device=device)
        alpha = torch.gather(alpha, dim=-1, index=alpha_indices)

        # Never allow more than two consecutive parallel joints
        consecutive_parallel = torch.logical_and(
            torch.logical_and(alpha[:, :self.nj - 2, :] == 0, alpha[:, 1:self.nj - 1, :] == 0),
            alpha[:, 2:self.nj, :] == 0)
        alpha[:, 2:, :][consecutive_parallel] = torch.tensor([torch.pi / 2]).to(device=device, dtype=ttype)

        # Randomly set some a to zero, but never for parallel joints
        a = self.min_len + torch.rand((b, self.nj, 1)).to(device=device, dtype=ttype) * delta
        a[torch.logical_and(alpha != 0, torch.rand((b, self.nj, 1)).to(device=device, dtype=ttype) < .5)] = 0

        # Randomly set some d to zero. Enforce this for parallel joints by convention.
        d = self.min_len + torch.rand((b, self.nj, 1)).to(device=device, dtype=ttype) * delta
        d[torch.logical_or(alpha == 0, torch.rand((b, self.nj, 1)).to(device=device, dtype=ttype) < .5)] = 0

        if self.graph_ik_constraints:
            raise NotImplementedError("Graph IK constraints not implemented for MDH parameters.")

        return torch.cat([alpha, a, d], dim=-1)

    def __call__(self,
                 avoid_collisions: bool = True,
                 avoid_underground: bool = True,
                 **kwargs) -> T:
        """
        Generates random cartesian goals.

        :param return_type: The type of parameters to be returned. Either 'broadcasted', which returns a transform
            with b * gpr transforms, or 'robots', which returns the unique b (alpha, a, d) parameters used.
        :param avoid_collisions: If True, avoids self-collisions in the generated goals.
        :param avoid_underground: If True, avoids goals with negative z-coordinates.
        :param kwargs: Unused.
        :return: A tuple of (DHLikeTransform, goals) where the transform is of size according to return_type and the
            goals are a Transform3d of length (b * gpr).
        """
        ngoals = int(self.oversampling * self.gpr) if (avoid_collisions or avoid_underground) else self.gpr
        parameters = self.get_random_parameters(oversample=True)
        broadcasted = torch.repeat_interleave(parameters, ngoals, dim=0)
        theta = torch.unsqueeze(torch.rand(broadcasted.shape[0], broadcasted.shape[1],
                                           device=self.device) * 2 * torch.pi, dim=-1) - torch.pi

        transform = self.parameter_class(torch.concat([broadcasted, theta], dim=-1))
        fk = forward_kinematics(joint_offsets=transform, full=True)

        indices = torch.arange(transform.b[0], device=self.device)
        if avoid_underground:
            avoid = fk.eef.get_matrix()[..., 2, 3] < 0
            indices[avoid] = -1

        if avoid_collisions:
            self_collisions = has_self_collisions(transform, fk, self.link_radius).view(self.b, -1)
            indices[torch.nonzero(self_collisions.flatten()).squeeze()] = -1

        # indices contains possible goal/robot samples. now, make sure there's gpr goals for every one of b robots
        candidates = indices.view(-1, ngoals) >= 0
        candidates[candidates.cumsum(dim=-1) > self.gpr] = False  # Choose at most gpr per robot
        num_goals = candidates.cumsum(dim=-1).max(dim=-1)[0]
        robot_selection = torch.argsort(num_goals, descending=True)[:self.b]
        overwrite = num_goals[robot_selection] < self.gpr
        robot_selection[overwrite] = robot_selection[~overwrite][:overwrite.sum()]
        goal_selection = candidates[robot_selection]

        parameters = transform.parameters.view(parameters.shape[0], ngoals, -1, self.parameter_class.get_num_parameters())[robot_selection]
        parameters = parameters[goal_selection].view(self.b, self.gpr, -1, self.parameter_class.get_num_parameters())

        full_dh = self.parameter_class(parameters)
        fk = forward_kinematics(joint_offsets=full_dh, full=True)
        if self.obstacle_generator is None:
            obstacles = None
        elif avoid_collisions:
            obstacles = self.obstacle_generator(avoid_goals=fk.eef, avoid_robot=(full_dh, fk, self.link_radius))
        else:
            obstacles = self.obstacle_generator()

        return full_dh, fk.eef, obstacles


@torch.no_grad()
def get_training_data(goal_generator: GoalGenerator,
                      step: int,
                      cfg: DictConfig, **kwargs) -> Tuple[NetworkInput, DHLikeBase]:
    """
    Utility function to generate training data using a goal generator.

    Args:
        goal_generator: The goal generator to use.
        step: Current step
        cfg: Configuration for the training.
    """
    kwargs.setdefault('avoid_collisions', cfg.collision.avoid_collisions_in_data)
    parameters, poses, env = goal_generator(**kwargs)
    data = NetworkInput(goals=get_scalar_goals(poses, cfg.b, cfg.goals.goal_mode.value), env=env, step=step)
    return data, parameters


def main():
    MyGenerator = GoalGenerator(num_joints=2)
    dh, fk,_= MyGenerator()
    print(dh)
    print(fk)

main()

