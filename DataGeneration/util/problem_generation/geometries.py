#!/usr/bin/env python3
# Author: Jonathan Külz
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch

from timor.utilities.visualization import MeshcatVisualizer
from timor import Geometry as tg
from timor import Obstacle, Transformation
from transforms.batched import BatchTransform, Rotate
from transforms.parameterized import DHTransform
from utilities.rotations import continuous_rotmat_representation


def ensure_2d_tensor(t: torch.Tensor):
    if len(t.shape) <= 1:
        t = t.unsqueeze(0)
    return t


class Geometry(ABC):

    def __init__(self, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None):
        """An abstract base class to represent geometries that support differentiable distance functions."""
        self.dtype: torch.dtype = dtype
        self.device: Union[str, torch.device] = device

    @property
    @abstractmethod
    def b(self) -> Union[Tuple[int], Tuple[int, int]]:
        """Returns the batch size of the geometry. Can be one integer or two integers."""
        pass

    @property
    @abstractmethod
    def flat(self) -> torch.Tensor:
        """A flattened version of the geometry, used for learning."""

    @abstractmethod
    def stack(self, other: Geometry) -> Geometry:
        """Stacks all geometries from this class with geometries from other."""
        pass

    @abstractmethod
    def signed_distance(self, other: Geometry) -> torch.Tensor:
        """Returns the signed distance from self to other along the batch dimension(s)."""
        pass

    def visualize(self, viz: Optional[MeshcatVisualizer] = None) -> MeshcatVisualizer:
        """Visualize the obstacle in meshcat"""
        pass

    @abstractmethod
    def __getitem__(self, item: torch.Tensor) -> Geometry:
        """Returns a geometry with the selected items."""
        pass


class Capsule(Geometry):
    """This class holds geometry information about capsules."""

    eps: float = 1e-6  # Minimum extent of the capsule in height and radius.
    eps_numeric: float = 1e-9  # Numerical stability constant

    def __init__(self,
                 center: BatchTransform,
                 height: torch.Tensor,
                 radius: torch.Tensor,
                 ):
        """
        Initializes the capsule, using it's center, height, and radius.

        Following the convention of this wikipedia article (height is without the radius and along z-axis):
        https://en.wikipedia.org/wiki/Capsule_(geometry)

        :param center: The center of the capsule. Also defines its orientation.
        :param height: The height of the capsule along the z-axis.
        :param radius: The radius of the capsule.
        :param dtype: The data type to use.
        :param device: The device to use.
        """
        device = center.device
        dtype = center.dtype
        super().__init__(dtype=dtype, device=device)
        self.center: BatchTransform = center
        flip = height < 0
        R_FLIP = Rotate.axis_angle(angle=torch.ones([flip.sum()], device=self.device) * torch.pi, axis='X', degrees=False)
        center.get_matrix()[flip] = center[flip].compose(R_FLIP).get_matrix()
        height = torch.abs(height)
        if not height.min() > self.eps and radius.min() > self.eps:
            raise ValueError(f"The height (norm) and radius of capsules must be greater than {self.eps}."
                             f"Use a sphere instead if necessary.")
        if not center.b == tuple(height.shape) == tuple(radius.shape):
            raise ValueError("The batches of center, height, and radius must match.")
        self.height: torch.Tensor = height.to(device, dtype)
        self.radius: torch.Tensor = radius.to(device, dtype)

    @classmethod
    def from_dh_parameters(cls, parameters: torch.Tensor) -> Capsule:
        """Creates a capsule from DH parameters."""
        return cls(center=DHTransform.from_parameters(parameters), height=parameters[..., 2], radius=parameters[..., 3])

    @property
    def axis(self) -> torch.Tensor:
        """Returns the axis of the capsule."""
        return self.center.get_matrix()[..., :3, 2]

    @property
    def b(self) -> Union[Tuple[int], Tuple[int, int]]:
        """Returns the batch size of the capsule. Can be one integer or two integers."""
        return tuple(self.height.shape)

    @property
    def base(self) -> torch.Tensor:
        """Returns the base point of the capsule"""
        t = torch.vstack((torch.zeros_like(self.height), torch.zeros_like(self.height), -self.height / 2)).transpose(0,
                                                                                                                     1)
        return self.center.translate(t).translation

    @property
    def flat(self) -> torch.Tensor:
        """A flattened version of the capsule, used for learning."""
        return torch.cat((
            self.center.translation,
            continuous_rotmat_representation(self.center.rotation),
            self.height,
            self.radius
        ))

    @property
    def tip(self) -> torch.Tensor:
        """Returns the tip point of the capsule"""
        t = torch.vstack((torch.zeros_like(self.height), torch.zeros_like(self.height), self.height / 2)).transpose(0,
                                                                                                                    1)
        return self.center.translate(t).translation

    def stack(self, *others: Capsule) -> Capsule:
        """Stacks all capsules from this class with capsules from other."""
        heights = [self.height] + [other.height for other in others]
        radii = [self.radius] + [other.radius for other in others]
        center = self.center.stack(*[other.center for other in others])
        height = torch.cat(heights, dim=0)
        radius = torch.cat(radii, dim=0)
        return Capsule(center=center, height=height, radius=radius)

    def signed_distance(self, other: Geometry) -> torch.Tensor:
        """Signed distance between self and the other obstacle(s)."""
        if isinstance(other, Capsule):
            return self.signed_distance_capsule(other)
        elif isinstance(other, Sphere):
            return self.signed_distance_sphere(other)
        raise NotImplementedError(f"Signed distance between {type(self)} and {type(other)} is not implemented.")

    def signed_distance_capsule(self, other: Capsule) -> torch.Tensor:
        """
        Returns the distance between this capsule and another capsule along the batch dimension.

        :reference: https://arrowinmyknee.com/2021/03/15/some-math-about-capsule-collision/
        :param other: The other capsule.
        :return: The distance.
        """
        this_tip = self.tip
        this_base = self.base
        other_tip = other.tip
        other_base = other.base

        _A = self.axis
        _B = other.axis

        ray = torch.cross(_A, _B, dim=1)
        denominator = torch.square(torch.linalg.norm(ray, dim=-1, ord=2))

        distance = torch.zeros(denominator.shape[0]).to(denominator)

        d0 = torch.sum(_A * (other_base - this_base), axis=1)
        d1 = torch.sum(_A * (other_tip - this_base), axis=1)

        condition_1 = denominator < self.eps_numeric
        condition_2 = condition_1 & (d0 <= self.eps_numeric) & (self.eps_numeric >= d1)
        condition_3 = condition_1 & (d0 >= self.height) & (self.height <= d1)

        case_0 = torch.linalg.norm(_A * d0.unsqueeze(1) + this_base - other_base, dim=-1, ord=2)
        distance = torch.where(condition_1, case_0, distance)

        case_1 = torch.where(torch.abs(d0) < torch.abs(d1),
                             torch.linalg.norm(this_base - other_base, dim=-1, ord=2),
                             torch.linalg.norm(this_base - other_tip, dim=-1, ord=2))
        distance = torch.where(condition_2, case_1, distance)

        case_2 = torch.where(torch.abs(d0) < torch.abs(d1),
                             torch.linalg.norm(this_tip - other_base, dim=-1, ord=2),
                             torch.linalg.norm(this_tip - other_tip, dim=-1, ord=2), )
        distance = torch.where(condition_3, case_2, distance)

        t = other_base - this_base

        detA_vector = torch.stack((t, _B, ray), dim=1)
        detB_vector = torch.stack((t, _A, ray), dim=1)

        detA = torch.linalg.det(detA_vector)
        detB = torch.linalg.det(detB_vector)

        t0 = detA / torch.max(torch.ones_like(denominator) * self.eps_numeric, denominator)  # Why is this necessary?
        t1 = detB / torch.max(torch.ones_like(denominator) * self.eps_numeric, denominator)  # Why is this necessary?

        pA = this_base + _A * t0.unsqueeze(1)
        pB = other_base + _B * t1.unsqueeze(1)

        pA = torch.where(t0.unsqueeze(1) < 0.0, this_base, pA)
        pA = torch.where(t0.unsqueeze(1) > self.height.unsqueeze(1), this_tip, pA)

        pB = torch.where(t1.unsqueeze(1) < 0.0, other_base, pB)
        pB = torch.where(t1.unsqueeze(1) > other.height.unsqueeze(1), other_tip, pB)

        dot_0 = torch.sum(_B * (pA - other_base), axis=-1)
        dot_0 = torch.where(dot_0 < 0.0, 0.0, dot_0)
        dot_0 = torch.where(dot_0 > other.height, other.height, dot_0)
        pB = torch.where((dot_0.unsqueeze(1) < 0.0) | (dot_0.unsqueeze(1) > self.height.unsqueeze(1)),
                         other_base + _B * dot_0.unsqueeze(1), pB)

        dot_1 = torch.sum(_A * (pB - this_base), axis=1)
        dot_1 = torch.where(dot_1 < 0.0, 0.0, dot_1)
        dot_1 = torch.where(dot_1 > self.height, self.height, dot_1)
        pA = torch.where((dot_1.unsqueeze(1) < 0.0) | (dot_1.unsqueeze(1) > other.height.unsqueeze(1)),
                         this_base + _A * dot_1.unsqueeze(1), pA)

        distance = torch.where(denominator >= self.eps_numeric, torch.linalg.norm(pA - pB, dim=1, ord=2), distance)

        return distance - self.radius - other.radius

    def signed_distance_sphere(self, other: Sphere) -> torch.Tensor:
        """Returns the distance between this capsule and another sphere along the batch dimension."""
        len_squared = self.height ** 2
        t = torch.clamp(torch.einsum('ab,ab->a', (other.center - self.base), (self.tip - self.base)) / len_squared, 0.0, 1.0)
        proj = self.base + t.unsqueeze(1) * (self.tip - self.base)
        return torch.linalg.norm(proj - other.center, dim=-1, ord=2) - self.radius - other.radius

    def visualize(self, viz: Optional[MeshcatVisualizer] = None) -> MeshcatVisualizer:
        """Visualize the capsule in meshcat"""
        if viz is None:
            viz = MeshcatVisualizer()
            viz.initViewer()
        for idx in torch.arange(*self.b):
            p = self.center[idx].get_matrix().detach().cpu().numpy()
            r = self.radius[idx].item()
            h = self.height[idx].item()
            delta = Transformation.from_translation([0, 0, h / 2])
            cg = tg.ComposedGeometry([
                tg.Cylinder({'r': r, 'z': h}, pose=Transformation(p)),
                tg.Sphere({'r': r}, pose=Transformation(p) @ delta),
                tg.Sphere({'r': r}, pose=Transformation(p) @ delta.inv),
            ])
            o = Obstacle.Obstacle(f'Capsule_{idx.item()}', cg)
            o.visualize(viz)
        return viz

    def __getitem__(self, item):
        """Returns a capsule with the selected batch items."""
        return Capsule(center=self.center[item], height=self.height[item], radius=self.radius[item])


class Sphere(Geometry):
    """The simplest kind of geometry out there."""

    def __init__(self,
                 center: Union[BatchTransform, torch.Tensor],
                 radius: torch.Tensor,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[Union[str, torch.device]] = None
                 ):
        """Sphere constructor."""
        if device is None:
            device = center.device
        if dtype is None:
            dtype = center.dtype
        super().__init__(dtype=dtype, device=device)
        if isinstance(center, BatchTransform):
            center = center.translation
        if not center.shape[-1] == 3:
            raise ValueError("The center of a sphere must have three dimensions.")
        self.center: torch.Tensor = center
        self.radius: torch.Tensor = radius.to(device, dtype)

    @property
    def b(self) -> Union[Tuple[int], Tuple[int, int]]:
        """Returns the batch size of the sphere. Can be one integer or two integers."""
        return self.radius.shape[:self.num_batch_levels]

    @property
    def num_batch_levels(self) -> int:
        """Returns the number of batch levels."""
        return len(self.center.shape) - 1

    @property
    def flat(self) -> torch.Tensor:
        """A flattened version of the capsule, used for learning."""
        return torch.cat((
            self.center,
            self.radius.unsqueeze(-1)
        ), dim=-1)

    def stack(self, *others: Sphere) -> Sphere:
        """Stacks all spheres from this class with spheres from other."""
        center = torch.cat([self.center] + [other.center for other in others], dim=0)
        radius = torch.cat([self.radius] + [other.radius for other in others], dim=0)
        return Sphere(center=center, radius=radius)

    def signed_distance(self, other: Geometry) -> torch.Tensor:
        """Signed distance between self and the other obstacle(s)."""
        if isinstance(other, Sphere):
            return self.signed_distance_sphere(other)
        elif isinstance(other, Capsule):
            return other.signed_distance(self)
        raise NotImplementedError(f"Signed distance between {type(self)} and {type(other)} is not implemented.")

    def signed_distance_sphere(self, other: Sphere) -> torch.Tensor:
        """Returns the distance between this sphere and another sphere along the batch dimension."""
        return torch.linalg.norm(self.center - other.center, dim=-1, ord=2) - self.radius - other.radius

    def visualize(self, viz: Optional[MeshcatVisualizer] = None) -> MeshcatVisualizer:
        """Visualize the sphere in meshcat"""
        if viz is None:
            viz = MeshcatVisualizer()
            viz.initViewer()
        for idx in torch.arange(self.radius.numel()):
            p = self.center.view(-1, 3)[idx].detach().cpu().numpy()
            r = self.radius.view(-1, 1)[idx].item()
            o = Obstacle.Obstacle(f'Sphere_{idx.item()}', tg.Sphere({'r': r},
                                                                    pose=Transformation.from_translation(p)))
            o.visualize(viz)
        return viz

    def __getitem__(self, item):
        """Returns a sphere with the selected batch items."""
        return Sphere(center=self.center[item], radius=self.radius[item])
