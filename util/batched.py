"""
This file contains modified code from PyTorch3D, which is licensed under the BSD License.

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Copyright (c) 19.06.2024, Jonathan Külz. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Meta nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

from math import prod
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch


class BatchTransform:
    """
    A class similar to pytorch3d Transform3d objects that supports N batch levels.
    """

    def __init__(self, matrix: torch.Tensor) -> None:
        """A BatchTransform needs to be initialized with a matrix that determines its batch size."""
        if matrix.shape[-2] != 4 or matrix.shape[-1] != 4:
            raise ValueError(
                '"matrix" has to be a tensor of shape (..., 4, 4) or (4, 4).'
            )
        self.dtype = matrix.dtype
        self.device = matrix.device
        self._matrix = matrix

    @property
    def b(self) -> Union[Tuple[int, ...], None]:
        if self._matrix.dim() == 2:
            return None
        return tuple(map(int, self._matrix.shape[:-2]))

    @property
    def eef(self) -> BatchTransform:
        """
        Returns a slice of this transform, containing only the last element for the last batch dimension.

        We use BatchTransforms to describe robot forward kinematics, and the batch dimensions are used to represent
        (robots, joints). If that's the case, this method returns the FK for the last joint only, but all robots.
        """
        return BatchTransform(self.get_matrix()[..., -1, :, :])

    @property
    def flat_tensor(self) -> torch.Tensor:
        """Returns the matrix representation of the transform with the batch dimensions flattened."""
        return self._matrix.view(-1, 4, 4)

    @property
    def translation(self) -> torch.Tensor:
        """Returns the translation part of the transform."""
        return self._matrix[..., :3, 3]

    @property
    def rotation(self) -> torch.Tensor:
        """Returns the rotation part of the transform."""
        return self._matrix[..., :3, :3]

    def __eq__(self, other):
        """Compares the matrix representation of the transforms."""
        if not isinstance(other, BatchTransform):
            return False
        return self.get_matrix().equal(other.get_matrix())

    def __getitem__(
        self, index: Union[int, List[int], slice, torch.BoolTensor, torch.LongTensor]
    ) -> BatchTransform:
        """
        Args:
            index: Specifying the index of the transform to retrieve.
                Can be an int, slice, list of ints, boolean, long tensor.
                Supports negative indices.

        Returns:
            BatchedTransform object with selected transforms. The tensors are not cloned.
        """
        if isinstance(index, int):
            index = [index]
        if isinstance(index, Iterable) and not isinstance(index, torch.Tensor):
            index = [i if not isinstance(i, int) else slice(i, i + 1) for i in index]
        return BatchTransform(matrix=self.get_matrix()[index])

    def __len__(self):
        return prod(self.b)

    def __setitem__(
        self,
        index: Union[int, List[int], slice, torch.BoolTensor, torch.LongTensor],
        value,
    ):
        """Sets items of self._matrix"""
        if isinstance(value, BatchTransform):
            value = value.get_matrix()
        if isinstance(index, Iterable) and not isinstance(index, torch.Tensor):
            index = [i if not isinstance(i, int) else slice(i, i + 1) for i in index]
        self._matrix[index] = value

    def __matmul__(self, other):
        """Handy shortcut for matrix multiplication."""
        if isinstance(other, BatchTransform):
            return self.compose(other)
        return NotImplemented

    def clone(self) -> BatchTransform:
        """Clones the transform."""
        return self.__class__(self._matrix.clone())

    def compose(self, *others: BatchTransform) -> BatchTransform:
        """
        Return a new Transform representing the composition of self with the given other transforms.

        Args:
            *others: Any number of BatchedTransform objects.

        Returns:
            A new BatchedTransform object with the composition of the input transforms.
        """
        mat = self.flat_tensor.clone()
        for other in others:
            if other.b != self.b:
                raise ValueError("All transforms must have the same batch size.")
            mat = torch.bmm(mat, other.flat_tensor)
        return BatchTransform(mat.view(*self.b, 4, 4))

    def get_matrix(self) -> torch.Tensor:
        """Returns the matrix representation of the transform."""
        return self._matrix

    def inverse(self) -> BatchTransform:
        """
        Returns a new BatchedTransform object that represents an inverse of the current transformations.

        Returns:
            The element-wise inverse of the current transformations.
        """
        inv = homogeneous_inverse(self.flat_tensor)
        if self.b is not None:
            inv = inv.view(*self.b, 4, 4)
        return self.__class__(inv)

    def stack(self, *others: BatchTransform, dim: int = 0) -> BatchTransform:
        """Stacks self with others along the batch dimensions."""
        transforms = [self] + list(others)
        if not all(t.b == self.b for t in transforms):
            raise ValueError("All transforms must have the same batch size.")
        matrix = torch.cat([t._matrix for t in transforms], dim=dim).to(
            self.device, self.dtype
        )
        return self.__class__(matrix)

    def rotate(self, *args):
        return self.compose(Rotate(*args))

    def translate(self, *args):
        return self.compose(Translate(*args))


class Translate(BatchTransform):
    """A class representing 3D translations."""

    def __init__(self, xyz: torch.Tensor) -> None:
        """
        Create a new BatchTransform representing 3D translations.
        """
        if xyz.dim() == 1:
            b = None
        else:
            b = xyz.shape[:-1]
        mat = batch_eye4(b, dtype=xyz.dtype, device=xyz.device)
        mat[..., 3, :3] = xyz
        super().__init__(mat)

    def inverse(self) -> Translate:
        """
        Return the inverse of self.
        """
        return self.__class__(-self._matrix[..., 3, :3])


class Rotate(BatchTransform):
    """A Transform that rotates points in 3D space."""

    def __init__(self, r: torch.Tensor) -> None:
        """
        Create a new Transform representing 3D rotation using rotation matrices as the input.

        Args:
            r: A tensor of rotation matrices with any number of batch dimensions.
        """
        if r.dim() == 2:
            b = None
        else:
            b = r.shape[:-2]
        mat = batch_eye4(b, dtype=r.dtype, device=r.device)
        mat[..., :3, :3] = r
        super().__init__(mat)

    def inverse(self) -> Rotate:
        """
        Return the inverse of self.
        """
        return self.__class__(self._matrix[..., :3, :3].transpose(-2, -1).contiguous())

    @classmethod
    def axis_angle(
        cls, angle: torch.Tensor, axis: str, degrees: bool = False
    ) -> Rotate:
        """
        Create a new Transform representing 3D rotation using axis-angle representation.

        Args:
            angle: A tensor of angles in radians.
            axis: The axis of rotation. One of 'x', 'y', 'z'.
            degrees: If True, the angles are interpreted as degrees.

        Returns:
            A new Rotate object representing the rotation.
        """
        axis = axis.upper()
        if axis not in ["X", "Y", "Z"]:
            msg = "Expected axis to be one of ['X', 'Y', 'Z']; got %s"
            raise ValueError(msg % axis)
        angle = (angle / 180.0 * torch.pi) if degrees else angle

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)
        if axis == "X":
            r_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "Y":
            r_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "Z":
            r_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError("letter must be either X, Y or Z.")

        rot = torch.stack(r_flat, -1).reshape(angle.shape + (3, 3))
        return cls(rot)


def batch_eye4(b: Optional[Union[Sequence[int], torch.Size]], **kwargs) -> torch.Tensor:
    """Creates a batch of 4x4 identity matrices."""
    if b is None:
        return torch.eye(4, **kwargs)
    return torch.eye(4, **kwargs).expand(*b, 4, 4).clone()


def homogeneous_inverse(matrix: torch.Tensor) -> torch.Tensor:
    """
    Computes the inverse of a batch of 4x4 homogeneous transformation matrices.

    Args:
        matrix: A tensor of shape (minibatch, 4, 4) representing the transformation matrices.

    Returns:
        A tensor of shape (minibatch, 4, 4) representing the inverses of the input matrices.
    """
    batched = True
    if matrix.dim() == 2:
        batched = False
        matrix = matrix.unsqueeze(0)

    if matrix.dim() != 3:
        raise ValueError(
            f"Expected input to be of shape (minibatch, 4, 4) or (4, 4), but got {matrix.shape}"
        )

    inv = (
        torch.eye(4, device=matrix.device, dtype=matrix.dtype)
        .unsqueeze(0)
        .expand_as(matrix)
    )
    inv[..., :3, :3] = torch.transpose(matrix[..., :3, :3], -1, -2)
    inv[..., :3, 3] = -torch.bmm(
        inv[..., :3, :3], matrix[..., :3, 3].unsqueeze(-1)
    ).squeeze(-1)
    if not batched:
        inv = inv.squeeze(0)
    return inv  # TODO: test against torch.inverse
