import torch
from torch import Tensor


def triangle_normal(p0: Tensor, p1: Tensor, p2: Tensor) -> Tensor:
    """Compute the normal of a triangle."""
    return normalize(torch.linalg.cross(p1 - p0, p2 - p0))


def normalize(v: Tensor) -> Tensor:
    """Normalize a vector."""
    norm = torch.norm(v).item()
    if norm == 0:
        return v
    return v / norm


def dot(v1: Tensor, v2: Tensor) -> float:
    """Compute dot product between two vectors."""
    return torch.dot(v1, v2).item()


def cross(v1: Tensor, v2: Tensor) -> Tensor:
    """Compute cross product between two vectors."""
    return torch.linalg.cross(v1, v2, dim=0)


def length(v: Tensor) -> float:
    """Compute the length of a vector."""
    return torch.norm(v).item()


def area_triangle(p0: Tensor, p1: Tensor, p2: Tensor) -> float:
    v1 = p1 - p0
    v2 = p2 - p0
    return length(cross(v1, v2)) / 2
