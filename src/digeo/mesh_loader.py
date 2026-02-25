import numpy as np
from typing import Union
from pathlib import Path
import trimesh
import torch

from digeo.mesh import Mesh


def create_triangle() -> Mesh:
    """Create a simple triangle mesh."""

    # Define vertices
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Define triangles (faces)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)

    mesh = Mesh(positions=positions, triangles=triangles)

    return mesh


def create_tetrahedron() -> Mesh:
    """Create a tetrahedron mesh."""

    # Define vertices
    positions = np.array(
        [
            [0.0, 0.0, 0.0],  # Vertex 0
            [1.0, 0.0, 0.0],  # Vertex 1
            [0.0, 1.0, 0.0],  # Vertex 2
            [0.0, 0.0, 1.0],  # Vertex 3
        ],
        dtype=np.float64,
    )

    # Define triangles (faces)
    triangles = np.array(
        [
            [0, 1, 2],  # Face 0: Base triangle
            [0, 1, 3],  # Face 1: Side triangle
            [1, 2, 3],  # Face 2: Side triangle
            [0, 2, 3],  # Face 3: Side triangle
        ],
        dtype=np.int32,
    )

    mesh = Mesh(positions=positions, triangles=triangles)

    return mesh


def load_mesh_from_trimesh(
    mesh: trimesh.Trimesh, device="cpu", dtype=torch.float32
) -> Mesh:
    """
    Load a mesh from a trimesh object.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The trimesh object

    Returns
    -------
    Mesh
        The loaded mesh
    """
    # remove degenerate faces
    degenerate = np.any(
        [
            mesh.faces[:, 0] == mesh.faces[:, 1],
            mesh.faces[:, 1] == mesh.faces[:, 2],
            mesh.faces[:, 2] == mesh.faces[:, 0],
        ],
        axis=0,
    )
    mesh.update_faces(~degenerate)

    return Mesh(
        positions=mesh.vertices,
        triangles=mesh.faces,
        adjacencies=None,
        triangle_normals=mesh.face_normals.copy(),
        v2t=None,
        vertex_normals=mesh.vertex_normals.copy(),
        device=device,
        dtype=dtype,
    )


def load_mesh_from_file(
    filename: Union[str, Path], device="cpu", dtype=torch.float32
) -> Mesh:
    """
    Load a mesh from a file. The file type is determined from the extension.

    Parameters
    ----------
    filename: str | Path
        Path to the mesh file

    Returns
    -------
    Mesh
        The loaded mesh
    """
    mesh_trimesh = trimesh.load_mesh(filename, maintain_order=True, process=False)
    return load_mesh_from_trimesh(mesh_trimesh, device=device, dtype=dtype)
