import numpy as np
import torch

from digeo.mesh import Mesh, MeshBatch, MeshPointBatch


def uniform_sampling(mesh: Mesh, n_points: int) -> MeshPointBatch:
    """
    Sample points uniformly from the mesh.

    Parameters
    ----------
    mesh : Mesh | MeshBatch
        The mesh to sample from.
    n_points : int
        The number of points to sample.
    tensor : bool
        Whether to use torch tensors for barycentric coordinates.

    Returns
    -------
    MeshPointBatch
        The sampled points.
    """
    tri = mesh.faces
    pos = mesh.vertices

    p0 = pos[tri[:, 0]]
    p1 = pos[tri[:, 1]]
    p2 = pos[tri[:, 2]]

    # Compute triangle areas
    v0 = p1 - p0
    v1 = p2 - p0
    cross = torch.cross(v0, v1, dim=1).to(torch.float64)
    weights = 0.5 * torch.linalg.norm(cross, axis=1)

    if isinstance(mesh, MeshBatch):
        faces = []
        for k in range(len(mesh.triangle_idx) - 1):
            weights_k = weights[mesh.triangle_idx[k] : mesh.triangle_idx[k + 1]]
            weights_k /= weights_k.sum()
            triangle_id_np = np.random.choice(
                len(weights_k), p=weights_k, size=n_points
            )
            faces.append(
                torch.tensor(triangle_id_np, dtype=torch.int32) + mesh.triangle_idx[k]
            )

        faces = torch.cat(faces, dim=0)

        # Generate random uniform barycentric coordinates
        u = torch.rand(n_points * len(mesh), dtype=torch.float32)
        v = (1 - u) * torch.rand(n_points * len(mesh), dtype=torch.float32)
        uvs = torch.stack((u, v), dim=1)

        return MeshPointBatch(faces=faces, uvs=uvs)

    elif isinstance(mesh, Mesh):
        weights /= weights.sum()
        triangle_id_np = np.random.choice(
            len(tri), p=weights.detach().cpu(), size=n_points
        )
        triangle_id = torch.tensor(triangle_id_np, dtype=torch.int32)

        # Generate random uniform barycentric coordinates
        u = torch.rand(n_points, dtype=torch.float32)
        v = (1 - u) * torch.rand(n_points, dtype=torch.float32)
        uvs = torch.stack((u, v), dim=1)

        return MeshPointBatch(faces=triangle_id, uvs=uvs)

    else:
        raise TypeError("Input mesh must be of type Mesh or MeshBatch.")
