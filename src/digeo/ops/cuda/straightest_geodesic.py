import torch
from torch import Tensor
from typing import Tuple, Optional

from digeo.mesh import Mesh, MeshPointBatch

__all__ = ["straightest_geodesics"]


def straightest_geodesics(
    mesh: Mesh,
    start_points: MeshPointBatch,
    start_direction: Tensor,
    max_steps: int = 2000,
    debug: bool = False,
    print_warnings: bool = False,
    eps: float = -1.0,
    avoid_holes: bool = False,
) -> Tuple[MeshPointBatch, Tensor, Tensor, Optional[Tensor]]:

    geodesic_result, debug_tensor = (
        torch.ops.geodesic_utils.straightest_geodesics.default(
            mesh.vertices,
            mesh.faces,
            mesh.adjacencies,
            mesh.triangle_normals,
            mesh.v2t,
            mesh.vertex_normals,
            start_points.uvs,
            start_direction,
            start_points.faces,
            max_steps,
            debug,
            print_warnings,
            eps,
            avoid_holes,
        )
    )

    mesh_points = MeshPointBatch(
        faces=geodesic_result[:, 0, 0].to(torch.int32), uvs=geodesic_result[:, 0, 1:3]
    )
    directions = geodesic_result[:, 1, :]
    normals = geodesic_result[:, 2, :]
    if not debug:
        debug_tensor = None

    return mesh_points, directions, normals, debug_tensor


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("geodesic_utils::straightest_geodesics")
def _(
    mesh_positions: Tensor,
    mesh_triangles: Tensor,
    mesh_adjacencies: Tensor,
    mesh_triangle_normals: Tensor,
    mesh_v2t: Tensor,
    start_uv: Tensor,
    start_dir: Tensor,
    start_face: Tensor,
) -> Tuple[Tensor, Tensor]:
    torch._check(mesh_positions.ndim == 2)
    torch._check(mesh_triangles.ndim == 2)
    torch._check(mesh_adjacencies.ndim == 2)
    torch._check(mesh_triangle_normals.ndim == 2)
    torch._check(mesh_v2t.ndim == 2)
    torch._check(start_uv.ndim == 2)
    torch._check(start_dir.ndim == 2)
    torch._check(start_face.ndim == 1)

    torch._check(mesh_positions.shape[1] == 3)
    torch._check(mesh_triangles.shape[1] == 3)
    torch._check(mesh_adjacencies.shape[1] == 3)
    torch._check(mesh_triangle_normals.shape[1] == 3)
    torch._check(start_uv.shape[1] == 2)
    torch._check(start_dir.shape[1] == 3)

    torch._check(mesh_positions.dtype == torch.float)
    torch._check(mesh_triangles.dtype == torch.int32)
    torch._check(mesh_adjacencies.dtype == torch.int32)
    torch._check(mesh_triangle_normals.dtype == torch.float)
    torch._check(mesh_v2t.dtype == torch.int32)
    torch._check(start_uv.dtype == torch.float)
    torch._check(start_dir.dtype == torch.float)
    torch._check(start_face.dtype == torch.int32)

    return torch.empty((start_face.shape[0], 3, 3)), torch.empty((1, 1, 1, 1))
