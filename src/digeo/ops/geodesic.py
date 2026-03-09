import torch
from torch import Tensor
from typing import Tuple, List, Optional
from torch.autograd.function import once_differentiable

from digeo.ops.geodesic_utils import straightest_geodesic
from digeo.mesh import MeshPoint, Mesh, MeshPointBatch
from digeo.ops.cuda.straightest_geodesic import (
    straightest_geodesics as cuda_straightest_geodesics,
)


class GeodesicInfo:
    def __init__(
        self,
        debug_tensor: Optional[Tensor] = None,
        rotation: Optional[Tensor] = None,
        end_directions: Optional[Tensor] = None,
    ):
        """
        Represents the geodesic information including the rotation matrix and an
        optional debug tensor.
        """
        self.rotation_tensor = rotation
        self.path_len = None
        self.vertex_crossings = None
        self.path = None
        self.directions = None
        self.normals = None
        self.end_directions = end_directions
        if debug_tensor is not None:
            self.path_len = debug_tensor[:, 0, 0, 0]
            self.vertex_crossings = debug_tensor[:, 0, 0, 1]
            self.path = debug_tensor[:, 0, 1:]
            self.directions = debug_tensor[:, 1, 1:]
            self.normals = debug_tensor[:, 2, 1:]

    @property
    def rotation(self) -> Tensor:
        """
        Returns the rotation matrix used for parallel transport.
        """
        if self.rotation_tensor is None:
            raise RuntimeError(
                "Rotation tensor is not set. Set 'save_parallel_transport=True' in " \
                "trace_geodesics to save it."
            )
        return self.rotation_tensor

    def transport(self, v: Tensor) -> Tensor:
        """
        Parallel transports a direction vector v.
        """
        if self.rotation_tensor is None:
            raise RuntimeError(
                "Rotation tensor is not set. Set 'save_parallel_transport=True' in " \
                "trace_geodesics to save it."
            )
        if v.dim() != 2 or v.shape[1] != 3:
            raise RuntimeError(
                f"Input tensor should be of shape [batch_size, 3], got {v.shape}."
            )
        if v.shape[0] != self.rotation_tensor.shape[0]:
            raise RuntimeError(
                f"Input tensor batch size should match rotation tensor batch size, "
                f"got {v.shape[0]} and {self.rotation_tensor.shape[0]}."
            )
        return torch.bmm(v.unsqueeze(1), self.rotation_tensor).squeeze(1)

    def transport_inv(self, v: Tensor) -> Tensor:
        """
        Inverse parallel transports a direction vector v.
        """
        if self.rotation_tensor is None:
            raise RuntimeError(
                "Rotation tensor is not set. Set 'save_parallel_transport=True' in " \
                "trace_geodesics to save it."
            )
        if v.dim() != 2 or v.shape[1] != 3:
            raise RuntimeError(
                f"Input tensor should be of shape [batch_size, 3], got {v.shape}."
            )
        if v.shape[0] != self.rotation_tensor.shape[0]:
            raise RuntimeError(
                f"Input tensor batch size should match rotation tensor batch size, "
                f"got {v.shape[0]} and {self.rotation_tensor.shape[0]}."
            )
        return torch.bmm(v.unsqueeze(1), self.rotation_tensor.transpose(1, 2)).squeeze(
            1
        )

    def get_path(self, i: int) -> Tensor:
        """
        Returns the positions of the points of the geodesic starting from starts[i].
        Outputs a tensor of shape [path_length, 3]
        """
        if self.path is None or self.path_len is None:
            raise RuntimeError(
                "Path is not set. Set 'debug=True' in trace_geodesics to save it."
            )
        return self.path[i, : int(self.path_len[i].item())]

    def get_directions(self, i: int) -> Tensor:
        """
        Returns the directions of the geodesic starting from starts[i].
        Outputs a tensor of shape [path_length, 3]
        """
        if self.directions is None or self.path_len is None:
            raise RuntimeError(
                "Directions are not set. Set 'debug=True' in trace_geodesics to "
                "save it."
            )
        return self.directions[i, : int(self.path_len[i].item())]

    def get_normals(self, i: int) -> Tensor:
        """
        Returns the normals of the geodesic starting from starts[i].
        Outputs a tensor of shape [path_length, 3]
        """
        if self.normals is None or self.path_len is None:
            raise RuntimeError(
                "Normals are not set. Set 'debug=True' in trace_geodesics to save it."
            )
        return self.normals[i, : int(self.path_len[i].item())]


def trace_geodesics_gfd(
    mesh: Mesh, x: MeshPointBatch, dirs: Tensor, kwargs: dict
) -> MeshPointBatch:
    """
    Compute the straightest geodesics on the mesh starting from point x with
    direction v.
    This function is a wrapper for the StraighestGeodesicsGFD autograd function.
    """
    # Project on the tangent plane
    normals = mesh.triangle_normals[x.faces]
    dirs = dirs - torch.sum(dirs * normals, dim=1, keepdim=True) * normals

    x1_faces, x1_uvs = StraighestGeodesicsGFD.apply(mesh, x.faces, x.uvs, dirs, kwargs)
    x1 = MeshPointBatch(faces=x1_faces, uvs=x1_uvs)
    return x1


def trace_geodesics_abfd(
    mesh: Mesh, x: MeshPointBatch, dirs: Tensor, kwargs: dict
) -> MeshPointBatch:
    """
    Compute the straightest geodesics on the mesh starting from point x with
    direction v.
    This function is a wrapper for the StraighestGeodesicsABFD autograd function.
    """
    # Project on the tangent plane
    normals = mesh.triangle_normals[x.faces]
    dirs = dirs - torch.sum(dirs * normals, dim=1, keepdim=True) * normals

    x1_faces, x1_uvs = StraighestGeodesicsABFD.apply(mesh, x.faces, x.uvs, dirs, kwargs)
    x1 = MeshPointBatch(faces=x1_faces, uvs=x1_uvs)
    return x1


def inv_2x2(x: Tensor) -> Tensor:
    """
    Computes the inverse of a batch of 2x2 tensors.
    """
    scale = x.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    x_scaled = x / scale

    a = x_scaled[:, 0, 0]
    b = x_scaled[:, 0, 1]
    c = x_scaled[:, 1, 0]
    d = x_scaled[:, 1, 1]

    det = a * d - b * c
    inv_det = 1.0 / det.clamp(min=1e-8)

    inv_x = torch.stack((d, -b, -c, a), dim=-1).reshape(-1, 2, 2)
    inv_x = inv_det[:, None, None] * inv_x  # [B, 2, 2]
    inv_x = inv_x / scale
    return inv_x


class StraighestGeodesicsGFD(torch.autograd.Function):
    EPS = 1e-2

    @staticmethod
    def forward(
        ctx, mesh: Mesh, x0_faces: Tensor, x0_uvs: Tensor, v: Tensor, kwargs: dict
    ):
        x0 = MeshPointBatch(faces=x0_faces, uvs=x0_uvs)
        x1, info = trace_geodesics(
            mesh, x0, v, gradient="none", save_end_direction=True, **kwargs
        )

        ctx.save_for_backward(
            x0.faces, x0_uvs, v, info.end_directions, x1.faces, x1.uvs
        )
        ctx.mesh = mesh
        ctx.kwargs = kwargs

        return x1.faces, x1.uvs

    @staticmethod
    @once_differentiable
    def backward(ctx, gr_x1_faces, grad_x1_uv: Tensor):
        # ignore gr_x1_faces, should be None

        x0_faces, x0_uvs, v, v1, x1_faces, x1_uvs = ctx.saved_tensors
        x0 = MeshPointBatch(
            faces=x0_faces, uvs=x0_uvs
        )  # Reconstruct MeshPointBatch from saved tensors
        x1 = MeshPointBatch(
            faces=x1_faces, uvs=x1_uvs
        )  # Reconstruct MeshPointBatch from saved tensors
        mesh = ctx.mesh
        eps = StraighestGeodesicsGFD.EPS

        p0 = mesh.vertices[mesh.faces[x0.faces][:, 0]]
        p1 = mesh.vertices[mesh.faces[x0.faces][:, 1]]
        p2 = mesh.vertices[mesh.faces[x0.faces][:, 2]]
        x0_e1 = p1 - p0
        x0_e2 = p2 - p0
        x0_e1 = x0_e1 / torch.norm(x0_e1, dim=1, keepdim=True)  # (B, 3)
        x0_e2 = x0_e2 / torch.norm(x0_e2, dim=1, keepdim=True)  # (B, 3)

        p0 = mesh.vertices[mesh.faces[x1.faces][:, 0]]
        p1 = mesh.vertices[mesh.faces[x1.faces][:, 1]]
        p2 = mesh.vertices[mesh.faces[x1.faces][:, 2]]
        x1_e1 = p1 - p0
        x1_e2 = p2 - p0
        T1 = torch.stack((x1_e1, x1_e2), dim=-1)  # (B, 3, 2)
        TtT_inv = inv_2x2(torch.bmm(T1.mT, T1))  # [B, 2, 2]
        T1_inv = torch.bmm(TtT_inv, T1.mT)  # [B, 2, 3]

        x0_e1_eps, info_xe1 = trace_geodesics(
            mesh,
            x0,
            eps * x0_e1,
            gradient="none",
            save_parallel_transport=True,
            **ctx.kwargs,
        )  # (B, 3)
        x0_e2_eps, info_xe2 = trace_geodesics(
            mesh,
            x0,
            eps * x0_e2,
            gradient="none",
            save_parallel_transport=True,
            **ctx.kwargs,
        )  # (B, 3)

        v_xe1 = info_xe1.transport(v)
        v_xe2 = info_xe2.transport(v)

        y1, _ = trace_geodesics(
            mesh, x0_e1_eps, v_xe1, gradient="none", **ctx.kwargs
        )  # (B, 3)
        y2, _ = trace_geodesics(
            mesh, x0_e2_eps, v_xe2, gradient="none", **ctx.kwargs
        )  # (B, 3)

        # Jacobian
        J_x0 = torch.stack(
            (
                (y1.interpolate(mesh) - x1.interpolate(mesh)) / eps,
                (y2.interpolate(mesh) - x1.interpolate(mesh)) / eps,
            ),
            dim=-1,
        )  # (B, 3, 2)

        J_x0_local = torch.bmm(T1_inv, J_x0)  # (B, 2, 2)

        grad_x0_uvs = torch.bmm(grad_x1_uv.unsqueeze(1), J_x0_local).squeeze(
            1
        )  # (B, 2) @ (B, 2, 2) -> (B, 2)

        n0 = mesh.triangle_normals[x0.faces]  # (B, 3)
        d1 = v / torch.norm(v, dim=1, keepdim=True)  # (B, 3)
        d2 = torch.cross(n0, d1, dim=1)  # (B, 3)

        # Start directly from x1 since the pertubation is along v
        y1, _ = trace_geodesics(
            mesh, x1, eps * v1, gradient="none", **ctx.kwargs
        )  # (B, 3)
        # Here we need to retrace the geodesic from x0
        y2, _ = trace_geodesics(
            mesh, x0, v + eps * d2, gradient="none", **ctx.kwargs
        )  # (B, 3)

        # Jacobian
        J_v = torch.stack(
            (
                (y1.interpolate(mesh) - x1.interpolate(mesh)) / eps,
                (y2.interpolate(mesh) - x1.interpolate(mesh)) / eps,
            ),
            dim=-1,
        )

        grad_x1 = torch.bmm(grad_x1_uv.unsqueeze(1), T1_inv).squeeze(1)  # (B, 3)

        grad_v_local = torch.bmm(grad_x1.unsqueeze(1), J_v).squeeze(
            1
        )  # (B, 2) @ (B, 2, 2) -> (B, 2)

        grad_v = (
            grad_v_local[:, 0:1] * d1 + grad_v_local[:, 1:2] * d2
        )  # (B,) * (B, 3) + (B,) * (B, 3) -> (B, 3)

        return None, None, grad_x0_uvs, grad_v, None


class StraighestGeodesicsABFD(torch.autograd.Function):
    EPS = 1e-2

    @staticmethod
    def forward(
        ctx, mesh: Mesh, x0_faces: Tensor, x0_uvs: Tensor, v: Tensor, kwargs: dict
    ):
        x0 = MeshPointBatch(faces=x0_faces, uvs=x0_uvs)
        x1, info = trace_geodesics(
            mesh, x0, v, gradient="none", save_end_direction=True, **kwargs
        )

        ctx.save_for_backward(
            x0.faces, x0_uvs, v, info.end_directions, x1.faces, x1.uvs
        )
        ctx.mesh = mesh
        ctx.kwargs = kwargs

        return x1.faces, x1.uvs

    @staticmethod
    @once_differentiable
    def backward(ctx, gr_x1_faces, grad_x1_uv: Tensor):
        x0_faces, x0_uvs, v, v1, x1_faces, x1_uvs = ctx.saved_tensors
        mesh = ctx.mesh
        x0 = MeshPointBatch(
            faces=x0_faces, uvs=x0_uvs
        )  # Reconstruct MeshPointBatch from saved tensors
        x1 = MeshPointBatch(
            faces=x1_faces, uvs=x1_uvs
        )  # Reconstruct MeshPointBatch from saved tensors
        eps = StraighestGeodesicsABFD.EPS

        v_norm = v.norm(dim=-1, keepdim=True)
        v1 = v_norm * v1
        n = mesh.triangle_normals[x0.faces]
        v = v - torch.sum(v * n, dim=-1, keepdim=True) * n
        v = v_norm * (v / v.norm(dim=-1, keepdim=True))

        p0 = mesh.vertices[mesh.faces[x1.faces][:, 0]]
        p1 = mesh.vertices[mesh.faces[x1.faces][:, 1]]
        p2 = mesh.vertices[mesh.faces[x1.faces][:, 2]]
        x1_e1 = p1 - p0
        x1_e2 = p2 - p0
        T1 = torch.stack((x1_e1, x1_e2), dim=-1)  # (B, 3, 2)
        TtT_inv = inv_2x2(torch.bmm(T1.mT, T1))  # [B, 2, 2]
        T1_inv = torch.bmm(TtT_inv, T1.mT)  # [B, 2, 3]

        x1_e = torch.bmm(grad_x1_uv.unsqueeze(1), T1_inv).squeeze(
            1
        )  # (B, 2) @ (B, 2, 3) -> (B, 3)
        len_grad_x1 = torch.norm(x1_e, dim=1, keepdim=True)
        x1_e = x1_e / x1_e.norm(dim=-1, keepdim=True)  # (B, 3)

        x1_eps, info = trace_geodesics(
            mesh,
            x1,
            eps * x1_e,
            gradient="none",
            save_parallel_transport=True,
            **ctx.kwargs,
        )  # (B, 3)
        v1_e = info.transport(v1)
        x0_e, _ = trace_geodesics(
            mesh, x1_eps, -v1_e, gradient="none", **ctx.kwargs
        )  # (B, 3)

        grad_v = (x0_e.interpolate(mesh) - x0.interpolate(mesh)) / eps
        grad_v = len_grad_x1 * grad_v

        n = mesh.triangle_normals[x0.faces]
        grad_v = grad_v - torch.sum(grad_v * n, dim=-1, keepdim=True) * n

        p0 = mesh.vertices[mesh.faces[x0.faces][:, 0]]
        p1 = mesh.vertices[mesh.faces[x0.faces][:, 1]]
        p2 = mesh.vertices[mesh.faces[x0.faces][:, 2]]
        x0_e1 = p1 - p0
        x0_e2 = p2 - p0

        grad_x0_u = torch.sum(grad_v * x0_e1, dim=1, keepdim=True)  # (B, 1)
        grad_x0_v = torch.sum(grad_v * x0_e2, dim=1, keepdim=True)  # (B, 1)
        grad_x0_uvs = torch.cat([grad_x0_u, grad_x0_v], dim=1)  # (B, 2)

        return None, None, grad_x0_uvs, grad_v, None


def project_batch_to_plane(dirs: Tensor, normals: Tensor) -> Tensor:
    return dirs - normals * torch.sum(dirs * normals, dim=1, keepdim=True)


def get_tensors_from_path(
    mesh: Mesh,
    end_points: MeshPointBatch,
    end_dirs: Tensor,
    end_normals: Tensor,
    starts: MeshPointBatch,
    start_dirs: Tensor,
    debug=False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Convert an endpoint to a tensor representation.
    """
    start_normals = mesh.triangle_normals[starts.faces]
    start_dirs_unit = project_batch_to_plane(start_dirs, start_normals)
    start_dirs_unit = start_dirs_unit / torch.linalg.norm(
        start_dirs_unit, dim=1, keepdim=True
    )

    end_dirs_norm = torch.linalg.norm(end_dirs, dim=1, keepdim=True)
    indices = torch.where(end_dirs_norm < 10e-6)[0]

    end_dirs_unit = end_dirs / end_dirs_norm

    start_cross = torch.cross(start_normals, start_dirs_unit, dim=1)
    end_cross = torch.cross(end_normals, end_dirs_unit, dim=1)

    # create matrix from orthonormal basis
    start_M = torch.stack([start_dirs_unit, start_normals, start_cross], dim=1)
    end_M = torch.stack([end_dirs_unit, end_normals, end_cross], dim=1)

    # rotation matrix from basis to basis
    R = torch.bmm(start_M.transpose(1, 2), end_M).detach()
    R[indices] = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0)

    start_points = starts.interpolate(mesh)

    target_end_intermediate = starts.interpolate(
        mesh
    ).detach() + project_batch_to_plane(start_dirs.detach(), start_normals)
    target_end = torch.bmm(target_end_intermediate.unsqueeze(1), R).squeeze(1).detach()

    T = end_points.interpolate(mesh).detach() - target_end

    tensor_point_intermediate = start_points + project_batch_to_plane(
        start_dirs, start_normals
    )
    tensor_point = torch.bmm(tensor_point_intermediate.unsqueeze(1), R).squeeze(1) + T

    return tensor_point, indices


def tri_bary_coords(p0: Tensor, p1: Tensor, p2: Tensor, p: Tensor) -> Tensor:
    """Compute barycentric coordinates of p in the triangle (p0, p1, p2)."""
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p - p0

    d00 = torch.sum(v0 * v0, dim=1)
    d01 = torch.sum(v0 * v1, dim=1)
    d11 = torch.sum(v1 * v1, dim=1)
    d20 = torch.sum(v2 * v0, dim=1)
    d21 = torch.sum(v2 * v1, dim=1)

    denom = d00 * d11 - d01 * d01
    denom_mask = denom == 0
    denom[denom_mask] = 1.0

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    u[denom_mask] = 1.0
    v[denom_mask] = 0.0
    w[denom_mask] = 0.0

    return torch.stack([u, v, w], dim=1)


def get_meshpoints(mesh: Mesh, faces: Tensor, points: Tensor) -> MeshPointBatch:
    triangle = mesh.faces[faces]
    p0 = mesh.vertices[triangle[:, 0]]
    p1 = mesh.vertices[triangle[:, 1]]
    p2 = mesh.vertices[triangle[:, 2]]
    bary = tri_bary_coords(p0, p1, p2, points)
    return MeshPointBatch(faces=faces, uvs=bary[:, 1:])


def trace_geodesics(
    mesh: Mesh,
    starts: List[MeshPoint] | MeshPointBatch,
    dirs: torch.Tensor,
    gradient: str = "gfd",
    use_python: bool = False,
    max_steps: int = 2000,
    save_parallel_transport: bool = False,
    save_end_direction: bool = False,
    debug: bool = False,
    print_warnings: bool = True,
    eps: float = -1.0,
    avoid_holes: bool = False,
) -> Tuple[MeshPointBatch, GeodesicInfo]:
    """
    Computes the straightest geodesic traces.

    Parameters
    ----------
    mesh : Mesh
        The mesh to trace the geodesics
    starts: List[MeshPoint] | MeshPointBatch
        The starting points from which the geodesics start
    dirs: torch.Tensor
        The direction tensor (in 3d world coordinates)
    gradient : str,
        What method to compute the gradient, can be "none", "ep",
        "abfd", or "gfd" (default: "gfd").
    use_python : bool,
        If the computation should be made with python, making it easier to debug
        (default: False).
    max_steps : int,
        The maximum steps for the traces (triangle hops) (default: 2000).
    save_parallel_transport : bool,
        If the parallel transport rotation should be saved in the GeodesicInfo
        (default: False).
    save_end_direction : bool,
        If the end direction should be saved in the GeodesicInfo (default: False).
    debug : bool,
        If the returned GeodesicInfo should contain additionnal debug info
        (default: False).
    print_warnings : bool,
        If warnings should be printed (default: True).
    eps : float,
        A small epsilon value to avoid numerical issues (defaults to 10e-4 for floats
        and 10e-7 for doubles)
    avoid_holes : bool,
        If the geodesic tracing should circumvent holes in the mesh (default: False)
        This is recommanded for meshes with small holes, but discrupts the geodesic
        tracing.

    Returns
    -------
    MeshPointBatch, GeodesicInfo
        The endpoints of the geodesic traces and geodesic information.

    >>> mesh = load_mesh_from_file(path_to_mesh, device=device)
    >>> start_meshpoints = uniform_sampling(mesh, N).to(device)
    >>> start_directions = torch.randn((N, 3), dtype=torch.float32).to(device)
    >>> meshpoints, geodesic_info = trace_geodesics(
    ...     mesh, start_meshpoints, start_directions
    ... )
    """
    if dirs.dim() != 2:
        raise RuntimeError("Dirs tensor should be of shape [batch_size, 3]")

    if len(starts) != dirs.shape[0]:
        raise RuntimeError(
            f"Starts and dirs should have the same batch size, got {len(starts)} "
            f"and {dirs.shape[0]}."
        )

    if use_python and str(mesh.device) != "cpu":
        raise ValueError(
            "Python geodesic tracing is only supported on CPU. Tensors must be on CPU "
            "for Python tracing."
        )

    if use_python and gradient == "gfd":
        raise ValueError(
            "GFD gradient is not supported with Python geodesic tracing. "
            "Use C++/CUDA for this feature."
        )

    starts = (
        MeshPointBatch.from_points(starts).to(device=mesh.device)
        if not isinstance(starts, MeshPointBatch)
        else starts
    )

    # Check if all tensors are on the same device
    if mesh.device != starts.uvs.device:
        raise RuntimeError(
            f"Tensors must be on the same device. Found mesh on {mesh.device}, start "
            f"uvs on {starts.uvs.device}."
        )
    if mesh.device != starts.faces.device:
        raise RuntimeError(
            f"Tensors must be on the same device. Found mesh on {mesh.device}, start "
            f"faces on {starts.faces.device}."
        )
    if mesh.device != dirs.device:
        raise RuntimeError(
            f"Tensors must be on the same device. Found mesh on {mesh.device}, dirs "
            f"on {dirs.device}."
        )

    debug_tensor = None
    with torch.no_grad():
        if not (
            gradient != "gfd"
            or gradient != "abfd"
            or debug
            or save_parallel_transport
            or save_end_direction
        ):
            pass
        elif use_python:
            meshpoints = []
            directions = torch.empty(len(dirs), 3, dtype=dirs.dtype, device=mesh.device)
            normals = torch.empty(len(dirs), 3, dtype=dirs.dtype, device=mesh.device)
            if debug:
                debug_tensor = torch.empty(
                    (len(dirs), 3, max_steps + 1, 3),
                    dtype=dirs.dtype,
                    device=mesh.device,
                )
            for i, (start, dir) in enumerate(zip(starts, dirs)):
                result = straightest_geodesic(
                    mesh,
                    start.detach(),
                    dir.detach(),
                    max_steps=max_steps,
                    debug_path=debug,
                    print_warnings=print_warnings,
                    eps=eps,
                    avoid_holes=avoid_holes,
                )
                meshpoints.append(result[0])
                directions[i] = result[1]
                normals[i] = result[2]
                if debug:
                    path, direction_path, normal_path, vertex_crossings = result[3]
                    debug_tensor[i, 0, 0, 0] = len(path)
                    debug_tensor[i, 0, 0, 1] = vertex_crossings
                    debug_tensor[i, 0, 1 : len(path) + 1, :] = torch.tensor(path)
                    debug_tensor[i, 1, 1 : len(direction_path) + 1, :] = torch.tensor(
                        direction_path
                    )
                    debug_tensor[i, 2, 1 : len(normal_path) + 1, :] = torch.tensor(
                        normal_path
                    )
            meshpoints = MeshPointBatch.from_points(meshpoints).to(device=mesh.device)
            results = (meshpoints, directions, normals)
        else:
            results = cuda_straightest_geodesics(
                mesh,
                starts,
                dirs.detach(),
                max_steps=max_steps,
                debug=debug,
                print_warnings=print_warnings,
                eps=eps,
                avoid_holes=avoid_holes,
            )
            if debug:
                debug_tensor = results[3]

    kwargs = dict(
        max_steps=max_steps,
        print_warnings=print_warnings,
        eps=eps,
        avoid_holes=avoid_holes,
    )

    # Compute gradients if needed
    if gradient == "none":
        mesh_points = results[0]
    elif gradient == "ep":
        tensor_points, error_indices = get_tensors_from_path(
            mesh, results[0], results[1], results[2], starts, dirs, debug=debug
        )
        mesh_points = get_meshpoints(mesh, results[0].faces, tensor_points)
        if debug and error_indices is not None:
            for i in error_indices:
                i = int(i.item())
                print(
                    f"Warning: null output for face {starts[i].face}, "
                    f"uv {starts[i].uv.tolist()}, dir {dirs[i].tolist()}."
                )
    elif gradient == "gfd":
        mesh_points = trace_geodesics_gfd(mesh, starts, dirs, kwargs)
    elif gradient == "abfd":
        mesh_points = trace_geodesics_abfd(mesh, starts, dirs, kwargs)
    else:
        raise ValueError(
            f"Unknown gradient method: {gradient}. Use 'none', 'ep', "
            f"'gfd', or 'abfd'."
        )

    # Compute parallel transport rotation if needed
    rotation = None
    if save_parallel_transport:
        start_normals = mesh.triangle_normals[starts.faces]
        end_normals = results[2]

        start_dirs_unit = (
            dirs - torch.sum(dirs * start_normals, dim=1, keepdim=True) * start_normals
        )
        start_dirs_norm = start_dirs_unit.norm(dim=1, keepdim=True)
        idle_idx = start_dirs_norm[:, 0] < 1e-6
        start_dirs_unit = start_dirs_unit / start_dirs_norm.clamp(min=1e-6)
        end_dirs_unit = results[1] / torch.linalg.norm(
            results[1], dim=1, keepdim=True
        ).clamp(min=1e-6)

        start_cross = torch.cross(start_normals, start_dirs_unit, dim=1)
        end_cross = torch.cross(end_normals, end_dirs_unit, dim=1)

        # create matrix from orthonormal basis
        start_M = torch.stack([start_dirs_unit, start_normals, start_cross], dim=1)
        end_M = torch.stack([end_dirs_unit, end_normals, end_cross], dim=1)

        # rotation matrix from basis to basis
        rotation = torch.bmm(start_M.transpose(1, 2), end_M).detach()
        rotation[idle_idx] = torch.eye(3, device=rotation.device).unsqueeze(0)

    end_dir = None
    if save_end_direction:
        end_dir = results[1]

    geodesic_info = GeodesicInfo(
        debug_tensor=debug_tensor, rotation=rotation, end_directions=end_dir
    )

    return mesh_points, geodesic_info
