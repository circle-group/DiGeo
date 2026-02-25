from typing import List, Optional, Tuple, Any
from torch import Tensor
import torch
import math
import numpy as np

from digeo.mesh import Mesh, MeshPoint
from digeo.utils import dot, cross, length, normalize, triangle_normal


DEBUG_MODE = False

GeodesicResult = Tuple[MeshPoint, Tensor, Tensor, Any]


class GeodesicPath:
    def __init__(self, start, end, path=[], dirs=[], normals=[]):
        self.start: MeshPoint = start
        self.end: MeshPoint = end
        self.path: List[Tensor] = path
        self.dirs: List[Tensor] = dirs
        self.normals: List[Tensor] = normals


def tri_bary_coords(p0: Tensor, p1: Tensor, p2: Tensor, p: Tensor) -> Tensor:
    """Compute barycentric coordinates of p in the triangle (p0, p1, p2)."""
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p - p0

    d00 = dot(v0, v0)
    d01 = dot(v0, v1)
    d11 = dot(v1, v1)
    d20 = dot(v2, v0)
    d21 = dot(v2, v1)

    denom = d00 * d11 - d01 * d01

    if abs(denom) == 0:
        return torch.tensor([1, 0, 0])

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return torch.tensor([u, v, w])


def tri_edge_coords(p0: Tensor, p1: Tensor, p2: Tensor, d: Tensor) -> Tensor:
    """Express direction vector d in the triangle (p0, p1, p2)'s edge basis."""
    e1 = p1 - p0
    e2 = p2 - p0

    d00 = dot(e1, e1)
    d01 = dot(e1, e2)
    d11 = dot(e2, e2)
    d20 = dot(d, e1)
    d21 = dot(d, e2)

    denom = d00 * d11 - d01 * d01
    if abs(denom) == 0:
        return torch.zeros(3)

    a = (d11 * d20 - d01 * d21) / denom
    b = (d00 * d21 - d01 * d20) / denom

    return torch.tensor([-a - b, a, b])


def bary_is_edge(bary: Tensor, eps: float) -> Tuple[bool, int]:
    """Check if barycentric coordinates are on an edge and return the edge index."""

    if bary[0].item() < eps:
        return True, 1
    if bary[1].item() < eps:
        return True, 2
    if bary[2].item() < eps:
        return True, 0

    return False, -1


def bary_is_vert(bary: Tensor, eps: float) -> Tuple[bool, int]:
    """Check if barycentric coordinates are on a vertex and return the vertex index."""

    if bary[0].item() > 1 - eps:
        return True, 0
    if bary[1].item() > 1 - eps:
        return True, 1
    if bary[2].item() > 1 - eps:
        return True, 2

    return False, -1


def project_vec(v: Tensor, normal: Tensor) -> Tensor:
    """Project a vector onto a plane defined by its normal."""
    normal = normalize(normal)
    return v - dot(v, normal) * normal


def trace_around_hole(
    mesh: Mesh,
    dir_3d: Tensor,
    curr_bary: Tensor,
    curr_pos: Tensor,
    curr_tri: int,
    max_len: float,
    eps: float,
) -> Tuple[Tensor, Tensor]:
    vertex_idx = -1
    edge_idx = -1
    for i in range(3):
        if curr_bary[i] > 1 - eps:
            vertex_idx = i
            break
        elif curr_bary[i] < eps:
            edge_idx = i

    next_vertex = -1
    if vertex_idx != -1:
        hole_edge = -1
        for k in range(3):
            if mesh.adjacencies[curr_tri][k] == -1:
                hole_edge = k

        if hole_edge == -1:
            raise RuntimeError("No hole edge found in the current triangle.")

        edge = [[0, 1], [1, 2], [2, 0]][hole_edge]
        next_vertex = edge[1] if edge[0] == vertex_idx else edge[0]

    elif edge_idx != -1:  # Select arbitrary next vertex on the edge
        next_vertex_map = [1, 2, 0]
        next_vertex = next_vertex_map[edge_idx]

    else:
        raise RuntimeError(
            "Current barycentric coordinates do not correspond to a vertex or edge."
        )

    dir = mesh.positions[mesh.triangles[curr_tri][next_vertex]] - curr_pos
    length_dir = length(dir)
    if length_dir > max_len:
        dir = dir / length_dir * max_len
    next_pos = curr_pos + dir
    next_bary = torch.zeros(3)
    next_bary[next_vertex] = 1.0
    return next_pos, next_bary


def trace_in_triangles(
    mesh: Mesh,
    dir_3d: Tensor,
    curr_bary: Tensor,
    curr_pos: Tensor,
    curr_tri: int,
    max_len: float,
    eps: float,
) -> Tuple[Tensor, Tensor]:
    """Trace a straight line within triangles."""
    p0 = mesh.positions[mesh.triangles[curr_tri][0]]
    p1 = mesh.positions[mesh.triangles[curr_tri][1]]
    p2 = mesh.positions[mesh.triangles[curr_tri][2]]
    dir_bary = tri_edge_coords(p0, p1, p2, dir_3d)

    best_t = 10000.0

    for i in range(3):
        if dir_bary[i] < -eps:
            t = -curr_bary[i] / dir_bary[i]
            if t > 0 and t < best_t:
                best_t = t

    if best_t > max_len:
        best_t = max_len

    next_pos = curr_pos + best_t * dir_3d
    next_bary = curr_bary + best_t * dir_bary

    # check if point is valid
    for i in range(3):
        if next_bary[i] < -eps:
            return curr_pos, curr_bary

    is_vertex = (next_bary[0] < eps) + (next_bary[1] < eps) + (next_bary[2] < eps) == 2

    if is_vertex:
        for i in range(3):
            if next_bary[i] > 0.5:
                next_bary[i] = 1.0
                next_pos = mesh.positions[mesh.triangles[curr_tri][i]]
            else:
                next_bary[i] = 0.0

    return next_pos, next_bary


def common_edge(
    triangles: Tensor, tri1: int, tri2: int
) -> Tuple[List[int], Optional[int], Optional[int]]:
    """Find the common edge between two triangles."""
    # Get the vertices of both triangles
    verts1 = set(triangles[tri1, :].tolist())
    verts2 = set(triangles[tri2, :].tolist())

    # Find the common vertices
    common_verts = verts1.intersection(verts2)
    diff_vert_1 = verts1 - common_verts
    diff_vert_2 = verts2 - common_verts
    diff_vert_1 = diff_vert_1.pop() if diff_vert_1 else None
    diff_vert_2 = diff_vert_2.pop() if diff_vert_2 else None

    if len(common_verts) != 2:
        return [], None, None

    return list(common_verts), diff_vert_1, diff_vert_2


def signed_angle(A: Tensor, B: Tensor, N: Tensor) -> float:
    """
    Compute the signed angle between two vectors A and B with respect to a
    normal vector N.
    """
    N = N / length(N)
    A = A - dot(A, N) * N
    B = B - dot(B, N) * N
    if length(A) == 0 or length(B) == 0:
        return 0.0
    A = A / length(A)
    B = B / length(B)

    cross_prod = cross(A, B)
    dot_prod = dot(A, B)
    sign = dot(N, cross_prod)
    angle = np.arctan2(sign, dot_prod).item()
    return angle  # in radians


def compute_parallel_transport_edge(
    mesh: Mesh, curr_tri: int, next_tri: int, dir_3d: Tensor, curr_normal: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Compute the parallel transport of a vector from one triangle to another at an edge.
    """
    if curr_tri == next_tri:
        return dir_3d, curr_normal

    # Find the common edge between triangles
    common_e, other_v1, other_v2 = common_edge(mesh.triangles, curr_tri, next_tri)

    if len(common_e) == 0:
        return dir_3d, curr_normal  # No common edge found

    # Get the normals of both triangles
    p0 = mesh.positions[common_e[0]]
    p1 = mesh.positions[common_e[1]]
    p2_curr = mesh.positions[other_v1]
    p2_next = mesh.positions[other_v2]

    n1 = triangle_normal(p0, p1, p2_curr)
    n2 = triangle_normal(p1, p0, p2_next)

    # Get the edge direction
    edge_dir = mesh.positions[common_e[1]] - mesh.positions[common_e[0]]
    axis = normalize(edge_dir)

    # Compute the rotation angle
    angle = signed_angle(n1, n2, axis)

    dir = rotate_vector(dir_3d, angle, axis)

    normal_sign = dot(n1, curr_normal)
    normal = normal_sign * n2
    return dir, normal


def compute_parallel_transport_vertex_hole(
    mesh: Mesh,
    curr_tri: int,
    vertex_id: int,
    dir_3d: Tensor,
    curr_normal: Tensor,
    eps: float,
) -> Tuple[Tensor, int, Tensor, bool]:
    tri_list = mesh.v2t[vertex_id, 1 : mesh.v2t[vertex_id, 0] + 1]
    best_tri_final = -1
    best_tri = -1
    best_cos_final = -1.0
    best_cos = -1.0
    for tri in tri_list:
        if tri == curr_tri:
            continue

        # Get the vertices of the triangle
        vertices = [0, 0]
        idx = 0
        for v_t in mesh.triangles[tri]:
            v = int(v_t.item())
            if v != vertex_id:
                vertices[idx] = v
                idx += 1

        # Compute the normal of the triangle
        p0 = mesh.positions[vertex_id]
        p1 = mesh.positions[vertices[0]]
        p2 = mesh.positions[vertices[1]]
        e1 = normalize(p1 - p0)
        e2 = normalize(p2 - p0)

        n = mesh.triangle_normals[best_tri]
        dir_proj = normalize(project_vec(dir_3d, n))
        cos = dot(dir_proj, dir_3d)

        if (
            dot(cross(dir_proj, e1), cross(dir_proj, e2)) <= 0
            and np.arccos(dot(dir_proj, e1)) + np.arccos(dot(dir_proj, e2))
            < np.pi - eps
        ):
            # The direction is inside the triangle
            if cos > best_cos_final:
                best_cos_final = cos
                best_tri_final = tri
        elif cos > best_cos:
            # The direction is outside the triangle
            best_cos = cos
            best_tri = tri

    is_near_hole = False
    if best_tri_final == -1:
        # No valid triangle found, use next edge close to hole
        edges = [[0, 1], [1, 2], [2, 0]]
        for tri in tri_list:
            if tri == curr_tri:
                continue

            for k, (v1, v2) in enumerate(edges):
                v1 = int(mesh.triangles[tri][v1].item())
                v2 = int(mesh.triangles[tri][v2].item())
                if v1 != vertex_id and v2 != vertex_id:
                    continue

                if mesh.adjacencies[tri][k].item() == -1:
                    best_tri_final = tri
                    is_near_hole = True
                    break

            if is_near_hole:
                break

        # No valid triangle found, select best triangle instead
        if best_tri_final == -1:
            best_tri_final = best_tri

    n = mesh.triangle_normals[best_tri_final]
    dir_proj = normalize(project_vec(dir_3d, n))
    return dir_proj, best_tri_final, n, is_near_hole


def compute_parallel_transport_vertex(
    mesh: Mesh,
    curr_tri: int,
    vertex_id: int,
    dir_3d: Tensor,
    curr_normal: Tensor,
    eps: float,
) -> Tuple[Tensor, int, Tensor]:
    """
    Compute the parallel transport of a vector at a vertex.
    """
    len_connected_triangles = mesh.v2t[vertex_id, 0]
    total_angle = 0.0
    for tri_id in mesh.v2t[vertex_id, 1 : len_connected_triangles + 1]:
        vertices = [0, 0]
        idx = 0
        for v_t in mesh.triangles[tri_id]:
            v = int(v_t.item())
            if v != vertex_id:
                vertices[idx] = v
                idx += 1

        n = mesh.triangle_normals[tri_id]

        angle = signed_angle(
            mesh.positions[vertices[0]] - mesh.positions[vertex_id],
            mesh.positions[vertices[1]] - mesh.positions[vertex_id],
            n,
        )
        total_angle += abs(angle)

    half_angle = total_angle / 2

    # get initial angle
    dir_3d = -dir_3d
    v1 = -1
    for v in mesh.triangles[curr_tri]:
        if v.item() != vertex_id:
            v1 = int(v.item())
            break
    v2 = (set(mesh.triangles[curr_tri].tolist()) - {v1, vertex_id}).pop()
    p0 = mesh.positions[vertex_id]
    p1 = mesh.positions[v1]
    p2 = mesh.positions[v2]
    n = triangle_normal(p0, p1, p2)
    if dot(n, curr_normal) > 0:
        normal_sign = 1
    else:
        normal_sign = -1
    angle = signed_angle(dir_3d, p1 - p0, n)

    if angle > half_angle:
        angle = signed_angle(p2 - p0, dir_3d, n)
        v1, v2 = v2, v1

    while angle < half_angle:
        # Get next triangle
        v2 = (set(mesh.triangles[curr_tri].tolist()) - {v1, vertex_id}).pop()

        p0 = mesh.positions[vertex_id]
        p1 = mesh.positions[v1]
        p2 = mesh.positions[v2]
        n = triangle_normal(p0, p1, p2)

        tri_angle = abs(
            signed_angle(
                mesh.positions[v1] - mesh.positions[vertex_id],
                mesh.positions[v2] - mesh.positions[vertex_id],
                n,
            )
        )

        if angle + tri_angle >= half_angle - eps:
            angle_diff = half_angle - angle
            axis = n
            edge = normalize(p1 - p0)
            new_dir = rotate_vector(edge, angle_diff, axis)
            return new_dir, curr_tri, normal_sign * n

        local_edge_idx = -1
        for idx, v_t in enumerate(mesh.triangles[curr_tri]):
            v = int(v_t.item())
            if v != v2 and v != vertex_id:
                local_edge_idx = (idx + 1) % 3
                break

        next_tri = int(mesh.adjacencies[curr_tri][local_edge_idx].item())
        if next_tri == -1:
            angle_diff = half_angle - angle
            axis = n
            edge = normalize(mesh.positions[v1] - mesh.positions[vertex_id])
            new_dir = rotate_vector(edge, angle_diff, axis)
            return new_dir, curr_tri, normal_sign * n

        curr_tri = next_tri
        v1 = v2
        angle += tri_angle


def bary_to_uv(bary: Tensor) -> Tensor:
    """Convert barycentric coordinates to UV coordinates."""
    return bary[1:3]


def get_next_bary(
    mesh: Mesh, curr_tri: int, next_tri: int, curr_bary: Tensor, common: List[int]
) -> Tensor:
    """
    Compute the barycentric coordinates in the next triangle based on the current
    barycentric coordinates.
    """
    if curr_tri == next_tri:
        return curr_bary

    next_bary = torch.zeros(3)

    for i in range(3):
        v = mesh.triangles[curr_tri, i].item()
        if v == common[0] or v == common[1]:
            # find index of v in next_triangle
            for j in range(3):
                if mesh.triangles[next_tri, j].item() == v:
                    next_bary[j] = curr_bary[i]
                    break

    return next_bary


def rotate_vector(v: Tensor, angle: float, n: Tensor) -> Tensor:
    """
    Rotates vector v by angle, according to normal vector n
    """
    return (
        v * math.cos(angle)
        + cross(n, v) * math.sin(angle)
        + n * dot(n, v) * (1 - math.cos(angle))
    )


def project_vertex_dir(mesh, vertex_id: int, dir: Tensor) -> Tuple[int, Tensor]:
    """
    select edge with minimal projection to vertex tangent plane, this will be the basis
    calculate angle between the basis and dir
    get local_angle = (angle / 2*pi) * vertex_angle
    rotate edge around the mesh by local_angle
    """
    v_normal = mesh.vertex_normals[vertex_id]
    len_connected_triangles = mesh.v2t[vertex_id, 0]
    total_angle = 0.0
    best_proj, best_tri, best_v = 100000, -1, -1
    v_pos = mesh.positions[vertex_id]

    for tri_id in mesh.v2t[vertex_id, 1 : len_connected_triangles + 1]:
        for k in range(3):
            if int(mesh.triangles[tri_id][k].item()) == vertex_id:
                v1 = int(mesh.triangles[tri_id][(k + 1) % 3].item())
                v2 = int(mesh.triangles[tri_id][(k + 2) % 3].item())
                break

        e1 = mesh.positions[v1] - v_pos
        e2 = mesh.positions[v2] - v_pos
        e1 = normalize(e1)
        e2 = normalize(e2)

        total_angle += math.acos(dot(e1, e2))

        proj_loss = abs(dot(e1, v_normal))
        if proj_loss < best_proj:
            best_proj = proj_loss
            best_tri = tri_id
            best_v = v1

        proj_loss = abs(dot(e2, v_normal))
        if proj_loss < best_proj:
            best_proj = proj_loss
            best_tri = tri_id
            best_v = v2

    mesh_edge = mesh.positions[best_v] - mesh.positions[vertex_id]
    tangent_edge = mesh_edge - dot(mesh_edge, v_normal) * v_normal

    tangent_angle = signed_angle(tangent_edge, dir, v_normal) % (2 * torch.pi)
    mesh_angle = (tangent_angle / (2 * torch.pi)) * total_angle

    angle = 0
    tri = best_tri
    v1 = best_v
    for k in range(3):
        v_k = int(mesh.triangles[tri][k].item())
        if v_k != vertex_id and v_k != v1:
            v2 = v_k
            break

    while angle < mesh_angle:
        e1 = mesh.positions[v1] - mesh.positions[vertex_id]
        e2 = mesh.positions[v2] - mesh.positions[vertex_id]
        tri_angle = abs(signed_angle(e1, e2, mesh.triangle_normals[tri]))

        if tri_angle + angle >= mesh_angle:
            e1 = normalize(e1)
            axis = normalize(cross(e1, e2))
            new_dir = rotate_vector(e1, mesh_angle - angle, axis)
            return tri, new_dir

        angle += tri_angle
        for k in range(3):
            if mesh.triangles[tri][k].item() == v1:
                next_tri = mesh.adjacencies[tri, (k + 1) % 3]

        if next_tri == -1:
            # No adjacent triangle, stop here
            return tri, normalize(mesh.positions[v2] - mesh.positions[vertex_id])

        tri = next_tri
        v1 = v2
        for k in range(3):
            v_k = int(mesh.triangles[tri][k].item())
            if v_k != vertex_id and v_k != v1:
                v2 = v_k
                break

    return tri, normalize(mesh.positions[v2] - mesh.positions[vertex_id])


@torch.no_grad()
def straightest_geodesic(
    mesh: Mesh,
    start: MeshPoint,
    start_dir: Tensor,
    max_steps=2000,
    debug_path=False,
    print_warnings=False,
    eps: float = -1.0,
    avoid_holes: bool = True,
) -> GeodesicResult:
    """
    Compute the straightest geodesic path on a mesh.
    """
    if eps < 0:
        if start_dir.dtype == torch.float32:
            eps = 1e-4
        else:
            eps = 1e-7

    curr_point: MeshPoint = start
    curr_tri: int = start.face

    # Project the direction onto the triangle plane
    path_len: float = 0.0
    curr_bary: Tensor = torch.tensor(
        [1 - start.uv[0] - start.uv[1], start.uv[0], start.uv[1]]
    )
    is_vert_bary, vert_idx = bary_is_vert(curr_bary, eps)
    v_id = int(mesh.triangles[start.face][vert_idx].item())
    if is_vert_bary:
        current_normal = mesh.vertex_normals[v_id]
        dir = project_vec(start_dir, current_normal)
        path_len = length(dir)
        curr_tri, dir = project_vertex_dir(mesh, v_id, dir)
        uv_coord = torch.zeros(2)
        for k in range(1, 3):
            if mesh.triangles[curr_tri][k].item() == v_id:
                uv_coord[k - 1] = 1.0
        curr_point = MeshPoint(curr_tri, uv_coord)
        curr_bary = curr_point.get_barycentric_coords()
        current_normal = mesh.triangle_normals[curr_tri]
    else:
        current_normal = mesh.triangle_normals[curr_point.face]
        dir = project_vec(start_dir, current_normal)
        path_len = length(dir)

    len_path: float = 0.0
    dir = normalize(dir)

    if debug_path:
        path: List[List[float]] = [start.interpolate(mesh).tolist()]
        dirs: List[List[float]] = [dir.tolist()]
        normals: List[List[float]] = [current_normal.tolist()]

    next_bary: Tensor = torch.zeros(3)
    curr_pos: Tensor = curr_point.interpolate(mesh)
    next_pos: Tensor = torch.zeros(3)
    next_tri: int = -1
    is_near_hole: bool = False
    vertex_crossings: int = 0

    step = 0

    if avoid_holes:
        is_edge_bary, edge_idx = bary_is_edge(curr_bary, eps)
        is_vert_bary, vert_idx = bary_is_vert(curr_bary, eps)
        is_near_hole_flag = False
        if not is_edge_bary and not is_vert_bary:
            # Point is in the interior of the triangle, no need to check for holes
            is_near_hole_flag = False
        elif is_vert_bary:
            for k in range(3):
                if k != vert_idx + 1:
                    if mesh.adjacencies[curr_tri][k].item() == -1:
                        is_near_hole_flag = True
                        break
        elif is_edge_bary:
            # Check if the edge is adjacent to a hole
            edge_local_idx = edge_idx - 1
            if mesh.adjacencies[curr_tri][edge_local_idx].item() == -1:
                is_near_hole_flag = True

        is_near_hole = is_near_hole_flag

    while len_path < path_len - eps:
        step += 1
        if step >= max_steps:
            if print_warnings:
                print(
                    f"Warning: Too many steps ({step}) in straightest_geodesic, "
                    f"stopping."
                )
                print(
                    f"UV: {start.uv.tolist()},triangle: {start.face}, "
                    f"direction: {start_dir.tolist()}"
                )
            break

        if length(dir) < eps:
            # Direction is perpendicular to the triangle, cannot proceed
            break

        # Trace the ray in the current triangle
        if is_near_hole:
            next_pos, next_bary = trace_around_hole(
                mesh, dir, curr_bary, curr_pos, curr_tri, path_len - len_path, eps
            )
        else:
            next_pos, next_bary = trace_in_triangles(
                mesh, dir, curr_bary, curr_pos, curr_tri, path_len - len_path, eps
            )

        # Check if the point is on an edge or vertex
        is_edge_bary, edge_idx = bary_is_edge(next_bary, eps)
        is_vert_bary, vert_idx = bary_is_vert(next_bary, eps)

        # Update the path
        len_path += length(next_pos - curr_pos)

        if DEBUG_MODE:
            print(
                f"Current triangle: {curr_tri}, Next position: {next_pos}, "
                f"Barycenter: {next_bary}, dir: {dir}"
            )

        # Determine the next triangle
        if is_vert_bary:
            vertex_crossings += 1
            # Point is on a vertex
            v_idx = int(mesh.triangles[curr_tri][vert_idx].item())

            # Transport the direction to the next triangle
            hole_parallel_transport = False
            if is_near_hole:
                hole_parallel_transport = True
                dir, next_tri, current_normal, is_near_hole = (
                    compute_parallel_transport_vertex_hole(
                        mesh, curr_tri, v_idx, dir, current_normal, eps
                    )
                )
            else:
                dir, next_tri, current_normal = compute_parallel_transport_vertex(
                    mesh, curr_tri, v_idx, dir, current_normal, eps
                )

            # Compute barycentric coordinates in the adjacent triangle
            p0_adj = mesh.positions[mesh.triangles[next_tri][0]]
            p1_adj = mesh.positions[mesh.triangles[next_tri][1]]
            p2_adj = mesh.positions[mesh.triangles[next_tri][2]]
            next_bary = torch.zeros(3)
            dist = [
                (p0_adj - next_pos) @ (p0_adj - next_pos),
                (p1_adj - next_pos) @ (p1_adj - next_pos),
                (p2_adj - next_pos) @ (p2_adj - next_pos),
            ]
            vert_idx = np.argmin(dist)
            next_bary[vert_idx] = 1

            # Check if adjacent to a hole
            if avoid_holes and not hole_parallel_transport:
                for k in range(3):
                    if k != vert_idx + 1 and mesh.adjacencies[next_tri][k].item() == -1:
                        is_near_hole = True
                        break

            # Update current state
            curr_tri = next_tri
            curr_pos = next_pos.clone()
            curr_bary = next_bary.clone()

            # Create a new mesh point for current position
            curr_point = MeshPoint(curr_tri, bary_to_uv(curr_bary))

        elif is_edge_bary:
            # Point is on an edge
            # Find the adjacent triangle across this edge
            edge_local_idx = edge_idx
            adj_tri = int(mesh.adjacencies[curr_tri][edge_local_idx].item())

            if adj_tri == -1:
                is_near_hole = True
                if not avoid_holes:
                    adj_tri = curr_tri
                    curr_point = MeshPoint(adj_tri, bary_to_uv(next_bary))
                    if debug_path:
                        # Append the current position to the path
                        path.append(next_pos.tolist())
                        dirs.append(dir.tolist())
                        normals.append(current_normal.tolist())
                    break
                else:
                    adj_tri = curr_tri

            else:
                is_near_hole = False
                # Get the common edge vertices
                e, _, _ = common_edge(mesh.triangles, curr_tri, adj_tri)

                if len(e) == 0:
                    # No common edge found
                    curr_point = MeshPoint(adj_tri, bary_to_uv(next_bary))
                    if debug_path:
                        # Append the current position to the path
                        path.append(next_pos.tolist())
                        dirs.append(dir.tolist())
                        normals.append(current_normal.tolist())
                    break

                next_bary = get_next_bary(mesh, curr_tri, adj_tri, next_bary, e)

                # Transport the direction to the adjacent triangle
                dir, current_normal = compute_parallel_transport_edge(
                    mesh, curr_tri, adj_tri, dir, current_normal
                )

            # Update current state
            curr_tri = adj_tri
            curr_pos = next_pos.clone()
            curr_bary = next_bary.clone()

            # Create a new mesh point for current position
            curr_point = MeshPoint(curr_tri, bary_to_uv(curr_bary))

        else:
            # Point is in the interior of the triangle
            # Update current state
            curr_tri = curr_tri  # No change in triangle
            curr_pos = next_pos.clone()
            curr_bary = next_bary.clone()

            # Create a new mesh point for current position
            curr_point = MeshPoint(curr_tri, bary_to_uv(curr_bary))

        if debug_path:
            # Append the current position to the path
            path.append(curr_pos.tolist())
            dirs.append(dir.tolist())
            normals.append(current_normal.tolist())

    if DEBUG_MODE:
        print(
            f"Current triangle: {curr_tri}, Next position: {curr_pos}, "
            f"Barycenter: {curr_bary}, dir: {dir}"
        )

    if debug_path:
        return curr_point, dir, current_normal, (path, dirs, normals, vertex_crossings)
    else:
        return curr_point, dir, current_normal, None
