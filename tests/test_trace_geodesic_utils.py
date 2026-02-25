import torch
import numpy as np

from digeo.mesh_loader import create_triangle
from digeo import MeshPoint
from digeo.ops.geodesic_utils import (
    triangle_normal,
    tri_bary_coords,
    bary_is_edge,
    bary_is_vert,
    trace_in_triangles,
    common_edge,
    signed_angle
)

EPS = 1e-5


def test_triangle_normal_basic():
    # Define a triangle in the XY plane
    p0 = torch.tensor([0.0, 0.0, 0.0])
    p1 = torch.tensor([1.0, 0.0, 0.0])
    p2 = torch.tensor([0.0, 1.0, 0.0])

    expected_normal = torch.tensor([0.0, 0.0, 1.0])  # Normal to the XY plane

    result = triangle_normal(p0, p1, p2)
    assert np.allclose(result, expected_normal)


def test_triangle_normal_reverse_order():
    # Changing order of vertices should reverse the normal
    p0 = torch.tensor([0.0, 0.0, 0.0])
    p1 = torch.tensor([0.0, 1.0, 0.0])
    p2 = torch.tensor([1.0, 0.0, 0.0])

    expected_normal = torch.tensor([0.0, 0.0, -1.0])  # Reversed normal

    result = triangle_normal(p0, p1, p2)
    assert np.allclose(result, expected_normal)


def test_triangle_normal_degenerate():
    # Degenerate triangle (all points on a line or same point)
    p0 = p1 = p2 = torch.tensor([1.0, 1.0, 1.0])
    result = triangle_normal(p0, p1, p2)
    expected = torch.tensor([0.0, 0.0, 0.0])  # No defined normal
    assert np.allclose(result, expected)


def test_barycentric_vertex_points():
    p0 = torch.tensor([0.0, 0.0])
    p1 = torch.tensor([1.0, 0.0])
    p2 = torch.tensor([0.0, 1.0])

    assert np.allclose(tri_bary_coords(p0, p1, p2, p0), [1.0, 0.0, 0.0])
    assert np.allclose(tri_bary_coords(p0, p1, p2, p1), [0.0, 1.0, 0.0])
    assert np.allclose(tri_bary_coords(p0, p1, p2, p2), [0.0, 0.0, 1.0])


def test_barycentric_inside_triangle():
    p0 = torch.tensor([0.0, 0.0])
    p1 = torch.tensor([1.0, 0.0])
    p2 = torch.tensor([0.0, 1.0])
    p = torch.tensor([0.25, 0.25])  # Inside the triangle

    result = tri_bary_coords(p0, p1, p2, p)
    assert np.allclose(result.sum(), 1.0)
    assert all(0 <= val <= 1 for val in result)


def test_barycentric_on_edge():
    p0 = torch.tensor([0.0, 0.0])
    p1 = torch.tensor([1.0, 0.0])
    p2 = torch.tensor([0.0, 1.0])
    p = torch.tensor([0.5, 0.0])  # On edge between p0 and p1

    result = tri_bary_coords(p0, p1, p2, p)
    assert np.allclose(result.sum(), 1.0)
    assert np.isclose(result[2], 0.0)


def test_barycentric_degenerate_triangle():
    p0 = p1 = p2 = torch.tensor([1.0, 1.0])
    p = torch.tensor([1.0, 1.0])  # Same point

    result = tri_bary_coords(p0, p1, p2, p)  # Should fail

    assert np.allclose(result.sum(), 1.0)
    assert np.isclose(result[0], 1.0)


# Tests for bary_is_edge


def test_bary_edge_0_opposite_vertex_0():
    bary = torch.tensor([0.0, 0.5, 0.5])
    is_edge, edge_idx = bary_is_edge(bary, EPS)
    assert is_edge
    assert edge_idx == 1  # Edge opposite vertex 0


def test_bary_edge_1_opposite_vertex_1():
    bary = torch.tensor([0.5, 0.0, 0.5])
    is_edge, edge_idx = bary_is_edge(bary, EPS)
    assert is_edge
    assert edge_idx == 2  # Edge opposite vertex 1


def test_bary_edge_2_opposite_vertex_2():
    bary = torch.tensor([0.5, 0.5, 0.0])
    is_edge, edge_idx = bary_is_edge(bary, EPS)
    assert is_edge
    assert edge_idx == 0  # Edge opposite vertex 2


def test_bary_not_on_edge():
    bary = torch.tensor([0.2, 0.3, 0.5])
    is_edge, edge_idx = bary_is_edge(bary, EPS)
    assert not is_edge
    assert edge_idx == -1


# Tests for bary_is_vert


def test_bary_vertex_0():
    bary = torch.tensor([1.0, 0.0, 0.0])
    is_vert, vert_idx = bary_is_vert(bary, EPS)
    assert is_vert
    assert vert_idx == 0


def test_bary_vertex_1():
    bary = torch.tensor([0.0, 1.0, 0.0])
    is_vert, vert_idx = bary_is_vert(bary, EPS)
    assert is_vert
    assert vert_idx == 1


def test_bary_vertex_2():
    bary = torch.tensor([0.0, 0.0, 1.0])
    is_vert, vert_idx = bary_is_vert(bary, EPS)
    assert is_vert
    assert vert_idx == 2


def test_bary_not_on_vertex():
    bary = torch.tensor([0.2, 0.3, 0.5])
    is_vert, vert_idx = bary_is_vert(bary, EPS)
    assert not is_vert
    assert vert_idx == -1


# Tests for trace in triangle


def test_trace_in_triangles_1():
    mesh = create_triangle()
    dir_3d = torch.tensor([1, 0, 0], dtype=torch.float32)
    curr_point = MeshPoint(0, torch.tensor([0, 0], dtype=torch.float32))
    max_len = 2.0

    # Call the function to test
    next_pos, next_bary = trace_in_triangles(
        mesh,
        dir_3d,
        curr_point.get_barycentric_coords(),
        curr_point.interpolate(mesh),
        curr_point.face,
        max_len,
        EPS,
    )

    assert np.allclose(next_pos, [1, 0, 0])
    assert np.allclose(next_bary, [0, 1, 0])


def test_trace_in_triangles_2():
    mesh = create_triangle()
    dir_3d = torch.tensor([1, 0, 0], dtype=torch.float32)
    curr_point = MeshPoint(0, torch.tensor([0, 0], dtype=torch.float32))
    max_len = 0.5

    # Call the function to test
    next_pos, next_bary = trace_in_triangles(
        mesh,
        dir_3d,
        curr_point.get_barycentric_coords(),
        curr_point.interpolate(mesh),
        curr_point.face,
        max_len,
        EPS,
    )

    assert np.allclose(next_pos, [0.5, 0, 0])
    assert np.allclose(next_bary, [0.5, 0.5, 0])


def test_trace_in_triangles_3():
    mesh = create_triangle()
    dir_3d = torch.tensor([1, 1, 0], dtype=torch.float32)
    curr_point = MeshPoint(0, torch.tensor([0, 0], dtype=torch.float32))
    max_len = 2.0

    # Call the function to test
    next_pos, next_bary = trace_in_triangles(
        mesh,
        dir_3d,
        curr_point.get_barycentric_coords(),
        curr_point.interpolate(mesh),
        curr_point.face,
        max_len,
        EPS,
    )

    assert np.allclose(next_pos, [0.5, 0.5, 0])
    assert np.allclose(next_bary, [0, 0.5, 0.5])


# Tests for common_edge


def test_common_edge_found():
    triangles = torch.tensor(
        [
            [0, 1, 2],  # Triangle 1 vertices
            [1, 2, 3],  # Triangle 2 vertices
        ]
    )

    # Common edge between triangle 0 and 1: vertices 1 and 2
    tri1 = 0
    tri2 = 1
    common, diff1, diff2 = common_edge(triangles, tri1, tri2)

    # Expected common edge [1, 2] and remaining vertices
    expected_common = torch.tensor([1, 2])

    assert np.allclose(common, expected_common)
    assert diff1 == 0
    assert diff2 == 3


def test_no_common_edge():
    triangles = torch.tensor(
        [
            [0, 1, 2],  # Triangle 1 vertices
            [3, 4, 5],  # Triangle 2 vertices
        ]
    )

    # No common edge between triangle 0 and 1
    tri1 = 0
    tri2 = 1
    common, diff1, diff2 = common_edge(triangles, tri1, tri2)

    # Expected result: no common edge
    assert len(common) == 0
    assert diff1 is None
    assert diff2 is None


def test_single_common_vertex():
    triangles = torch.tensor(
        [
            [0, 1, 2],  # Triangle 1 vertices
            [4, 1, 3],  # Triangle 2 vertices (shared vertex 1 and 2, but no full edge)
        ]
    )

    # Single common vertex between triangle 0 and 1: vertex 1
    tri1 = 0
    tri2 = 1
    common, diff1, diff2 = common_edge(triangles, tri1, tri2)

    # Expected result: no common edge
    assert len(common) == 0
    assert diff1 is None
    assert diff2 is None


# Tests for signed_angle


def test_signed_angle_zero():
    """
    Test that the signed angle is 0 when A and B are the same.
    """
    A = torch.tensor([1.0, 0.0, 0.0])
    B = torch.tensor([1.0, 0.0, 0.0])
    N = torch.tensor([0.0, 0.0, 1.0])
    angle = signed_angle(A, B, N)
    assert np.isclose(angle, 0.0)


def test_signed_angle_positive_90_degrees():
    """
    Test that the signed angle is +90 degrees (π/2 radians) when B is rotated CCW
    from A.
    """
    A = torch.tensor([1.0, 0.0, 0.0])
    B = torch.tensor([0.0, 1.0, 0.0])
    N = torch.tensor([0.0, 0.0, 1.0])
    angle = signed_angle(A, B, N)
    assert np.isclose(angle, torch.pi / 2)


def test_signed_angle_negative_90_degrees():
    """
    Test that the signed angle is -90 degrees (-π/2 radians) when B is rotated CW
    from A.
    """
    A = torch.tensor([1.0, 0.0, 0.0])
    B = torch.tensor([0.0, -1.0, 0.0])
    N = torch.tensor([0.0, 0.0, 1.0])
    angle = signed_angle(A, B, N)
    assert np.isclose(angle, -torch.pi / 2)


def test_signed_angle_180_degrees():
    """
    Test that the signed angle is ±π when A and B are opposite directions.
    """
    A = torch.tensor([1.0, 0.0, 0.0])
    B = torch.tensor([-1.0, 0.0, 0.0])
    N = torch.tensor([0.0, 0.0, 1.0])
    angle = signed_angle(A, B, N)
    assert np.isclose(abs(angle), torch.pi)


def test_signed_angle_off_plane_projection():
    """
    Test that A and B are properly projected to the plane defined by N before
    angle computation.
    """
    A = torch.tensor([1.0, 0.0, 1.0])  # Not in plane
    B = torch.tensor([0.0, 1.0, 1.0])  # Not in plane
    N = torch.tensor([0.0, 0.0, 1.0])  # XY plane
    angle = signed_angle(A, B, N)
    assert np.isclose(angle, torch.pi / 2)
