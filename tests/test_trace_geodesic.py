import os
import potpourri3d as pp3d
import numpy as np
import pytest
import torch
from torch import Tensor

from digeo import (
    load_mesh_from_file,
    Mesh,
    MeshPointBatch,
)
from digeo.ops import (
    trace_geodesics,
    uniform_sampling,
)
from digeo.utils import length


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def trace_geodesic_pp3d(device, use_python):
    set_seed(12)

    mesh_path = os.path.join(DATA_DIR, "bunny.obj")
    assert os.path.exists(mesh_path), "Mesh file does not exist."
    mesh = load_mesh_from_file(mesh_path, device=device)

    start_points = uniform_sampling(mesh, 512).to(device)
    start_directions = 0.3 * (
        torch.rand((512, 3), dtype=torch.float32).to(device) - 0.5
    )

    meshpoints, _ = trace_geodesics(
        mesh, start_points, start_directions, use_python=use_python, gradient="none"
    )

    V, F = pp3d.read_mesh(mesh_path)
    tracer = pp3d.GeodesicTracer(V, F)

    errors = []
    for k, (start_point, start_direction) in enumerate(
        zip(start_points, start_directions)
    ):
        pp3d_trace = tracer.trace_geodesic_from_face(
            start_point.face,
            start_point.get_barycentric_coords().detach().cpu().numpy(),
            start_direction.detach().cpu().numpy(),
        )
        pp3d_end_point = torch.tensor(
            pp3d_trace[-1], dtype=torch.float32, device=device
        )
        end_point = meshpoints[k].interpolate(mesh)

        error = length(pp3d_end_point - end_point)
        errors.append(error)
        assert error < 10e-5, f"Error too high for point {k}: {error}"

    assert np.mean(errors) < 10e-5, f"Average error is too high: {np.mean(errors)}"


def sphere_exp_map(mesh: Mesh, x: MeshPointBatch, v: Tensor) -> Tensor:
    """
    Exponential map on the sphere.
    """
    x_point = x.interpolate(mesh)
    v = v - torch.sum(v * x_point, dim=-1, keepdim=True) * x_point
    norm_v = torch.norm(v, dim=-1, keepdim=True)
    return x_point * torch.cos(norm_v) + (v / norm_v) * torch.sin(norm_v)


def trace_geodesic_sphere(method, min_grad_x_sim, min_grad_v_sim):
    set_seed(12)
    device = "cpu"

    mesh_path = os.path.join(DATA_DIR, "sphere.obj")
    assert os.path.exists(mesh_path), "Mesh file does not exist."

    mesh = load_mesh_from_file(mesh_path, dtype=torch.float32, device=device)
    start_points = uniform_sampling(mesh, 2048).to(device=device, dtype=torch.float32)
    target = torch.tensor([1, 0, 0], dtype=torch.float32, device=device)

    start_dirs = (
        5
        * 2
        * (torch.rand([len(start_points), 3], dtype=torch.float32, device=device) - 0.5)
    )
    start_normals = mesh.triangle_normals[start_points.faces]
    start_dirs = (
        start_dirs
        - torch.sum(start_dirs * start_normals, dim=-1, keepdim=True) * start_normals
    )

    x0 = start_points.clone().detach()
    x0.uvs = x0.uvs.requires_grad_(True)
    v0 = start_dirs.clone().detach().requires_grad_(True)
    x1_gt = sphere_exp_map(mesh, x0, v0)
    loss = torch.mean(torch.linalg.norm(x1_gt - target, dim=-1))
    loss.backward()
    grad_x_gt = x0.uvs.grad.clone()
    grad_v_gt = v0.grad.clone()

    x0 = start_points.clone().detach()
    x0.uvs = x0.uvs.requires_grad_(True)
    v0 = start_dirs.clone().detach().requires_grad_(True)
    x1, _ = trace_geodesics(mesh, x0, v0, gradient=method, max_steps=10000)
    x1_torch = x1.interpolate(mesh)
    loss = torch.mean(torch.linalg.norm(x1_torch - target, dim=-1))
    loss.backward()
    grad_x_torch = x0.uvs.grad.clone()
    grad_v_torch = v0.grad.clone()
    cos_sim_x = torch.nn.functional.cosine_similarity(grad_x_torch, grad_x_gt, dim=-1)
    cos_sim_v = torch.nn.functional.cosine_similarity(grad_v_torch, grad_v_gt, dim=-1)

    errors = torch.linalg.norm(x1_gt - x1_torch, dim=1)

    assert torch.mean(errors) < 5 * 10e-4, (
        f"Average error is too high: {torch.mean(errors).item()}"
    )
    assert torch.median(cos_sim_x) > min_grad_x_sim, (
        f"Median cosine similarity for x gradients too low: "
        f"{torch.median(cos_sim_x).item()}"
    )
    assert torch.median(cos_sim_v) > min_grad_v_sim, (
        f"Median cosine similarity for v gradients too low: "
        f"{torch.median(cos_sim_v).item()}"
    )


def trace_geodesic_plane(method, min_grad_x_sim, min_grad_v_sim):
    set_seed(12)
    device = "cpu"

    mesh_path = os.path.join(DATA_DIR, "plane.obj")
    assert os.path.exists(mesh_path), "Mesh file does not exist."

    mesh = load_mesh_from_file(mesh_path, dtype=torch.float32, device=device)
    start_points = MeshPointBatch(
        faces=5418 * torch.ones(2048, dtype=torch.int32, device=device),
        uvs=0.3 * torch.ones((2048, 2), dtype=torch.float32, device=device),
    )
    target = torch.tensor([1, 0, 0], dtype=torch.float32, device=device)

    start_dirs = 2 * (
        torch.rand([len(start_points), 3], dtype=torch.float32, device=device) - 0.5
    )
    start_normals = mesh.triangle_normals[start_points.faces]
    start_dirs = (
        start_dirs
        - torch.sum(start_dirs * start_normals, dim=-1, keepdim=True) * start_normals
    )

    x0 = start_points.clone().detach()
    x0.uvs = x0.uvs.requires_grad_(True)
    v0 = start_dirs.clone().detach().requires_grad_(True)
    x1_gt = x0.interpolate(mesh) + v0
    loss = torch.mean(torch.linalg.norm(x1_gt - target, dim=-1))
    loss.backward()
    grad_x_gt = x0.uvs.grad.clone()
    grad_v_gt = v0.grad.clone()

    x0 = start_points.clone().detach()
    x0.uvs = x0.uvs.requires_grad_(True)
    v0 = start_dirs.clone().detach().requires_grad_(True)
    x1, _ = trace_geodesics(mesh, x0, v0, gradient=method, max_steps=10000)
    x1_torch = x1.interpolate(mesh)
    loss = torch.mean(torch.linalg.norm(x1_torch - target, dim=-1))
    loss.backward()
    grad_x_torch = x0.uvs.grad.clone()
    grad_v_torch = v0.grad.clone()
    cos_sim_x = torch.nn.functional.cosine_similarity(grad_x_torch, grad_x_gt, dim=-1)
    cos_sim_v = torch.nn.functional.cosine_similarity(grad_v_torch, grad_v_gt, dim=-1)

    errors = torch.linalg.norm(x1_gt - x1_torch, dim=1)

    assert torch.mean(errors) < 5 * 10e-4, (
        f"Average error is too high: {torch.mean(errors).item()}"
    )
    assert torch.median(cos_sim_x) > min_grad_x_sim, (
        f"Median cosine similarity for x gradients too low: "
        f"{torch.median(cos_sim_x).item()}"
    )
    assert torch.median(cos_sim_v) > min_grad_v_sim, (
        f"Median cosine similarity for v gradients too low: "
        f"{torch.median(cos_sim_v).item()}"
    )


@pytest.mark.parametrize(
    ("device", "use_python"), [("cuda", False), ("cpu", True), ("cpu", False)]
)
def test_trace_geodesic_mesh(device, use_python):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    trace_geodesic_pp3d(device, use_python=use_python)


@pytest.mark.parametrize(
    ("method", "min_grad_x_sim", "min_grad_v_sim"),
    [
        ("gfd", 0.99, 0.99),
        ("ep", 0.80, 0.80),
        ("abfd", 0.99, 0.80),
    ],
)
def test_trace_geodesic_gradient_sphere(method, min_grad_x_sim, min_grad_v_sim):
    trace_geodesic_sphere(method, min_grad_x_sim, min_grad_v_sim)


@pytest.mark.parametrize(
    ("method", "min_grad_x_sim", "min_grad_v_sim"),
    [
        ("gfd", 0.99, 0.99),
        ("ep", 0.99, 0.99),
        ("abfd", 0.99, 0.99),
    ],
)
def test_trace_geodesic_gradient_plane(method, min_grad_x_sim, min_grad_v_sim):
    trace_geodesic_plane(method, min_grad_x_sim, min_grad_v_sim)
