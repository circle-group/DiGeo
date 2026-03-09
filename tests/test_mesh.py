import torch
import numpy as np

from digeo import Mesh, MeshBatch, MeshPoint, MeshPointBatch
from digeo.ops import uniform_sampling
from digeo.mesh_loader import create_tetrahedron, create_triangle


def test_mesh():
    # Create a tetrahedral mesh
    vertices = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.int32
    )

    mesh = Mesh(vertices=vertices, faces=faces)
    assert mesh.vertices.shape == (4, 3), "Mesh positions shape is incorrect."
    assert mesh.faces.shape == (4, 3), "Mesh triangles shape is incorrect."
    assert mesh.triangle_normals.shape == (4, 3), (
        "Mesh triangle normals shape is incorrect."
    )
    assert mesh.adjacencies.shape == (4, 3), "Mesh adjacencies shape is incorrect."
    assert mesh.v2t.shape == (4, 4), (
        "Mesh vertex to triangle mapping shape is incorrect."
    )


def test_mesh_batch_shape():
    # Create a batch of tetrahedral meshes
    num_meshes = 5
    meshes = [create_tetrahedron() for _ in range(num_meshes)]
    mesh_batch = MeshBatch(meshes)

    assert len(mesh_batch) == num_meshes, "Mesh batch length is incorrect."
    assert mesh_batch.vertex_idx.shape == (num_meshes + 1,), (
        "Mesh batch vertex index shape is incorrect."
    )
    assert mesh_batch.triangle_idx.shape == (num_meshes + 1,), (
        "Mesh batch triangle index shape is incorrect."
    )
    assert mesh_batch.vertices.shape == (num_meshes * 4, 3), (
        "Mesh batch vertices shape is incorrect."
    )
    assert mesh_batch.faces.shape == (num_meshes * 4, 3), (
        "Mesh batch faces shape is incorrect."
    )

    mesh_list = mesh_batch.unbatch()
    assert isinstance(mesh_list[0], Mesh), "Unbatched mesh is not of type Mesh."
    assert len(mesh_list) == num_meshes, "Unbatched mesh list length is incorrect."


def test_mesh_unbatch():
    num_meshes = 8
    meshes = [
        create_tetrahedron() if i % 2 == 0 else create_triangle()
        for i in range(num_meshes)
    ]
    mesh_batch = MeshBatch(meshes)
    mesh_list = mesh_batch.unbatch()
    for i in range(num_meshes):
        assert isinstance(mesh_list[i], Mesh), (
            f"Unbatched mesh {i} is not of type Mesh."
        )
        assert torch.allclose(mesh_list[i].vertices, meshes[i].vertices), (
            f"Unbatched mesh {i} positions do not match."
        )
        assert torch.allclose(mesh_list[i].faces, meshes[i].faces), (
            f"Unbatched mesh {i} triangles do not match."
        )
        assert torch.allclose(
            mesh_list[i].triangle_normals, meshes[i].triangle_normals
        ), f"Unbatched mesh {i} triangle normals do not match."
        assert torch.allclose(mesh_list[i].adjacencies, meshes[i].adjacencies), (
            f"Unbatched mesh {i} adjacencies do not match."
        )

        max_v2t = meshes[i].v2t.shape[1]
        assert torch.allclose(
            mesh_list[i].v2t[:, :max_v2t], meshes[i].v2t[:, :max_v2t]
        ), f"Unbatched mesh {i} vertex to triangle mapping does not match."
        assert torch.allclose(
            mesh_batch.v2t[:, 0], torch.cat([mesh.v2t[:, 0] for mesh in meshes], dim=0)
        ), f"Unbatched mesh {i} vertex to triangle mapping first column does not match."


def test_meshpoint_batch():
    meshes = [create_tetrahedron(), create_triangle(), create_tetrahedron()]
    meshpoints = [uniform_sampling(mesh, 10) for mesh in meshes]

    mesh_batch = MeshBatch(meshes)
    meshpoint_batch = mesh_batch.batch_points(meshpoints)

    points_batch = meshpoint_batch.interpolate(mesh_batch, return_batch=True)
    points_list = [x.interpolate(mesh) for mesh, x in zip(meshes, meshpoints)]

    for i in range(len(points_batch)):
        assert points_batch[i].shape == (10, 3), (
            f"Interpolated points shape for mesh {i} is incorrect."
        )
        assert points_list[i].shape == (10, 3), (
            f"Interpolated points shape for mesh {i} is incorrect."
        )
        assert torch.allclose(points_batch[i].detach(), points_list[i].detach()), (
            f"Interpolated points do not match for mesh {i}."
        )

    meshpoints_unbatched = mesh_batch.unbatch_points(meshpoint_batch)
    assert len(meshpoints_unbatched) == len(meshes), (
        "Unbatched mesh points length is incorrect."
    )
    for i in range(len(meshpoints_unbatched)):
        assert isinstance(meshpoints_unbatched[i], MeshPointBatch), (
            f"Unbatched mesh point {i} is not of type MeshPointBatch."
        )
        assert np.allclose(
            meshpoints_unbatched[i].uvs.detach(), meshpoints[i].uvs.detach()
        ), f"Unbatched mesh point {i} uv do not match."
        assert np.allclose(
            meshpoints_unbatched[i].faces.detach(), meshpoints[i].faces.detach()
        ), f"Unbatched mesh point {i} faces do not match."


def test_mesh_batch_sampling():
    num_meshes = 5
    meshes = [create_tetrahedron() for _ in range(num_meshes)]
    mesh_batch = MeshBatch(meshes)

    num_samples = 10
    meshpoints = uniform_sampling(mesh_batch, num_samples)
    assert len(meshpoints) == num_meshes * num_samples, (
        "Number of sampled mesh points is incorrect."
    )
    for i in range(len(meshpoints)):
        meshpoint = meshpoints[i]
        assert isinstance(meshpoint, MeshPoint), (
            "Sampled mesh point is not of type MeshPoint."
        )

        start_vertex = mesh_batch.vertex_idx[i // 10]
        end_vertex = mesh_batch.vertex_idx[i // 10 + 1]

        assert start_vertex <= meshpoint.face < end_vertex, (
            "Mesh point face index is out of bounds."
        )

    meshpoints_list = mesh_batch.unbatch_points(meshpoints)
    assert len(meshpoints_list) == num_meshes, (
        "Unbatched mesh points length is incorrect."
    )
    for meshpoint_batch in meshpoints_list:
        assert isinstance(meshpoint_batch, MeshPointBatch), (
            "Unbatched mesh point is not of type MeshPointBatch."
        )
        assert len(meshpoint_batch) == num_samples, (
            "Unbatched mesh point batch length is incorrect."
        )
