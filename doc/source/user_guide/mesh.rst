Meshes in DiGeo
===============

The DiGeo library represents meshes using PyTorch tensors to enable GPU acceleration and differentiable operations. Under the hood, it uses the ``trimesh`` library to parse and load 3D models from files.

Loading Meshes
--------------

1. **From a file** with :func:`digeo.load_mesh_from_file`. Uses ``trimesh`` internally to read standard 3D formats (like ``.obj``).
2. **From an existing trimesh.Trimesh object** with :func:`digeo.load_mesh_from_trimesh`. Useful if you have already loaded or generated a mesh using ``trimesh``.

.. code-block:: python

    from digeo import load_mesh_from_file, load_mesh_from_trimesh
    import trimesh
    import torch

    # Load directly from a file
    mesh = load_mesh_from_file("path/to/mesh.obj", device="cuda", dtype=torch.float32)

    # Load from an existing trimesh object
    trimesh_obj = trimesh.creation.icosphere()
    mesh = load_mesh_from_trimesh(trimesh_obj, device="cuda", dtype=torch.float32)

Batching Meshes
---------------

To process multiple meshes simultaneously, you can group them into a ``MeshBatch`` object. A ``MeshBatch`` inherits from ``Mesh`` and concatenates the vertices and faces of multiple meshes into a single large mesh with disconnected components.

When working with batched meshes, any points on the mesh (represented by ``MeshPointBatch``) must also be batched to align with the shifted face indices of the ``MeshBatch``.

.. code-block:: python

    from digeo import MeshBatch
    from digeo.ops import uniform_sampling

    # Load individual meshes
    mesh1 = load_mesh_from_file("mesh1.obj", device="cuda")
    mesh2 = load_mesh_from_file("mesh2.obj", device="cuda")

    # Create a MeshBatch
    mesh_batch = MeshBatch([mesh1, mesh2])

    # Sample points on individual meshes
    points1 = uniform_sampling(mesh1, num_points=100)
    points2 = uniform_sampling(mesh2, num_points=100)

    # Batch the points using the MeshBatch
    batched_points = mesh_batch.batch_points([points1, points2])

    # Process the batched points with the MeshBatch
    # For example, trace geodesics on the batched mesh with the batched points
    new_batched_points, _ = trace_geodesics(mesh_batch, batched_points, directions)

    # Unbatch the points back to individual meshes
    unbatched_points = mesh_batch.unbatch_points(new_batched_points)
    new_points1, new_points2 = unbatched_points[0], unbatched_points[1]


MeshPoints
----------

Points on the mesh are represented by the ``MeshPoint`` class, which encodes a point as a face index and barycentric coordinates within that face. This representation allows for efficient interpolation of face attributes (like normals) at the point location.

However, it is prefered to use directly the ``MeshPointBatch`` class, which is a batch of ``MeshPoint`` objects, as it is more efficient to process points in batches, especially when tracing geodesics or computing gradients.

To create a ``MeshPointBatch``, you can use the sampling functions provided in ``src/digeo/ops/sampling.py``, such as ``uniform_sampling``. It is also possible to create a ``MeshPointBatch`` from explicit face indices and uv coordinates.

It is also possible to convert these meshpoints to 3d coordinates using :func:`MeshPointBatch.interpolate`.

.. code-block:: python

    from digeo import MeshPointBatch, uniform_sampling

    # Sample points on the mesh
    points = uniform_sampling(mesh, num_points=100)

    # Alternatively, create a MeshPointBatch from explicit face indices and uv coordinates
    face_indices = torch.tensor([0, 1, 2])
    bary_coords = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]])
    # We convert the bary_coords to UV coords
    uv_coords = bary_coords[:, 1:]
    points = MeshPointBatch(face_indices, uv_coords)

    # Interpolate the 3D coordinates of the points
    point_3d = points.interpolate(mesh)