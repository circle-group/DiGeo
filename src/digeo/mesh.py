import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Union, Optional, Iterable, List, overload
from numpy.typing import NDArray
import trimesh


def get_device_dtype(*args, **kwargs) -> Tuple[Optional[str], Optional[torch.dtype]]:
    device: Optional[Union[str, torch.device]] = None
    dtype: Optional[torch.dtype] = None

    # Handle different overloads
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, (torch.device, str)):
            device = str(arg)
        elif isinstance(arg, torch.dtype):
            dtype = arg
        elif hasattr(arg, "device") and hasattr(arg, "dtype"):  # tensor-like
            device = str(arg.device)
            dtype = arg.dtype
        else:
            raise TypeError(f"Unsupported argument type: {type(arg)}")
    elif len(args) == 2:
        device, dtype = args
        if isinstance(device, torch.device):
            device = str(device)
        if not isinstance(dtype, torch.dtype):
            raise TypeError(
                f"Expected torch.dtype for second argument, got {type(dtype)}"
            )

    # Keyword arguments override args
    device = kwargs.get("device", device)
    dtype = kwargs.get("dtype", dtype)

    if device is None and dtype is None:
        raise ValueError("At least one of device or dtype must be specified.")

    if isinstance(device, torch.device):
        device = str(device)

    return device, dtype


class Mesh:
    def __init__(
        self,
        vertices: Union[NDArray, Tensor],
        faces: Union[NDArray, Tensor],
        adjacencies: Optional[Union[NDArray, Tensor]] = None,
        triangle_normals: Optional[Union[NDArray, Tensor]] = None,
        v2t: Optional[Union[NDArray, Tensor]] = None,
        vertex_normals: Optional[Union[NDArray, Tensor]] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize a Mesh object with vertices, faces, and optional precomputed data.

        Parameters
        ----------
        vertices: NDArray | Tensor
            (V, 3) array of vertex positions.
        faces: NDArray | Tensor
            (F, 3) array of triangle vertex indices.
        adjacencies: Optional[NDArray | Tensor]
            (F, 3) stores the adjacent triangle indices for each triangle edge.
        triangle_normals: Optional[NDArray | Tensor]
            (F, 3) array of triangle normal vectors.
        v2t: Optional[NDArray | Tensor]
            (V, max_tris_per_vertex + 1) array mapping vertices to incident triangles.
            The first column stores the number of incident triangles, followed by the
            triangle indices.
        vertex_normals: Optional[NDArray | Tensor]
            (V, 3) array of vertex normal vectors.
        device: str | torch.device
            The device to store the mesh on.
        dtype: torch.dtype
            The data type for the vertex positions and normals.
        """

        self.dtype: torch.dtype = dtype
        self.vertices = torch.as_tensor(
            vertices, dtype=dtype, device=device
        ).requires_grad_(False)
        self.faces = torch.as_tensor(
            faces, dtype=torch.int32, device=device
        ).requires_grad_(False)

        if vertex_normals is None:
            mesh_trimesh = trimesh.Trimesh(
                vertices=self.vertices.cpu().numpy(),
                faces=self.faces.cpu().numpy(),
            )
            vertex_normals = mesh_trimesh.vertex_normals

            if triangle_normals is None:
                triangle_normals = mesh_trimesh.face_normals

        self.adjacencies = torch.as_tensor(
            adjacencies if adjacencies is not None else self._compute_adjacencies(),
            dtype=torch.int32,
            device=device,
        ).requires_grad_(False)

        self.triangle_normals = torch.as_tensor(
            triangle_normals, dtype=dtype, device=device
        ).requires_grad_(False)

        self.v2t = torch.as_tensor(
            v2t if v2t is not None else self._compute_vertex_to_triangle_map(),
            dtype=torch.int32,
            device=device,
        ).requires_grad_(False)

        self.vertex_normals = torch.as_tensor(
            vertex_normals, dtype=dtype, device=device
        ).requires_grad_(False)

        self.device: torch.device = self.vertices.device

    @overload
    def to(self, device: Union[str, torch.device]) -> "Mesh": ...

    @overload
    def to(self, dtype: torch.dtype) -> "Mesh": ...

    @overload
    def to(self, device: Union[str, torch.device], dtype: torch.dtype) -> "Mesh": ...

    @overload
    def to(self, tensor: "Tensor") -> "Mesh": ...

    @overload
    def to(
        self,
        *,
        device: Optional[Union[str, torch.device]] = ...,
        dtype: Optional[torch.dtype] = ...,
    ) -> "Mesh": ...

    def to(self, *args, **kwargs) -> "Mesh":
        """
        Move the mesh to a different device and/or dtype.
        Supports PyTorch-like overloads:
        - to(device)
        - to(dtype)
        - to(device, dtype)
        - to(device=..., dtype=...)
        """
        device, dtype = get_device_dtype(*args, **kwargs)

        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        self.vertices = self.vertices.to(device=device, dtype=dtype)
        self.faces = self.faces.to(device=device)
        self.adjacencies = self.adjacencies.to(device=device)
        self.triangle_normals = self.triangle_normals.to(device=device, dtype=dtype)
        self.v2t = self.v2t.to(device=device)
        self.vertex_normals = self.vertex_normals.to(device=device, dtype=dtype)

        self.device = self.vertices.device
        self.dtype = dtype

        return self

    def _compute_adjacencies(self) -> NDArray[np.int32]:
        triangles = self.faces.cpu().numpy()
        num_triangles = len(triangles)
        adjacencies = -np.ones((num_triangles, 3), dtype=np.int32)
        edge_to_triangle = {}

        for tri_idx, tri in enumerate(triangles):
            for local_edge_idx, (i, j) in enumerate([(0, 1), (1, 2), (2, 0)]):
                edge = tuple(sorted((tri[i], tri[j])))
                if edge in edge_to_triangle:
                    other_tri_idx, other_local_edge_idx = edge_to_triangle[edge]
                    adjacencies[tri_idx, local_edge_idx] = other_tri_idx
                    adjacencies[other_tri_idx, other_local_edge_idx] = tri_idx
                else:
                    edge_to_triangle[edge] = (tri_idx, local_edge_idx)
        return adjacencies

    def _compute_vertex_to_triangle_map(self) -> NDArray[np.int32]:
        triangles = self.faces.cpu().numpy()
        num_vertices = self.vertices.shape[0]
        v2t = [[] for _ in range(num_vertices)]
        max_len = 0

        for tri_idx, tri in enumerate(triangles):
            for v in tri:
                v2t[v].append(tri_idx)
                max_len = max(max_len, len(v2t[v]))

        v2t_array = -np.ones((num_vertices, max_len + 1), dtype=np.int32)
        for i, tris in enumerate(v2t):
            v2t_array[i, 0] = len(tris)
            v2t_array[i, 1 : len(tris) + 1] = tris
        return v2t_array


class MeshBatch(Mesh):
    def __init__(self, meshes: List[Mesh]):
        """
        Allows batched operations of meshes.
        MeshPointsBatch must also be batched using the MeshBatch with batch_points
        and unbatch_points.

        Paramseters
        -----------
        meshes: List[Mesh]
            A list of Mesh objects to batch together.

        >>> mesh1 = load_mesh_from_file(path_to_mesh1, device=device)
        >>> mesh2 = load_mesh_from_file(path_to_mesh2, device=device)
        >>> start_points1 = uniform_sampling(mesh1, N).to(device)
        >>> start_points2 = uniform_sampling(mesh2, N).to(device)
        >>> mesh_batch = MeshBatch([mesh1, mesh2])
        >>> start_batched_points = mesh_batch.batch_points(
        ...     [start_points1, start_points2]
        ... )
        >>> start_dirs = torch.randn(
        ...     (2*N, 3), dtype=torch.float32
        ... ).to(device)
        >>> end_batched_points, geodesic_info = trace_geodesics(
        ...     mesh_batch, start_batched_points, start_dirs
        ... )
        >>> [end_points1, end_points2] = mesh_batch.unbatch_points(
        ...     end_batched_points
        ... )
        """
        if not meshes:
            raise ValueError("MeshBatch must contain at least one mesh.")

        device = meshes[0].device

        self.vertex_idx = torch.cumsum(
            torch.tensor(
                [0] + [mesh.vertices.shape[0] for mesh in meshes],
                dtype=torch.int32,
                device=device,
            ),
            dim=0,
        ).requires_grad_(False)
        self.triangle_idx = torch.cumsum(
            torch.tensor(
                [0] + [mesh.faces.shape[0] for mesh in meshes],
                dtype=torch.int32,
                device=device,
            ),
            dim=0,
        ).requires_grad_(False)

        vertices = torch.cat([mesh.vertices for mesh in meshes], dim=0)
        faces = torch.cat(
            [mesh.faces + self.vertex_idx[i] for i, mesh in enumerate(meshes)],
            dim=0,
        )

        adjacencies = torch.cat(
            [mesh.adjacencies + self.triangle_idx[i] for i, mesh in enumerate(meshes)],
            dim=0,
        )

        triangle_normals = torch.cat([mesh.triangle_normals for mesh in meshes], dim=0)

        max_v2t_length = max(mesh.v2t.shape[1] for mesh in meshes)
        padded_v2t = [
            torch.nn.functional.pad(
                torch.cat(
                    [
                        mesh.v2t[:, :1],  # Keep first column unchanged
                        mesh.v2t[:, 1:] + self.triangle_idx[i],  # Offset other columns
                    ],
                    dim=1,
                ),
                (0, max_v2t_length - mesh.v2t.shape[1]),
            )
            for i, mesh in enumerate(meshes)
        ]
        v2t = torch.cat(padded_v2t, dim=0)

        vertex_normals = torch.cat([mesh.vertex_normals for mesh in meshes], dim=0)

        super().__init__(
            vertices=vertices,
            faces=faces,
            adjacencies=adjacencies,
            triangle_normals=triangle_normals,
            v2t=v2t,
            vertex_normals=vertex_normals,
            device=device,
            dtype=meshes[0].dtype,
        )

    def unbatch(self) -> List[Mesh]:
        """
        Unbatch the mesh batch into a list of Mesh objects.

        Returns
        -------
        List[Mesh]
            A list of Mesh objects corresponding to the original meshes used to
            create the batch.
        """
        meshes = []
        for i in range(len(self)):
            start_vertex = self.vertex_idx[i]
            start_triangle = self.triangle_idx[i]
            end_vertex = self.vertex_idx[i + 1]
            end_triangle = self.triangle_idx[i + 1]

            v2t = self.v2t[start_vertex:end_vertex]
            v2t = torch.cat(
                [
                    v2t[:, :1],  # Keep first column unchanged
                    v2t[:, 1:] - start_triangle,  # Subtract only from other columns
                ],
                dim=1,
            )
            meshes.append(
                Mesh(
                    vertices=self.vertices[start_vertex:end_vertex],
                    faces=self.faces[start_triangle:end_triangle]
                    - start_vertex,
                    adjacencies=self.adjacencies[start_triangle:end_triangle]
                    - start_triangle,
                    triangle_normals=self.triangle_normals[start_triangle:end_triangle],
                    v2t=v2t,
                    vertex_normals=self.vertex_normals[start_vertex:end_vertex],
                    device=self.device,
                )
            )
        return meshes

    def batch_points(self, meshpoints: "List[MeshPointBatch]") -> "MeshPointBatch":
        """
        Batch a list of MeshPointBatch objects into a single MeshPointBatch.
        The output MeshPointBatch will be detached from the original MeshPointBatch
        objects.

        Parameters
        ----------
        meshpoints: List[MeshPointBatch]
            A list of MeshPointBatch objects to batch together. Where meshpoints[i]
            corresponds to the points on mesh i in the MeshBatch.

        Returns
        -------
        MeshPointBatch
            A single MeshPointBatch containing all the points from the input batches,
            with face indices adjusted to match the concatenated faces of the MeshBatch.
        """
        if not meshpoints:
            raise ValueError("meshpoints must contain at least one MeshPointBatch.")

        if len(meshpoints) != len(self):
            raise ValueError(
                "Number of MeshPointBatch objects must match the number of meshes in " \
                "the MeshBatch."
            )

        faces = torch.cat(
            [mp.faces + self.triangle_idx[i] for i, mp in enumerate(meshpoints)], dim=0
        )
        uvs = torch.cat([mp.uvs.detach() for mp in meshpoints], dim=0)

        return MeshPointBatch(faces=faces, uvs=uvs).to(self.device)

    def unbatch_points(self, meshpoints: "MeshPointBatch") -> "List[MeshPointBatch]":
        """
        Unbatch a MeshPointBatch into a list of MeshPointBatch objects.

        Parameters
        ----------
        meshpoints: MeshPointBatch
            A MeshPointBatch containing points on the MeshBatch. The face indices in
            meshpoints should correspond to the concatenated faces of the MeshBatch.

        Returns
        -------
        List[MeshPointBatch]
            A list of MeshPointBatch objects, where the i-th batch contains the points
            corresponding to the i-th mesh in the MeshBatch. The face indices in each
            batch will be adjusted to correspond to the original faces of the individual
            meshes.
        """
        meshpoint_batches = []
        for i in range(len(self)):
            start_triangle = self.triangle_idx[i]
            end_triangle = self.triangle_idx[i + 1]
            mask = (meshpoints.faces >= start_triangle) & (
                meshpoints.faces < end_triangle
            )

            meshpoint_batches.append(
                MeshPointBatch(
                    faces=meshpoints.faces[mask] - start_triangle,
                    uvs=meshpoints.uvs[mask],
                ).to(self.device)
            )

        return meshpoint_batches

    @overload
    def to(self, device: Union[str, torch.device]) -> "MeshBatch": ...

    @overload
    def to(self, dtype: torch.dtype) -> "MeshBatch": ...

    @overload
    def to(
        self, device: Union[str, torch.device], dtype: torch.dtype
    ) -> "MeshBatch": ...

    @overload
    def to(self, tensor: "Tensor") -> "MeshBatch": ...

    @overload
    def to(
        self,
        *,
        device: Optional[Union[str, torch.device]] = ...,
        dtype: Optional[torch.dtype] = ...,
    ) -> "MeshBatch": ...

    def to(self, *args, **kwargs) -> "MeshBatch":
        """
        Move the mesh to a different device and/or dtype.
        Supports PyTorch-like overloads:
        - to(device)
        - to(dtype)
        - to(device, dtype)
        - to(device=..., dtype=...)
        """
        device, dtype = get_device_dtype(*args, **kwargs)

        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        self.vertices = self.vertices.to(device=device, dtype=dtype)
        self.faces = self.faces.to(device=device)
        self.adjacencies = self.adjacencies.to(device=device)
        self.triangle_normals = self.triangle_normals.to(device=device, dtype=dtype)
        self.v2t = self.v2t.to(device=device)
        self.vertex_normals = self.vertex_normals.to(device=device, dtype=dtype)
        self.vertex_idx = self.vertex_idx.to(device=device)
        self.triangle_idx = self.triangle_idx.to(device=device)

        self.device = device
        self.dtype = dtype

        return self

    def __len__(self) -> int:
        """Return the number of meshes in the batch."""
        return len(self.vertex_idx) - 1

    def __getitem__(self, idx: int) -> Mesh:
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self.vertex_idx) - 1:
                raise IndexError("Index out of bounds for MeshBatch.")
            start_vertex = self.vertex_idx[idx]
            start_triangle = self.triangle_idx[idx]
            end_vertex = self.vertex_idx[idx + 1]
            end_triangle = self.triangle_idx[idx + 1]

            return Mesh(
                vertices=self.vertices[start_vertex:end_vertex],
                faces=self.faces[start_triangle:end_triangle] - start_vertex,
                adjacencies=self.adjacencies[start_triangle:end_triangle],
                triangle_normals=self.triangle_normals[start_triangle:end_triangle],
                v2t=self.v2t[start_vertex:end_vertex],
                vertex_normals=self.vertex_normals[start_vertex:end_vertex],
                device=self.device,
            )
        else:
            raise TypeError("Index must be an integer.")


class MeshPoint:
    def __init__(self, face: int, uv: Tensor):
        """
        Initialize a MeshPoint with a face index and UV coordinates.
        This should generally not be used directly, instead use MeshPointBatch for
        batching.
        """
        self.face = face
        self.uv = uv

    def __str__(self) -> str:
        return f"MeshPoint(face={self.face}, uv={self.uv.tolist()})"

    def interpolate(self, mesh: Mesh) -> Tensor:
        face = self.face
        uv = self.uv
        p0 = mesh.vertices[mesh.faces[face, 0]]
        p1 = mesh.vertices[mesh.faces[face, 1]]
        p2 = mesh.vertices[mesh.faces[face, 2]]
        pos = (1 - uv[0] - uv[1]) * p0 + uv[0] * p1 + uv[1] * p2
        return pos

    def detach(self) -> "MeshPoint":
        return MeshPoint(self.face, self.uv.detach())

    def get_barycentric_coords(self) -> Tensor:
        return torch.tensor(
            [1.0 - self.uv[0] - self.uv[1], self.uv[0], self.uv[1]], dtype=self.uv.dtype
        )


class MeshPointBatch:
    def __init__(self, faces: Tensor, uvs: Tensor):
        """
        Initialize a batch of MeshPoints.
        """
        if faces.dim() != 1 or faces.dtype != torch.int32:
            raise ValueError("faces must be a 1D tensor of dtype torch.int32.")
        if uvs.dim() != 2 or uvs.size(1) != 2:
            raise ValueError("uvs must be a 2D tensor with shape (N, 2).")
        if faces.size(0) != uvs.size(0):
            raise ValueError(
                f"faces and uvs must have the same length. Found {faces.size(0)} and " \
                f"{uvs.size(0)}."
            )

        self.faces: Tensor = faces
        self.uvs: Tensor = uvs
        self.uvs.requires_grad_()

    @classmethod
    def from_points(cls, points: Iterable[MeshPoint]) -> "MeshPointBatch":
        """
        Create a MeshPointBatch from an iterable of MeshPoint objects.
        """
        faces = torch.tensor([p.face for p in points], dtype=torch.int32)
        uvs = torch.stack([p.uv for p in points], dim=0)
        return cls(faces, uvs)

    @overload
    def to(self, device: Union[str, torch.device]) -> "MeshPointBatch": ...

    @overload
    def to(self, dtype: torch.dtype) -> "MeshPointBatch": ...

    @overload
    def to(
        self, device: Union[str, torch.device], dtype: torch.dtype
    ) -> "MeshPointBatch": ...

    @overload
    def to(self, tensor: "Tensor") -> "MeshPointBatch": ...

    @overload
    def to(
        self,
        *,
        device: Optional[Union[str, torch.device]] = ...,
        dtype: Optional[torch.dtype] = ...,
    ) -> "MeshPointBatch": ...

    def to(self, *args, **kwargs) -> "MeshPointBatch":
        """
        Move the mesh to a different device and/or dtype.
        Supports PyTorch-like overloads:
        - to(device)
        - to(dtype)
        - to(device, dtype)
        - to(device=..., dtype=...)
        """
        device, dtype = get_device_dtype(*args, **kwargs)

        if device is None:
            device = str(self.uvs.device)
        if dtype is None:
            dtype = self.uvs.dtype

        return MeshPointBatch(
            faces=self.faces.to(device=device),
            uvs=self.uvs.to(device=device, dtype=dtype),
        )

    def detach(self) -> "MeshPointBatch":
        return MeshPointBatch(faces=self.faces.detach(), uvs=self.uvs.detach())

    def clone(self) -> "MeshPointBatch":
        return MeshPointBatch(faces=self.faces.clone(), uvs=self.uvs.clone())

    def interpolate(self, mesh: Mesh, return_batch: bool = False) -> Tensor:
        """
        Interpolate the positions of the mesh points in the batch.

        Parameters
        ----------
        mesh: Mesh
            The mesh on which the points lie. The face indices in self.faces should
            correspond to the faces of this mesh.
        return_batch: bool (default=False)
            If True, return the interpolated positions as a batch of shape (B, N, 3),
            where B is the number of meshes in the batch (if mesh is a MeshBatch)
            and N is the number of points in the batch. If False, return a tensor of
            shape (N, 3) containing the interpolated positions of the points.

        Returns
        -------
        Tensor
            The interpolated positions of the mesh points. Shape is (N, 3) if
            return_batch is False, or (B, N, 3) if return_batch is True and
            mesh is a MeshBatch.
        """
        p0 = mesh.vertices[mesh.faces[self.faces, 0]]
        p1 = mesh.vertices[mesh.faces[self.faces, 1]]
        p2 = mesh.vertices[mesh.faces[self.faces, 2]]
        pos = (
            (1 - self.uvs[:, 0:1] - self.uvs[:, 1:2]) * p0
            + self.uvs[:, 0:1] * p1
            + self.uvs[:, 1:2] * p2
        )

        if return_batch:
            if not isinstance(mesh, MeshBatch):
                raise ValueError(
                    "Mesh must be a MeshBatch to return a batch of positions."
                )

            batch_size = mesh.vertex_idx.shape[0] - 1
            return pos.view(batch_size, -1, 3)
        else:
            return pos

    def get_barycentric_coords(self) -> Tensor:
        """
        Get the barycentric coordinates of the mesh points in the batch.

        Returns
        -------
        Tensor
            A tensor of shape (N, 3) containing the barycentric coordinates of each
            point in the batch.
        """
        return torch.cat(
            [
                1.0 - self.uvs[:, 0:1] - self.uvs[:, 1:2],
                self.uvs[:, 0:1],
                self.uvs[:, 1:2],
            ],
            dim=1,
        )

    def to_list(self) -> list[MeshPoint]:
        return [
            MeshPoint(int(face.item()), uv) for face, uv in zip(self.faces, self.uvs)
        ]

    @overload
    def __getitem__(self, idx: int) -> MeshPoint: ...

    @overload
    def __getitem__(self, idx: slice | Tensor) -> "MeshPointBatch": ...

    def __getitem__(
        self, idx: int | slice | Tensor
    ) -> "Union[MeshPoint,MeshPointBatch]":
        if isinstance(idx, int):
            return MeshPoint(int(self.faces[idx].item()), self.uvs[idx])
        elif isinstance(idx, slice | Tensor):
            return MeshPointBatch(faces=self.faces[idx], uvs=self.uvs[idx])
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def __len__(self) -> int:
        return len(self.faces)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
