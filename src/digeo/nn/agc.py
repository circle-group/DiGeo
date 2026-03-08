import torch
import torch.nn as nn

from digeo.ops import trace_geodesics
from digeo import Mesh, MeshPointBatch


def batched_dir_rotation(
    dirs: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor
) -> torch.Tensor:
    """
    Parameters
    ----------
        dirs (Tensor): [V, 3]
        axis (Tensor): [V, 3]
        angle (Tensor): [n_theta]
    Returns
    -------
        Tensor: [V, n_theta, 3]
    """
    dirs = dirs.unsqueeze(1)
    axis = axis.unsqueeze(1)
    angle = angle.view(1, -1, 1)

    # Apply Rodrigues rotation formula
    new_dirs = (
        dirs * torch.cos(angle)
        + torch.cross(axis, dirs, dim=-1) * torch.sin(angle)
        + axis * torch.sum(axis * dirs, dim=-1, keepdim=True) * (1 - torch.cos(angle))
    )
    return new_dirs


class AGC(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        n_patches: int,
        rho_init: float = 0.1,
        n_rho: int = 2,
        n_theta: int = 8,
        learn_rho: bool = True,
    ):
        """
        Adaptive Geodesic Convolutional Layer.
        Takes as input a mesh and per-vertex features of shape (V, C).
        Where C=in_filters * n_patches, and produces output features of shape (V, C')
        where C'=out_filters * n_patches.

        Parameters
        ----------
        in_filters : int
            Number of input filters per patch.
        out_filters : int
            Number of output filters per patch.
        n_patches : int
            Number of patches to trace per vertex.
        rho_init : float
            Initial radius of the largest geodesic patch.
        n_rho : int
            Number of rings in the geodesic patch.
        n_theta : int
            Number of angular divisions in the geodesic patch.
        learn_rho : bool
            Whether to make the patch radii learnable.
        """
        super(AGC, self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.rho_init = rho_init
        self.n_rho = n_rho
        self.n_theta = n_theta
        self.learn_rho = learn_rho
        self.n_patches = n_patches

        self.in_channels = in_filters * n_patches
        self.out_channels = out_filters * n_patches

        rho = rho_init * torch.ones(
            (1, n_rho, 1, self.n_patches, 1), dtype=torch.float32
        )  # [1, n_rho, 1, n_patches, 1]
        for k in range(n_rho):
            rho[0, k, :, 0] = (k + 1) / n_rho * rho[0, n_rho - 1, :, 0]
        if learn_rho:
            self.rho = nn.Parameter(rho)
        else:
            self.register_buffer("rho", rho)

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=(n_rho, n_theta),
            bias=True,
        )

        self.self_conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, mesh: Mesh, X: torch.Tensor):
        """
        Forward pass of the AGC layer.

        Parameters
        ----------
        mesh: Mesh
            The mesh object containing vertices and faces.
        X: Tensor
            Input features of shape (V, C)
        Returns
        -------
            Tensor: Output features of shape (V, C')
        """
        num_vertices = mesh.vertices.shape[0]

        ### Get start vertices for geodesic tracing
        vertex_faces = mesh.v2t[:, 1]
        vertex_bary = mesh.faces[vertex_faces] == torch.arange(
            mesh.vertices.shape[0], dtype=torch.int32, device=mesh.vertices.device
        ).unsqueeze(1)
        vertex_bary = vertex_bary.float()
        start_vertices = MeshPointBatch(
            faces=vertex_faces.repeat_interleave(
                self.n_patches * self.n_rho * self.n_theta
            ),
            uvs=vertex_bary[:, 1:].repeat_interleave(
                self.n_patches * self.n_rho * self.n_theta, dim=0
            ),
        )

        ### Get the patch directions
        vertex_normals = mesh.vertex_normals  # [V, 3]
        directions = torch.tensor(
            [4.43948340493, 2.4309423923, -3.903940323],
            dtype=torch.float32,
            device=mesh.vertices.device,
        )
        directions = directions.unsqueeze(0).expand(num_vertices, -1)  # [V, 3]
        directions = (
            directions
            - torch.sum(directions * vertex_normals, dim=-1, keepdim=True)
            * vertex_normals
        )  # Project onto tangent plane
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp(
            min=1e-6
        )  # [V, 3]

        rotation_angles = torch.arange(
            0,
            2 * torch.pi,
            step=2 * torch.pi / self.n_theta,
            dtype=torch.float32,
            device=mesh.vertices.device,
        )  # [n_theta]
        directions = batched_dir_rotation(
            directions, vertex_normals, rotation_angles
        )  # [V, n_theta, 3]

        directions = directions.unsqueeze(1)  # [V, 1, n_theta, 3]
        directions = directions.unsqueeze(-2)  # [V, 1, n_theta, 1, 3]
        directions = directions * self.rho.abs()  # [V, n_rho, n_theta, n_traces, 3]

        meshpoints, _ = trace_geodesics(
            mesh=mesh,
            starts=start_vertices,  # [V * num_traces * n_patches]
            dirs=directions.contiguous().view(-1, 3),  # [V * num_traces * n_patches, 3]
            gradient="parallel_transport" if self.learn_rho else "none",
        )

        tri_idx = meshpoints.faces  # [V * num_traces * n_patches]
        idx = mesh.faces[tri_idx]  # [V * num_traces * n_patches, 3]
        idx = idx.view(
            -1, self.n_patches, 3
        )  # Reshape to [V * num_traces, n_patches, 3]
        idx = idx.unsqueeze(-2).expand(
            -1, self.n_patches, self.in_filters, 3
        )  # [V * num_traces, n_patches, in_filters, 3]
        idx = idx.reshape(-1)  # [V * num_traces * in_channels * 3]
        channel_idx = (
            torch.arange(
                self.in_channels, dtype=torch.long, device=mesh.vertices.device
            )
            .view(1, 1, 1, self.in_channels, 1)
            .expand(num_vertices, self.n_rho, self.n_theta, self.in_channels, 3)
            .reshape(-1)
        )  # [V * n_rho * n_theta * in_channels * 3]
        meshpoints_features = X[idx, channel_idx].view(
            num_vertices, self.n_rho, self.n_theta, self.in_channels, 3
        )  # [V, n_rho, n_theta, in_channels, 3]
        barycentric_coords = meshpoints.get_barycentric_coords().view(
            num_vertices, self.n_rho, self.n_theta, self.n_patches, 3
        )  # [V, n_rho, n_theta, n_patches, 3]
        barycentric_coords = barycentric_coords.unsqueeze(-2).expand(
            -1, -1, -1, -1, self.in_filters, -1
        )  # [V, n_rho, n_theta, trace_factor, in_filters, 3]
        barycentric_coords = barycentric_coords.reshape(
            num_vertices, self.n_rho, self.n_theta, self.in_channels, 3
        )  # [V, n_rho, n_theta, in_channels, 3]
        meshpoints_features = (
            meshpoints_features * barycentric_coords
        )  # Apply barycentric coordinates
        meshpoints_features = meshpoints_features.sum(
            dim=-1
        )  # [V, n_rho, n_theta, in_channels]
        meshpoints_features = meshpoints_features.permute(
            0, 3, 1, 2
        )  # [V, in_channels, n_rho, n_theta]
        meshpoints_features = torch.cat(
            (meshpoints_features, meshpoints_features[..., : self.n_theta - 1]), dim=-1
        )  # [V, in_channels, n_rho, 2*n_theta-1]

        meshpoints_features = self.conv(
            meshpoints_features
        )  # [V, out_channels, 1, n_theta-1]
        meshpoints_features = meshpoints_features.squeeze(
            2
        )  # [V, out_channels, n_theta-1]
        meshpoints_features = meshpoints_features.max(dim=-1)[0]  # [V, out_channels]

        self_features = self.self_conv(X.unsqueeze(-1)).squeeze(-1)  # [V, out_channels]

        return meshpoints_features + self_features
