import torch
import numpy as np
import scipy
from typing import Tuple
import robust_laplacian
import scipy.sparse.linalg

from digeo import Mesh, MeshPointBatch


def compute_mesh_laplacian(
    verts: np.ndarray, faces: np.ndarray
) -> Tuple[scipy.sparse.csc_matrix, np.ndarray]:
    lapl, mass = robust_laplacian.mesh_laplacian(verts, faces)
    return lapl, mass.diagonal()


def compute_eig_laplacian(
    lapl: scipy.sparse.csc_matrix,
    massvec: np.ndarray,
    k_eig: int = 128,
    eps: float = 1e-8,
) -> Tuple[np.typing.NDArray, np.typing.NDArray]:
    """
    Compute the eigendecomposition of the Laplacian

    Parameters
    ----------
    lapl: Sparse matrix
        [N x N] Laplacian
    massvec: Array
        [N] mass vector
    k_eig: int, optional
        number of eigenvalues and eigenvectors desired (default: 128).
    eps: float, optional
        constant used to perturb Laplacian during eigendecomposition (default: 1e-8).

    Raises
    ------
    ValueError: although multiple attempts were made, the eigendecomposition
        failed.

    Returns
    -------
    eigvals: Array
        [k,] eigenvalues
    eigvecs: Array
        [N x k] eigenvectors, where each column is an eigenvector.
    """
    lapl_eigsh = (lapl + scipy.sparse.identity(lapl.shape[0]) * eps).tocsc()
    mass_mat = scipy.sparse.diags(massvec)
    eigs_sigma = eps

    failcount = 0
    while True:
        try:
            evals, evecs = scipy.sparse.linalg.eigsh(
                lapl_eigsh.astype(np.float32),
                k=k_eig+1,
                M=mass_mat.astype(np.float32),
                sigma=eigs_sigma,
            )
            evals = np.clip(evals, a_min=0.0, a_max=float("inf"))
            break
        except RuntimeError as exc:
            if failcount > 3:
                raise ValueError("failed to compute eigendecomp") from exc
            failcount += 1
            print("--- decomp failed; adding eps ===> count: " + str(failcount))
            lapl_eigsh = lapl_eigsh + scipy.sparse.identity(lapl.shape[0]) * (
                eps * 10**failcount
            )
    return evals, evecs


def interpolate_barycentric_attr(f, fi, bc, attribute):
    return (attribute[f[fi]] * bc[:, :, None]).sum(1)


def compute_biharmonic_distance(
    evecs_i: torch.Tensor,
    evecs_j: torch.Tensor,
    evals: torch.Tensor,
    pairwise: bool = True,
):
    """
    Compute the biharmonic distance between two sets of points on the surface given
    their eigenfunctions. Use eigenvectors and eigenvalues from the isotropic LBO!
    The eigenvalues are normally computed on the vertices of a mesh, their value can be
    gathered as 'evecs[idxs_i]'. If need to get eigenvalues of points belonging to
    faces, need to interpolate first. See example below.

    Parameters
    ----------
    evecs_i: Tensor
        [#I_pts, K] eigenvectors at points i
    evecs_j: Tensor
        [#J_pts, K] eigenvectors at points j
    evals: Tensor
        [K,] eigenvalues
    pairwise: bool, optional
        whether it should compute all distances between I and J points. If False
        and #I_pts == #J_pts, it will compute the per attribute distances.
        (defaults: True).

    Returns
    -------
    Tensor
        [#I_pts, #J_pts] distances between points i and j

    Example
    -------
    >>> lapl, mass = utils.compute_mesh_laplacian(verts, faces)
    >>> evals, evecs = utils.compute_eig_laplacian(lapl, mass, 256)
    >>> v, f = torch.tensor(verts), torch.tensor(faces)
    >>> evals, evecs = torch.tensor(evals), torch.tensor(evecs)

    >>> # select eigenvalies of 2 points on faces of the mesh
    >>> fid = torch.tensor([0, 399])
    >>> bc = torch.tensor([[0.7410, 0.1356, 0.1234],
                            [0.3304, 0.1547, 0.5149]], dtype=torch.float64)

    >>> fevecs = utils.interpolate_barycentric_coords(f, fid, bc, evecs)

    >>> # select eigenvalies of 3 points on vertices of the mesh
    >>> pevecs = evecs[torch.tensor([10, 99, 800])]

    >>> # compute bharmonic distances
    >>> dists = utils.compute_biharmonic_distance(fevecs, pevecs, evals, True)
    """
    inv_evals = evals.pow(-1)  # (d,)

    if not pairwise:
        # simple 1-to-1 case
        diff = (evecs_i - evecs_j) * inv_evals
        return diff.pow(2).sum(dim=-1).sqrt()

    out = []
    for row in evecs_i:  # each row is (d,)
        diff = (row.unsqueeze(0) - evecs_j) * inv_evals  # (Nj, d)
        out.append(diff.pow(2).sum(dim=-1).sqrt())  # (Nj,)

    return torch.stack(out, dim=0)  # (Ni, Nj)


class BiharmonicDistance(torch.nn.Module):
    def __init__(self, mesh: Mesh, k_eig: int = 128, eps: float = 1e-5):
        """
        Initialize the BiharmonicDistance module.

        Parameters
        ----------
        mesh: Mesh
            The mesh used.
        k_eig: int
            The number of eigenvalues and eigenvectors to compute for the Laplacian.
        eps: float
            A small constant added to the Laplacian during eigendecomposition to
            ensure numerical stability.
        """
        super(BiharmonicDistance, self).__init__()
        self.mesh = mesh
        self.k_eig = k_eig
        self.eps = eps

        lapl, massvec = compute_mesh_laplacian(
            mesh.positions.cpu().numpy(), mesh.triangles.cpu().numpy()
        )
        evals, evecs = compute_eig_laplacian(lapl, massvec, k_eig=k_eig, eps=eps)

        # We ignore the first eigenvalue and eigenvector since it does not
        # contribute to the biharmonic distance.
        evals = evals[1:]
        evecs = evecs[:, 1:]

        # Convert to torch tensors
        evals = torch.as_tensor(evals, dtype=torch.float32, device=mesh.device)
        evecs = torch.as_tensor(evecs, dtype=torch.float32, device=mesh.device)

        # Register buffers
        self.register_buffer("evals", evals)
        self.register_buffer("evecs", evecs)

    def forward(
        self, points: MeshPointBatch, targets: MeshPointBatch, pairwise: bool = False
    ):
        """
        Compute the biharmonic distance between two sets of points on the surface given
        their eigenfunctions.

        Parameters
        ----------
        points: MeshPointBatch
            The first set of points for which to compute distances.
        targets: MeshPointBatch
            The second set of points for which to compute distances.
        pairwise: bool, optional
            Whether to compute all pairwise distances between points and targets, or
            just the per-attribute distances (if points and targets have the same
            number of points).
        """
        point_evecs = interpolate_barycentric_attr(
            self.mesh.triangles,
            points.faces,
            points.get_barycentric_coords(),
            self.evecs,
        )
        target_evecs = interpolate_barycentric_attr(
            self.mesh.triangles,
            targets.faces,
            targets.get_barycentric_coords(),
            self.evecs,
        )

        # compute bharmonic distances
        dists = compute_biharmonic_distance(
            point_evecs, target_evecs, self.evals, pairwise
        ).squeeze()

        if pairwise:
            return dists
        else:
            return torch.mean(dists)
