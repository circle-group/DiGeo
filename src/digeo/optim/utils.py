import torch
from abc import ABC, abstractmethod
from typing import Any, Tuple
from torch import Tensor

from digeo import Mesh, MeshPointBatch
from digeo.ops import trace_geodesics
from digeo.ops.geodesic import GeodesicInfo


class MeshLossFunc(ABC):
    def __call__(self, mesh: Mesh, points: MeshPointBatch) -> Tuple[float, Tensor]:
        return self.compute(mesh, points)

    @abstractmethod
    def compute(self, mesh: Mesh, points: MeshPointBatch) -> Tuple[float, Tensor]:
        """
        Loss function used in the mesh optimisers.

        Parameters
        ----------
        mesh : Mesh
            The mesh used
        points: MeshPointBatch
            The points on the mesh that are being optimized

        Returns
        -------
        loss: float
            The loss of the function
        gradient: Tensor
            The gradients in 3d space for each point
        """
        ...

    def get_logs(self) -> dict[str, Any]:
        """
        Optionnal method to return any additional information that should be logged
        during optimization, such as the value of certain terms in the loss function,
        or any other relevant information.

        Returns
        -------
        logs: dict[str, Any]
            A dictionary containing the information to log, where the keys are the
            names of the metrics and the values are the corresponding values to log.
        """
        return {}


@torch.no_grad()
def line_search(
    mesh: Mesh,
    x: MeshPointBatch,
    loss_func: MeshLossFunc,
    direction: Tensor,
    curr_loss: float,
    curr_grad: Tensor,
    use_wolfe=True,
    c1=1e-4,
    c2=0.9,
    alpha_init=1.0,
    tau=0.1,
    max_iter=3,
) -> Tuple[MeshPointBatch, Tensor, float, float, int, GeodesicInfo]:
    """
    Wolfe line search satisfying Armijo and curvature conditions.

    Parameters
    ----------
    mesh: Mesh
        The mesh used
    x: MeshPointBatch
        The current points on the mesh
    loss_func: MeshLossFunc
        The loss function
    direction: Tensor
        The direction used for the line search to optimize x
    curr_loss: float
        The current loss for x
    curr_grad: Tensor
        The current gradients for x for loss_func
    use_wolfe: bool
        If the Wolfe conditions should be used (default: True)
    c1: float
        First Wolfe condition constant (default: 1e-4)
    c2: float
        Second Wolfe condition constant (default: 0.9)
    alpha_init: float
        The initial step size (defailt: 1.0)
    tau: float
        The backtracking reduction factor (default: 0.1)
    max_iter: int
        The maximum number of iterations (default: 3)

    Returns
    -------
    x': MeshPointBatch
        The new meshpoints
    gradient': Tensor
        The gradients for x'
    loss: float
        The loss of the loss function for x'
    alpha: float
        step size chosen satisfying Wolfe conditions
    function_calls: int
        The number of times the loss function was called
    info: GeodesicInfo
        The geodesic info of the trace from x to x'
    """
    alpha = alpha_init
    f_x = curr_loss
    grad = curr_grad
    directional_derivative = torch.sum(-grad * direction).item()
    directional_derivative = min(directional_derivative, 0)
    function_calls = 0

    for k in range(max_iter):
        x_new, info = trace_geodesics(
            mesh,
            x,
            alpha * direction,
            gradient="none",
            save_parallel_transport=True,
            max_steps=10000,
            print_warnings=False,
        )
        loss, new_grad = loss_func(mesh, x_new)
        function_calls += 1

        with torch.no_grad():
            # Armijo condition
            if loss > f_x + c1 * alpha * directional_derivative:
                alpha *= tau
                continue

            if not use_wolfe:
                break

            transported_dir = torch.bmm(direction.unsqueeze(1), info.rotation).squeeze(
                1
            )
            new_derivative = torch.sum(-new_grad * transported_dir)

            # Curvature condition
            if -new_derivative <= -c2 * directional_derivative:
                break
            else:
                alpha *= tau

    return x_new, new_grad, loss, alpha, function_calls, info
