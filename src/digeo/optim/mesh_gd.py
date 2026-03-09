import time
from tqdm import tqdm
from typing import Tuple

from digeo import Mesh, MeshPointBatch
from digeo.ops import trace_geodesics
from digeo.optim.utils import MeshLossFunc, line_search


def mesh_gd(
    mesh: Mesh,
    x: MeshPointBatch,
    loss_func: MeshLossFunc,
    use_line_search=True,
    max_iter=100,
    lr=1.0,
    tol=1e-6,
    min_rel_improvement=1e-5,
    patience=3,
) -> Tuple[MeshPointBatch, dict]:
    """
    Gradient descent optimiser with line search and early stopping.

    Parameters
    ----------
    mesh: Mesh
        The mesh used for the optimisation.
    x: MeshPointBatch
        The initial points on the mesh to optimize.
    loss_func: MeshLossFunc
        The loss function to optimize.
    use_line_search: bool
        Whether to use line search to find the optimal step size at each iteration
        (default: True).
    max_iter: int
        The maximum number of iterations (default: 100).
    lr: float
        The initial learning rate for the line search (default: 1.0).
    tol: float
        The tolerance for the stopping criterion based on the gradient norm
        (default: 1e-6).
    min_rel_improvement: float
        The minimum relative improvement in the loss for early stopping
        (default: 1e-5).
    patience: int
        The number of iterations to wait for an improvement before stopping
        (default: 3).

    Returns
    -------
    x': MeshPointBatch
        The optimized points on the mesh.
    logs: dict[str, Any]
        A dictionary containing the information to log, where the keys are the
        names of the metrics and the values are the corresponding values to log.
        By default, the logs contain:

        - ``"loss"`` (List[float]): The loss value at each iteration.
        - ``"time"`` (float): The total time taken for the optimization.
        - ``"function_calls"`` (List[int]): The total number of function calls at
          each iteration.
        - ``"step_size"`` (List[float]): The step size at each iteration.
        - ``"mean_dir"`` (List[float]): The mean direction norm at each iteration.
    """
    loss, grad = loss_func(mesh, x)
    total_function_calls = 1
    function_calls = []
    progress_bar = tqdm(range(max_iter))

    total_loss = []
    total_step_size = []
    total_mean_dir = []

    t0 = time.process_time()
    no_improve_counter = 0
    for _ in progress_bar:
        if use_line_search:
            x, grad, loss_new, alpha, c, _ = line_search(
                mesh=mesh,
                x=x,
                loss_func=loss_func,
                direction=grad,
                curr_loss=loss,
                curr_grad=grad,
                alpha_init=lr,
                use_wolfe=False,
            )
            total_function_calls += c
        else:
            alpha = lr
            x, _ = trace_geodesics(mesh, x, alpha * grad, gradient="none")
            loss_new, grad = loss_func(mesh, x)
            total_function_calls += 1

        total_step_size.append(alpha)
        total_mean_dir.append((alpha * grad).norm(dim=1).mean().item())
        total_loss.append(loss_new)
        function_calls.append(total_function_calls)
        progress_bar.set_description(f"Loss: {loss_new:.6f}")

        # Early stopping based on relative loss improvement
        rel_improvement = abs(loss_new - loss) / (abs(loss_new) + 1e-12)
        if rel_improvement < min_rel_improvement:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                print(
                    f"[Early Stopping] No improvement for {patience} steps "
                    f"(rel_impr < {min_rel_improvement})"
                )
                break
        else:
            no_improve_counter = 0
        loss = loss_new

        if alpha * grad.norm(dim=-1).mean() < tol:
            print(
                f"[Early Stopping] Gradient norm {alpha * grad.norm():.2e} < tol={tol}"
            )
            break

    logs = {
        **loss_func.get_logs(),
        "loss": total_loss,
        "function_calls": function_calls,
        "time": time.process_time() - t0,
        "step_size": total_step_size,
        "mean_dir": total_mean_dir,
    }
    return x, logs
