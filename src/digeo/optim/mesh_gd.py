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
    Gradient descent optimizer with line search and early stopping.
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
