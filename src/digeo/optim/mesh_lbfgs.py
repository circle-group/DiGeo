import torch
import time
from tqdm import tqdm
from typing import Tuple

from digeo import Mesh, MeshPointBatch
from digeo.optim.utils import MeshLossFunc, line_search


def mesh_lbfgs(
    mesh: Mesh,
    x0: MeshPointBatch,
    loss_func: MeshLossFunc,
    max_iter: int = 100,
    list_size: int = 20,
    lr: float = 1.0,
    tol: float = 1e-6,
    min_rel_improvement: float = 1e-5,
    patience: int = 3,
    eps: float = 1e-6,
) -> Tuple[MeshPointBatch, dict]:
    """
    Mesh-LBFGS optimizer with early stopping.
    """
    x = x0
    loss, grad = loss_func(mesh, x)

    H_diag = 1.0
    s_list, y_list, sy_list, rot_list = [], [], [], []
    total_loss = []
    step_size = []
    mean_dir = []
    function_calls = []
    total_function_calls = 1
    t0 = time.process_time()
    no_improve_counter = -1

    progress_bar = tqdm(range(max_iter))
    for _ in progress_bar:
        grad_norm = grad.norm().item()

        if grad_norm < tol:
            print(f"[Early Stopping] Gradient norm {grad_norm:.2e} < tol={tol}")
            break

        dir = desc(
            grad, len(s_list) - 1, s_list, y_list, sy_list, rot_list, H_diag, eps
        )

        x_new, grad_new, loss_new, alpha, c, info = line_search(
            mesh=mesh,
            x=x,
            loss_func=loss_func,
            direction=dir,
            curr_loss=loss,
            curr_grad=grad,
            alpha_init=lr,
            use_wolfe=True,
        )

        total_function_calls += c
        function_calls.append(total_function_calls)
        step_size.append(alpha)
        total_loss.append(loss_new)
        mean_dir.append((alpha * dir).norm(dim=1).mean().item())

        if mean_dir[-1] < tol:
            print(
                f"[Early Stopping] Mean direction norm {mean_dir[-1]:.2e} < tol={tol}"
            )
            break

        with torch.no_grad():
            s = torch.bmm((alpha * dir).unsqueeze(1), info.rotation).squeeze(1)
            transported_grad = torch.bmm(grad.unsqueeze(1), info.rotation).squeeze(1)
            y = -(grad_new - transported_grad)

            s_list.append(s)
            y_list.append(y)
            sy_list.append(torch.sum(s * y))
            rot_list.append(info.rotation)

            if len(s_list) > list_size:
                del s_list[0], y_list[0], sy_list[0], rot_list[0]

            H_diag = sy_list[-1] / (torch.sum(y * y) + eps)

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

            x = x_new
            grad = grad_new
            loss = loss_new
            progress_bar.set_description(f"Loss: {loss:.6f}")

    logs = {
        **loss_func.get_logs(),
        "loss": total_loss,
        "time": time.process_time() - t0,
        "function_calls": function_calls,
        "step_size": step_size,
        "mean_dir": mean_dir,
    }

    return x, logs


@torch.no_grad()
def desc(p, t, s_list, y_list, sy_list, rot_list, H_diag, eps=1e-6):
    if t < 0:
        return H_diag * p
    start_p = p.clone()
    p = p - (torch.sum(s_list[t] * p) / (sy_list[t] + eps)) * y_list[t]
    pt = torch.bmm(p.unsqueeze(1), rot_list[t].transpose(1, 2)).squeeze(1)
    p = desc(pt, t - 1, s_list, y_list, sy_list, rot_list, H_diag, eps)
    p = torch.bmm(p.unsqueeze(1), rot_list[t]).squeeze(1)

    return (
        p
        - (torch.sum(y_list[t] * p) / (sy_list[t] + eps)) * s_list[t]
        + (torch.sum(s_list[t] * s_list[t]) / (sy_list[t] + eps)) * start_p
    )
