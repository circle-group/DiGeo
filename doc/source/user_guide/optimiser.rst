Mesh Optimisers
===============

Creating a Loss Function
------------------------

To create a loss function, you can subclass the :class:`digeo.optim.MeshLossFunc` class and implement the ``compute`` method.
The ``compute`` method should take in the mesh and the points, and return a tuple containing the loss value and the gradient with respect to the points.
The gradients should be in 3d space, and the loss value should be a scalar.

With the loss function defined, you can then use it in the optimisers provided in the ``digeo.optim`` module, such as ``mesh_gd`` and ``mesh_lbfgs``.

Example
-------

Here is an example of a simple loss function that computes the distance from the points to a target point:

.. code-block:: python

    import torch
    from digeo import Mesh, MeshPointBatch, load_mesh_from_file
    from digeo.optim import MeshLossFunc, mesh_lbfgs
    from digeo.ops import uniform_sampling

    class DistanceToTarget(MeshLossFunc):
        def __init__(self, target: torch.Tensor):
            self.target = target

        def compute(self, mesh: Mesh, points: MeshPointBatch) -> Tuple[float, torch.Tensor]:
            # Compute the distance from the points to the target
            distances = torch.norm(points - self.target, dim=-1)
            loss = distances.mean()
            # Compute the gradient with respect to the points
            grad = (points - self.target) / (distances.unsqueeze(-1) + 1e-8)
            return loss.item(), grad

    loss_func = DistanceToTarget(target=torch.tensor([0.0, 0.0, 0.0]))
    mesh = load_mesh_from_file("path/to/mesh.obj")
    x = uniform_sampling(mesh, num_points=100)

    result, logs = mesh_lbfgs(mesh, x, loss_func, max_iter=100)
