MeshFlow
========

.. figure:: /_static/duck_distrib.svg
   :align: center

   Visual representation of MeshFlow. Noise samples, :math:`\mathbf{p}_i`, are displaced via our :math:`K`-step exponential map according to the directions dictated by our learnable static vector field, :math:`\mathbf{v}_\theta`. The end-points, :math:`\hat{\mathbf{q}}_i`, are compared with the OT-coupled training samples, :math:`\mathbf{q}_{\sigma(i)}`, using a squared biharmonic distance function :math:`d_{BH}^2`.


By utilizing the differentiable exponential map, MeshFlow learns a stationary vector field to transport points from a base noise distribution to a target data distribution entirely on the manifold.

The method relies on the Geodesic Finite Differences (GFD) differentiation scheme to enable back-propagation through both point positions and vectors. Since the straightest geodesics algorithm naturally follows straight paths, MeshFlow breaks the trajectory into multiple discrete steps to allow for curved paths on the manifold.

Mathematical Formulation
------------------------

The objective of MeshFlow is to optimize a time-invariant vector field by comparing transported samples against data samples using point-wise correspondences.

.. math::

   \mathcal{L}(\theta) =
   \frac{1}{B}\sum_{i=1}^{B}
   d_{BH}^{2}\big(
      \operatorname{Exp}_{p_{i}}^{\bigcirc K}(v_{\theta}),
      q_{\sigma(i)}
   \big)

Where:

- :math:`B` is the batch size.

- :math:`p_{i}` represents initial samples drawn from a uniform base distribution over the mesh :math:`\mathcal{U}(\mathcal{M})`.

- :math:`q_{i}` represents samples drawn from the target distribution :math:`\mathcal{Q}(\mathcal{M})`.

- :math:`d_{BH}` represents the biharmonic distance, used as a computationally efficient surrogate for the true geodesic distance.

- :math:`\sigma` is a permutation of :math:`[1, B]` that represents the Optimal Transport (OT) coupling for the batch.

- :math:`\operatorname{Exp}^{\bigcirc K}` represents the :math:`K`-step exponential map, which iteratively breaks the trajectory into multiple steps to enable non-straightest curves on the mesh :math:`\mathcal{M}`.

- :math:`v_{\theta}` is the learned static vector field parameterized by a neural network.

Implementation
--------------

To implement this, the neural network uses a Multi-Layer Perceptron (MLP) that takes the 3D position of the samples as input and outputs the corresponding tangent vector. Because the differentiable exponential map operates intrinsically, it allows points to move directly along the surface of the mesh without introducing any projection errors during the transport process.

The full implementation is available on the `MeshFlow GitHub repository <https://github.com/Etyl/MeshFlow>`_.
