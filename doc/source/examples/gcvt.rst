Geodesic Centroidal Voronoi Tessellation
========================================

This example demonstrates how to perform **Geodesic Centroidal Voronoi Tessellation (GCVT)** on 3D meshes using the ``Mesh-LBFGS`` second-order optimizer provided by this library.

.. figure:: /_static/gcvt_examples.png
    :align: center
    :width: 90%

    GCVT examples using the Mesh-LBFGS optimizer.

GCVT aims to partition a manifold :math:`\mathcal{M}` into :math:`S` regions :math:`\{\Omega_i\}_{i=1}^S` based on the distance to a set of seeds :math:`\mathcal{S} = \{s_i\}_{i=1}^S \in \mathcal{M}`. A tessellation is *centroidal* when each seed :math:`s_i` coincides with the **Karcher mean** (center of mass) of its respective Voronoi cell :math:`\Omega_i`.

Our implementation leverages **Mesh-LBFGS**, the second-order Riemannian optimizer designed specifically for 3D meshes, which utilizes our parallelized GPU exponential map and parallel transport operators.

Mathematical Formulation
------------------------

To find the optimal seed positions, we minimize the following energy function:

.. math::

   E(\mathcal{S}) = \frac{1}{2S} \sum_{i=1}^{S} \sum_{x \in \Omega_{i}} A(x)\rho(x)\|Log_{s_{i}}(x)\|^{2}

Where:

- **:math:`A(x)`**: The vertex area, defined as one-third of the sum of areas of triangles connected to vertex :math:`x`.

- **:math:`\rho(x)`**: A density function defined via the scalar heat kernel.

- **:math:`Log_{s_i}(x)`**: The Riemannian logarithm map finding the shortest vector from seed :math:`s_i` to vertex :math:`x`.

Optimization via Mesh-LBFGS
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We first define the loss function used for optimization, which computes the energy and its gradient with respect to the seed positions.

.. code::
    class LossGCVT(MeshLossFunc):




The full implementation of the GCVT example is available on the `GCVT GitHub repository <https://github.com/Etyl/GCVT>`_