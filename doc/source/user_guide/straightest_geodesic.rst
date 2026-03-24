.. _straightest_geodesic:

Straightest geodesic
====================

In continuous Riemannian geometry, a geodesic is defined by having zero covariant acceleration, :math:`\nabla_{\dot{\gamma}(t)}\dot{\gamma}(t)=0`. On discrete 3D meshes, this concept is adapted using the angle preservation criteria. To maintain a straight path, the left and right surface angles must remain equal
(:math:`\theta_l = \theta_r`) for every point :math:`p` along the curve as it iteratively crosses faces, edges, and vertices.

.. figure:: /_static/angles.svg
    :align: center
    :width: 35%


Tracing geodesics
-----------------

In this section, we see how to compute the straightest geodesic, also refered as the exponential map.
The method takes as input a mesh, a batch of points, and directions to trace the geodesics in.

.. code-block:: python

    from digeo import load_mesh_from_file
    from digeo.ops import trace_geodesics, uniform_sampling

    # Load a mesh
    mesh = load_mesh_from_file("path/to/mesh.obj")

    # Sample points on the mesh
    points = uniform_sampling(mesh, num_points=100)

    # Random directions
    directions = torch.randn_like((100, 3))

    # Trace geodesics
    geodesics, _ = trace_geodesics(mesh, points, directions)

.. _differentiation_methods:

Differentiation methods
-----------------------

There exists different methods to compute the gradient of the geodesic tracing operation, which can be specified with the ``gradient`` argument of :func:`digeo.ops.trace_geodesics`:

**No gradient** (``"none"``)

No gradient is computed.


**Extrinsic Proxy** (``"ep"``)

.. figure:: /_static/EP_info.svg
    :align: center
    :width: 30%

The Extrinsic Proxy method computes the gradient by parallel transporting the point along the geodesic and differentiating through the parallel transport.
This method is the most efficient, but can be less accurate than the following methods.

**Geodesic Finite Differences** (``"gfd"``)

.. raw:: html

   <div style="display: flex; justify-content: space-around; align-items: center; gap: 20px; background: none; margin: 20px 0;">
       <div style="flex: 1; text-align: center;">
           <img src="../_static/Jp_info.svg" style="width: 60%;" alt="Jp info">
       </div>
       <div style="flex: 1; text-align: center;">
           <img src="../_static/Jv_info.svg" style="width: 60%;" alt="Jv info">
       </div>
   </div>

The Geodesic Finite Differences, method computes the gradient by finite differencing the geodesic tracing operation.
This method is more accurate than the extrinsic proxy method, but is also less efficient since it requires additional geodesic tracing operations to compute the gradient.

**Adjoint Backward Finite Differences** (``"abfd"``)

The Adjoint Backward Finite Differences method computes gradients by perturbing the endpoint and using parallel transport to map the perturbation back to the starting point.

.. warning::

    The ``"abfd"`` method is experimental and may cause undesirable behaviors when the mesh presents regions of high curvature.


Parallel transport
------------------

When tracing geodesics, it can be needed to parallel transport vectors along the geodesic. This can be done by settiing the ``save_parallel_transport`` argument of :func:`digeo.ops.trace_geodesics` to ``True``.
This will allow to parallel transport vectors along the geodesic, which can be useful for computing gradients or for other applications.

.. code-block:: python

    # Trace geodesics and return parallel transport
    x1, info = trace_geodesics(mesh, x0, directions, save_parallel_transport=True)

    # Random vectors to parallel transport
    v_rand = torch.randn_like(directions)
    # Parallel transport v_rand along the geodesic to get v1
    v1 = info.transport(v_rand)
     # Parallel transport v1 back to the original point to get v_rand
    v0 = info.transport_inv(v1)


Debugging
---------

When tracing geodesics, it can be useful to visualize the geodesics and the parallel transport. This can be done by setting the ``debug`` argument of :func:`digeo.ops.trace_geodesics` to ``True``. This will return additional information about the geodesic tracing operation, which can be used for debugging purposes.
This requires much more memory, since it saves the entire geodesic paths, as well as directions and normals.

.. code-block:: python

    # Trace geodesics with debugging information
    x1, info = trace_geodesics(mesh, x0, directions, debug=True)

    # Get the geodesic path for the k-th geodesic
    geodesic_path = info.get_path(k)
    # Get the directions along the k-th geodesic path
    directions_along_path = info.get_directions(k)
    # Get the normals along the k-th geodesic path
    normals_along_path = info.get_normals(k)