.. _straightest_geodesic:

Straightest geodesic
====================

Tracing geodesics
-----------------

In this section, we see how to compute the straightest geodesic, also refered as the exponential map.
The method takes as input a mesh, a batch of points, and directions to trace the geodesics in.

.. code-block:: python

    from digeo import trace_geodesics, uniform_sampling, load_mesh_from_file
    from digeo.utils import length

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
------------------------

There exists different methods to compute the gradient of the geodesic tracing operation, which can be specified with the ``gradient`` argument of :func:`digeo.ops.trace_geodesics`:

``"none"``:
No gradient is computed.

``"ep"``:
Uses the extrinsic proxy method, which computes the gradient by parallel transporting the point along the geodesic and differentiating through the parallel transport.
This method is the most efficient, but can be less accurate than the following methods.

``"gfd"``:
Uses geodesic finite differences, which computes the gradient by finite differencing the geodesic tracing operation.
This method is more accurate than the extrinsic proxy method, but is also less efficient.

``"jvp"``:
Uses the Jacobian-vector product, which computes the gradient by backtracking through the geodesic tracing operation using the Jacobian-vector product.
This method lies in between the extrinsic proxy and geodesic finite differences methods in terms of accuracy and efficiency.


TODO: add plots comparing the different methods, and a more detailed explanation of how they work.


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