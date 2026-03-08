Deep Learning Modules
=====================

.. _agc:

Adaptive Geodesic Convolution (AGC)
-----------------------------------

The Adaptive Geodesic Convolution (AGC) module provides a way to perform convolution operations on meshes, by computing geodesic patches around each vertex and applying a learnable filter to these patches. The size of the patches is also learned during training, allowing the model to adapt to the local geometry of the mesh.


.. _biharmonic_distance:

Biharmonic Distance
-------------------

This module calculates the biharmonic distance between points on a 3D mesh. Unlike standard geodesic pathways, biharmonic distance provides a smooth, geometry-aware distance metric that captures the global shape of the mesh.

.. note::

   **Reference:**
   This module is a direct implementation of the methodology detailed in
   `Biharmonic Distance <https://dl.acm.org/doi/10.1145/1805964.1805971>`_
   by Yaron Lipman, Raif M. Rustamov, and Thomas A. Funkhouser
   (ACM Transactions on Graphics, 2010).