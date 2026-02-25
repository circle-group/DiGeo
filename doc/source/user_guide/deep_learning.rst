Deep Learning Modules
=====================

Adaptive Geodesic Convolution (AGC)
-----------------------------------

The Adaptive Geodesic Convolution (AGC) module provides a way to perform convolution operations on meshes,
by computing geodesic patches around each vertex and applying a learnable filter to these patches.
The size of the patches is also learned during training, allowing the model to adapt to the local geometry of the mesh.

Biharmonic Distance
-------------------

The Biharmonic Distance module provides a way to compute the biharmonic distance between points on a mesh.
The biharmonic distance is a smooth distance function that takes into account the geometry of the mesh
and can be used for various applications such as shape analysis and mesh processing.