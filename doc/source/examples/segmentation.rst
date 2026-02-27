Shape segmentation
==================

.. figure:: /_static/agc_seg.png
    :align: center
    :width: 50%

    Segmentation examples using the AGC U-ResNet model.

The AGC U-ResNet model makes use of the Adaptive Geodesic Convolution (AGC) module to perform shape segmentation on meshes. In this example, we train the model on a dataset of 3D human shapes and predict the different body parts (e.g., head, torso, arms, legs) for each vertex of the mesh. The model takes as input the 3D coordinates of the vertices, or the Heat Kernel Signature (HKS) features, and outputs a segmentation label for each vertex.


Architecture
------------

The AGC U-ResNet architecture is a U-shaped ResNet that incorporates the AGC module in its convolutional layers. For the pooling layer, we subsampled the meshes using quadric decimation and targeted around a quarter of the faces of the mesh. Pooling and un-pooling are performed by matrix multiplying vertex features with sparse matrices obtained during quadric decimation, a common practice in many SotA methods. We also doubled the number of filters used on the subsampled meshes (in the stacks 2 and 3).

.. figure:: /_static/agc_unet.svg
    :align: center

    The main architecture of the AGC U-ResNet.

The ResNet stacks used in the AGC U-ResNet architecture consist of two AGC blocks, which are composed of two AGC layers with a skip connection.

.. figure:: /_static/agc_block.svg
    :align: center
    :width: 50%

    The ResNet block used in the AGC U-ResNet architecture.

The full implementation of the AGC U-ResNet architecture is available on the `AGC GitHub repository <https://github.com/Etyl/AGC>`_