:html_theme.sidebar_secondary.remove:


DiGeo
=====

**DiGeo** (Differentiable Geometry) is a Python package for differential geometry in learning and optimisation applications on triangular meshes. Built on PyTorch and custom CUDA kernels, it provides differentiable exponential maps, parallel transport, and geodesic tracing as core operations. DiGeo also features high-level tools including geodesic convolutions, biharmonic distance, and Riemannian optimisers (gradient descent and L-BFGS). It supports batched inputs, single and double precision, and runs on both CPU and GPU.

.. toctree::
   :hidden:

   install
   user_guide/index
   api
   examples/index
   whats_new


Examples
--------

.. grid:: 1 1 3 3
   :gutter: 4
   :margin: 3 3 0 0

   .. grid-item-card:: Voronoi Tessellation
      :link: examples/gcvt
      :link-type: doc

      Partition complex meshes with Geodesic Centroidal Voronoi Tessellation
      and the Mesh-LBFGS optimizer.

      .. image:: /_static/gcvt_examples.png
         :class: example-thumb
         :alt: GCVT tessellations

   .. grid-item-card:: MeshFlow
      :link: examples/meshflow
      :link-type: doc

      Learn how MeshFlow trains a stationary vector field with differentiable
      exponential maps and biharmonic losses.

      .. image:: /_static/duck_distrib.svg
         :class: example-thumb
         :alt: MeshFlow vector field

   .. grid-item-card:: Shape Segmentation
      :link: examples/segmentation
      :link-type: doc

      Explore the AGC U-ResNet architecture for dense vertex labeling on human
      meshes.

      .. image:: /_static/agc_seg.png
         :class: example-thumb
         :alt: AGC segmentation samples


Citing DiGeo
------------

If you use DiGeo in your research, please consider citing the following paper:

.. code-block:: bibtex

   @inproceedings{verninas2026disgeod,
      title={Parallelised Differentiable Straightest Geodesics for 3D Meshes},
      author={Verninas, Hippolyte and Korkmaz, Caner and Zafeiriou, Stefanos and Birdal, Tolga and Foti, Simone},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2026}
   }