:html_theme.sidebar_secondary.remove:


DiGeo
=====

.. toctree::
   :hidden:

   install
   user_guide/index
   api
   examples/index
   whats_new


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
