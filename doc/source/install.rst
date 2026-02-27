:html_theme.sidebar_primary.remove:

Install
=======

Dependencies
------------
* Python >= 3.10
* PyTorch >= 2.0 *(Note: Must be installed with CUDA support if using a GPU)*
* NumPy, tqdm, SciPy, Trimesh, Robust Laplacian

Install via pip
---------------

You can install the latest release directly from PyPI:

.. code-block:: bash

    pip install digeo

*Note: Pre-compiled CUDA kernels are currently included for Linux x86_64. If you are installing on macOS, Windows, or another architecture,* ``pip`` *will automatically fall back to the CPU-only version. If you require CUDA, you must build from source.*

Install from source
-------------------

You will need to build the package from source if you want to use the latest development version or if you need to compile the CUDA kernels specifically for your local GPU architecture.

**Prerequisites:** Building from source requires a working C++ compiler and the NVIDIA CUDA Toolkit.

.. code-block:: bash

    pip install --no-build-isolation -e digeo@"git+ssh://git@github.com/circle-group/DiGeo.git"