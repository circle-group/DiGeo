:html_theme.sidebar_primary.remove:

Installation
============

Dependencies
------------

Before installing ``DiGeo``, ensure you have a compatible Python version.
The necessary libraries will be installed automatically via ``pip``.

- **Python:** >= 3.10
- **Libraries:** ``Pytorch``, ``NumPy``, ``tqdm``, ``SciPy``, ``Trimesh``, ``Robust Laplacian``


Standard Installation (via pip)
--------------------------------

The easiest way to install the latest stable release is through PyPI. The wheels are precompiled using Pytorch 2.10 and CUDA 12.8.

.. code-block:: bash

   pip install torch==2.10 --index-url https://download.pytorch.org/whl/cu128
   pip install digeo

.. important::

   **Compatibility Note:** If you are using another version of PyTorch or CUDA, you will need to build from source to ensure binary compatibility.


Platform & Hardware Support
---------------------------

``DiGeo`` utilizes custom CUDA kernels. Please note the following hardware
limitations for the ``pip`` installation:

- **Linux (x86_64) and Windows (ARM64):** Includes pre-compiled CUDA kernels.
- **Linux (ARM64) and macOS:** ``pip`` will default to a **CPU-only** version. For GPU support, you will need to build the package from source.
- **Other platforms or architectures:** You must build the package from source.


Install from Source
-------------------

**Requirements:** A working **C++ compiler** and the **NVIDIA CUDA Toolkit**.

To install version x.y.z of DiGeo from source:

.. code-block:: bash

   pip install "digeo @ git+ssh://git@github.com/circle-group/DiGeo.git@x.y.z" --no-build-isolation