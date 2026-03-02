# DiGeo

[![PyPI version](https://badge.fury.io/py/digeo.svg)](https://badge.fury.io/py/digeo)
[![Python Versions](https://img.shields.io/pypi/pyversions/digeo.svg)](https://pypi.org/project/digeo/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Tests](https://github.com/circle-group/digeo/actions/workflows/tests.yml/badge.svg)](https://github.com/circle-group/digeo/actions/workflows/tests.yml)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**DiGeo** (Differentiable Geometry) is a Python package designed to trace differentiable geodesics (compute the exponential map) on triangulated meshes.

Implemented with PyTorch and custom CUDA kernels, DiGeo is built for efficiency and scalability. It supports batched inputs, single and double-precision floating-point numbers, and seamless execution on both CPU and GPU.

<p align="center">
    <img src="assets/geodesic_traces_bunny.png" height="400" alt="Geodesic traces on a bunny mesh"/>
    <!-- TODO: change this to url -->
</p>

## Key Features

* **Fast Geodesic Tracing:** Highly optimized C++ and CUDA kernels for tracing straightest geodesics on meshes.
* **Differentiable:** Compute gradients with respect to start points and directions.
* **Batched Operations:** Process multiple meshes and points simultaneously using `MeshBatch` and `MeshPointBatch`.
* **Mesh Optimization:** Built-in Riemannian optimization algorithms including Gradient Descent (`mesh_gd`) and L-BFGS (`mesh_lbfgs`).
* **Geometric Deep Learning:** Includes neural network modules like the Adaptive Geodesic Convolutional Layer (`AGC`) and Biharmonic Distance.

## Installation

### Dependencies

Before installing `DiGeo`, ensure you have a compatible python version, the necessary libraries will be installed automatically via pip:

* **Python:** $\ge$ 3.10
* **PyTorch:** $\ge$ 2.0 (Must be installed with **CUDA support** for GPU acceleration)
* **Libraries:** `NumPy`, `tqdm`, `SciPy`, `Trimesh`, `Robust Laplacian`

### Standard Installation (via pip)

The easiest way to install the latest stable release is through PyPI.

```bash
pip install digeo
```

> **Compatibility Note:** The PyPI version is compiled against the latest stable PyTorch release and CUDA 12.8. If you are using a specific or older version of PyTorch, we recommend building from source to ensure binary compatibility.


### Platform & Hardware Support

`DiGeo` utilizes custom CUDA kernels. Please note the following hardware limitations for the `pip` installation:

* **Linux (x86_64) and Windows (ARM64):** Includes pre-compiled CUDA kernels.
* **Linux (ARM64) and macOS:** `pip` will default to a **CPU-only** version. For GPU support, you will need to build the package from source.
* **For other platforms or architectures:** You must build the package from source.


### Install from Source

**Requirements:** A working **C++ compiler** and the **NVIDIA CUDA Toolkit**.

To install version x.y.z of DiGeo from source:
```bash
pip install --no-build-isolation -e "digeo @ git+ssh://git@github.com/circle-group/DiGeo.git@x.y.z"
```

## Development

To contribute or modify the package, clone the repository and install the development dependencies:

```bash
git clone https://github.com/circle-group/DiGeo.git
cd digeo
pip install --no-build-isolation -e .[test]
```

Run the test suite using `pytest`:
```bash
pytest -v
```

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

