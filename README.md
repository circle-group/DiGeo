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
* Python >= 3.10
* PyTorch >= 2.0 *(Note: Must be installed with CUDA support if using a GPU)*
* NumPy, tqdm, SciPy, Trimesh, Robust Laplacian

### Install via pip

You can install the latest release directly from PyPI:

```bash
pip install digeo
```

*Note: Pre-compiled CUDA kernels are currently included for Linux x86_64. If you are installing on macOS, Windows, or another architecture, `pip` will automatically fall back to the CPU-only version. If you require CUDA, you must build from source.*

### Install from source

You will need to build the package from source if you want to use the latest development version or if you need to compile the CUDA kernels specifically for your local GPU architecture.

**Prerequisites:** Building from source requires a working C++ compiler and the NVIDIA CUDA Toolkit.

```bash
pip install --no-build-isolation -e digeo@"git+ssh://git@github.com/circle-group/DiGeo.git"
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

