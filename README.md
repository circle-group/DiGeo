
[![PyPI version](https://badge.fury.io/py/digeo.svg)](https://badge.fury.io/py/digeo)
[![Python Versions](https://img.shields.io/pypi/pyversions/digeo.svg)](https://pypi.org/project/digeo/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Tests](https://github.com/circle-group/digeo/actions/workflows/tests.yml/badge.svg)](https://github.com/circle-group/digeo/actions/workflows/tests.yml)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/circle-group/DiGeo/56d3663e6778e88b8d4b10e0b5991990e3d516fb/doc/source/_static/digeo_dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/circle-group/DiGeo/56d3663e6778e88b8d4b10e0b5991990e3d516fb/doc/source/_static/digeo.svg" />
    <img src="https://raw.githubusercontent.com/circle-group/DiGeo/56d3663e6778e88b8d4b10e0b5991990e3d516fb/doc/source/_static/digeo.svg" height="125" alt="DiGeo logo" />
  </picture>
</p>

**DiGeo** (Differentiable Geometry) is a Python package designed to enable differential geometry in learning and optimisation applications on triangular meshes.

Built on PyTorch and custom CUDA kernels, DiGeo provides differentiable exponential map, parallel transport, and geodesic tracing as core operations, which are leveraged across the higher-level modules and examples in the package. It supports batched inputs, single and double precision, and runs on both CPU and GPU.

## Key Features

* **Differentiable Geometry:** Differentiable exponential map, parallel transport, and geodesic tracing, implemented with highly optimized C++ and CUDA kernels.
* **Batched Operations:** Process multiple meshes and points simultaneously using `MeshBatch` and `MeshPointBatch`.
* **Mesh Optimization:** Built-in Riemannian optimization algorithms including Gradient Descent (`mesh_gd`) and L-BFGS (`mesh_lbfgs`).
* **Geometric Deep Learning:** Includes neural network modules like the Adaptive Geodesic Convolutional Layer (`AGC`) and Biharmonic Distance.


## Installation

### Dependencies

Before installing DiGeo, ensure you have a compatible python version, the necessary libraries will be installed automatically via pip:

* **Python:** $\ge$ 3.10
* **Libraries:** `Pytorch`, `NumPy`, `tqdm`, `SciPy`, `Trimesh`, `Robust Laplacian`

### Standard Installation (via pip)

The easiest way to install the latest stable release is through PyPI. The wheels are precompiled using **Pytorch 2.10 and CUDA 12.8**.

```bash
pip install torch==2.10 --index-url https://download.pytorch.org/whl/cu128
pip install digeo
```

> **Compatibility Note:** If you are using another version of PyTorch or CUDA, you will need to build from source to ensure binary compatibility.

### Platform & Hardware Support

DiGeo utilizes custom CUDA kernels. Please note the following hardware limitations for the `pip` installation:

* **Linux (x86_64) and Windows (ARM64):** Includes pre-compiled CUDA kernels.
* **Linux (ARM64) and macOS:** `pip` will default to a **CPU-only** version. For GPU support, you will need to build the package from source.
* **For other platforms or architectures:** You must build the package from source.

### Install from Source

**Requirements:** A working **C++ compiler** and the **NVIDIA CUDA Toolkit**.

To install version X.Y.Z of DiGeo from source:
```bash
pip install "digeo @ git+ssh://git@github.com/circle-group/DiGeo.git@vX.Y.Z" --no-build-isolation
```

For example, to install version 1.2.3:
```bash
pip install "digeo @ git+ssh://git@github.com/circle-group/DiGeo.git@v1.2.3" --no-build-isolation
```

## Examples

You can find some applications using DiGeo on the [DSG-Applications repository](https://github.com/circle-group/DSG-Applications). These applications make use of the differentiable straightest geodesics and parallel transport provided by DiGeo, as well as the Biharmonic Distance, AGC layers, and mesh optimisers.

## Citing

If you use DiGeo in your research, please consider citing the following paper:

```
@inproceedings{verninas2026disgeod,
  title={Parallelised Differentiable Straightest Geodesics for 3D Meshes},
  author={Verninas, Hippolyte and Korkmaz, Caner and Zafeiriou, Stefanos and Birdal, Tolga and Foti, Simone},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```


## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

