import os
import glob
import torch

from setuptools import setup
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "digeo"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and CUDA_HOME is not None
    print(f"Using CUDA: {use_cuda}")

    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"] += ["-g", "-DTORCH_USE_CUDA_DSA"]
        extra_compile_args["nvcc"] += ["-g", "-G", "-lineinfo", "-DTORCH_USE_CUDA_DSA"]
        extra_link_args += ["-O0", "-g"]

    extensions_dir = "src/digeo/ops/cuda"
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}/ops/cuda/._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules

setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)