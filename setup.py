import os
import glob
import sys

from setuptools import setup
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "digeo"


def get_extensions():
    is_windows = sys.platform == "win32"
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"

    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and CUDA_HOME is not None
    print(f"Using CUDA: {use_cuda}")

    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {"cxx": [], "nvcc": []}

    if is_windows:
        # MSVC Flags
        if debug_mode:
            extra_compile_args["cxx"] = ["/Od", "/Z7", "-DTORCH_USE_CUDA_DSA"]
            extra_link_args = ["/DEBUG"]
        else:
            extra_compile_args["cxx"] = ["/O2"]
    else:
        # GCC / Clang Flags
        if debug_mode:
            extra_compile_args["cxx"] = [
                "-O0",
                "-g",
                "-fdiagnostics-color=always",
                "-DTORCH_USE_CUDA_DSA",
            ]
            extra_link_args = ["-O0", "-g"]
        else:
            extra_compile_args["cxx"] = ["-O3", "-fdiagnostics-color=always"]

    # Device NVCC Compiler Flags
    extra_compile_args["nvcc"] = [
        "-O0" if debug_mode else "-O3",
        "--fmad=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    ]

    if debug_mode:
        extra_compile_args["nvcc"] += ["-g", "-G", "-lineinfo", "-DTORCH_USE_CUDA_DSA"]

    if is_windows:
        extra_compile_args["nvcc"] += ["-D_WIN32=1"]

    # --- 3. Sources ---
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
            py_limited_api=False,
        )
    ]

    return ext_modules


setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
