from setuptools import setup, Extension
import pybind11

from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="action_module",
    ext_modules=[
        CppExtension(
            name="action_module",
            sources=["search_action_space.cpp"],  # C++ source file
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension  # Use PyTorch's BuildExtension
    }
)
