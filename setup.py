from setuptools import setup, find_packages, Extension
import sys
import os

sampen_module = Extension(
    "pyhctsa.Toolboxes.physionet.libsampen",
    sources=["pyhctsa/Toolboxes/physionet/sampen.c"],
    extra_compile_args=["-O3", "-fPIC"],
    libraries=["m"] if not sys.platform.startswith("win") else [],
)

close_ret_module = Extension(
    "pyhctsa.Toolboxes.Max_Little.ML_close_ret",
    sources=["pyhctsa/Toolboxes/Max_Little/ML_close_ret.c"],
    extra_compile_args=["-O3", "-fPIC"],
    libraries=["m"] if not sys.platform.startswith("win") else [],
)

setup(
    name="pyhctsa",
    version="0.1",
    packages=find_packages(),
    ext_modules=[sampen_module, close_ret_module],
)