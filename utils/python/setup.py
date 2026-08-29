#!/usr/bin/env python
# coding=utf-8
"""llm_ops packaging."""

from pathlib import Path

import setuptools
from setuptools import find_namespace_packages
from setuptools.dist import Distribution
from configparser import ConfigParser

try:
    from torch_npu.utils.cpp_extension import NpuExtension
except ImportError:
    NpuExtension = None


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


WORKING_DIR = Path(__file__).resolve().parent
config = ConfigParser()
config.read(WORKING_DIR / "llm_ops" / "config.ini")
_version = config.get("global", "version")

ext_modules = []
if NpuExtension is not None:
    ext_modules = [NpuExtension("llm_ops._C", sources=[])]

setuptools.setup(
    name="llm_ops_ascend",
    version=_version,
    description="Python APIs for Ascend LLM operators (ACLNN wrappers over ops-transformer / CANN)",
    packages=find_namespace_packages(exclude=("tests*",)),
    package_data={"llm_ops": ["lib/**", "config.ini"]},
    include_package_data=True,
    ext_modules=ext_modules,
    distclass=BinaryDistribution,
    python_requires=">=3.8",
    license="BSD-3-Clause",
)
