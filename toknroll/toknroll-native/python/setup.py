"""The wheel carries a C library and no extension module: any Python 3, any ABI, one platform. setuptools makes
a platform wheel only of a distribution with binaries, and tags it after the interpreter that builds it, so
both are said here: WHEEL_PLAT, from the Makefile, is the platform tag read off the library (its glibc floor,
its minimum macOS); without it, this machine's. Everything else is in pyproject.toml."""
import glob
import os

from setuptools import Distribution, setup

if not os.path.exists("LICENSE") or not glob.glob("toknroll/lib/*/*"):  # what `make python` stages
    raise SystemExit("the package is built from a staged tree: run `make python` in toknroll-native first")
from setuptools.command.bdist_wheel import bdist_wheel


class Binary(Distribution):
    def has_ext_modules(self):
        return True


class Wheel(bdist_wheel):
    def get_tag(self):
        return "py3", "none", os.environ.get("WHEEL_PLAT") or super().get_tag()[2]


setup(distclass=Binary, cmdclass={"bdist_wheel": Wheel})
