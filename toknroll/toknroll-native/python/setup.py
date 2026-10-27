"""The package carries a C library, so its wheel is a platform wheel (Root-Is-Purelib: false), which setuptools
only concludes from a distribution that says it has binaries. Everything else is in pyproject.toml."""
from setuptools import Distribution, setup


class Binary(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=Binary)
