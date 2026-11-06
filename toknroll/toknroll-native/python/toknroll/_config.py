"""toknroll-config: the wheel's headers and library for a C build; the flags are _native.config's."""
from ._native import config


def main() -> int:
    return config("toknroll")
