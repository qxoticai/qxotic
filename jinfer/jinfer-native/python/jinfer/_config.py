"""jinfer-config: the wheel's headers and library for a C build; the flags are toknroll's _native.config's."""
from toknroll._native import config


def main() -> int:
    return config("jinfer")
