"""python -m toknroll, and python -m jinfer through it: where this install's headers and library are, for a C build, as
python -m pybind11 --includes does.

    cc tokenize.c $(python -m toknroll --cflags) $(python -m toknroll --libs)
    PKG_CONFIG_PATH=$(python -m toknroll --pkgconfig) pkg-config --cflags --libs toknroll

The flags are the platform's compiler's: the include flag, then the library file itself and an rpath to it on Linux
and macOS, /I and /LIBPATH with the import library on Windows, one per line. Nothing else: the library is the API, jbang toknroll@qxoticai the tool."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._native import locate


def main(name: str = "toknroll", argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {name}", description=f"where this install's C header{'s' if name == 'jinfer' else ''} and library are")
    parser.add_argument("--version", action="store_true", help="the package's version, the library's too")
    parser.add_argument("--cflags", action="store_true", help="the include flag")
    parser.add_argument("--libs", action="store_true", help="the link flags")
    parser.add_argument("--pkgconfig", action="store_true", help=f"the directory holding {name}.pc, for PKG_CONFIG_PATH")
    args = parser.parse_args(argv)
    if not any(vars(args).values()):
        parser.print_help()
        return 2
    import importlib

    library = locate(name)
    libdir, include = library.parent, Path(importlib.import_module(name).__file__).resolve().parent / "include"
    windows = sys.platform == "win32"
    if args.version:
        print(importlib.import_module(name).__version__)
    if args.cflags:
        print(f"/I{include}" if windows else f"-I{include}")
    if args.libs:
        print(f"/LIBPATH:{libdir}\n{name}.lib" if windows else f"{library} -Wl,-rpath,{libdir}")
    if args.pkgconfig:
        print(libdir / "pkgconfig")
    return 0


if __name__ == "__main__":
    sys.exit(main())
