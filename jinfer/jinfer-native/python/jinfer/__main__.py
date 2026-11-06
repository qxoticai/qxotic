"""python -m jinfer: where this install's headers and library are, for a C build, as python -m pybind11 --includes does.

    cc tokenize.c $(python -m jinfer --cflags) $(python -m jinfer --libs)
    PKG_CONFIG_PATH=$(python -m jinfer --pkgconfig) pkg-config --cflags --libs jinfer

The flags are the platform's compiler's: the include flag, then the library file itself and an rpath to it on Linux
and macOS, /I and /LIBPATH with the import library on Windows, one per line. Nothing else: the library is the API, jbang jinfer@qxoticai the tool."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from ._native import locate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jinfer", description="where this install's C headers and library are")
    parser.add_argument("--version", action="store_true", help="the package's version, the library's too")
    parser.add_argument("--cflags", action="store_true", help="the include flag")
    parser.add_argument("--libs", action="store_true", help="the link flags")
    parser.add_argument("--pkgconfig", action="store_true", help="the directory holding jinfer.pc, for PKG_CONFIG_PATH")
    args = parser.parse_args(argv)
    if not any(vars(args).values()):
        parser.print_help()
        return 2
    library = locate("jinfer")
    libdir, include = library.parent, Path(__file__).resolve().parent / "include"
    windows = sys.platform == "win32"
    if args.version:
        print(__version__)
    if args.cflags:
        print(f"/I{include}" if windows else f"-I{include}")
    if args.libs:
        print(f"/LIBPATH:{libdir}\njinfer.lib" if windows else f"{library} -Wl,-rpath,{libdir}")
    if args.pkgconfig:
        print(libdir / "pkgconfig")
    return 0


if __name__ == "__main__":
    sys.exit(main())
