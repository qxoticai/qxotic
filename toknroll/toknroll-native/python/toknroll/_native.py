"""ctypes bindings of include/toknroll.h. Nothing public lives here; see __init__.py.

ctypes on purpose: portability over per-call speed. The stdlib and no compiler, so one py3-none wheel per
platform runs on every Python version, free-threaded builds included. The cost is a fixed overhead per call,
about a microsecond, which a compiled extension over toknroll.h would remove at the price of that portability."""
from __future__ import annotations

import _thread
import ctypes as C
import functools
import logging
import os
import platform
import sys
from pathlib import Path

OK, ERROR, EINVAL, ENOTFOUND, EUNSUPPORTED = 0, -1, -2, -3, -4


def locate(name: str) -> Path:
    """The shared library called `name` (toknroll, jinfer): `<NAME>_LIBRARY` alone when it is set, else a source
    checkout's fresh build, then the library the package carries, laid out per platform as `make python` does."""
    system, machine = platform.system().lower(), platform.machine().lower()
    arch = {"x86_64": "x86-64", "amd64": "x86-64", "arm64": "aarch64"}.get(machine, machine)
    folder = f"{system}-{arch}"
    package = Path(sys.modules[name].__path__[0]).resolve()  # the package's directory
    built = f"{name}.dll" if system == "windows" else f"lib{name}.dylib" if system == "darwin" else f"lib{name}.so"
    # the package's copy is named by its soname, lib<name>.so.<version>, the name a C program linked against it loads
    shipped = f"{name}.dll" if system == "windows" else f"lib{name}.*.dylib" if system == "darwin" else f"lib{name}.so.*"
    override = os.environ.get(f"{name.upper()}_LIBRARY")
    # an override is the one candidate: a typo in it must not run another build in silence
    looked = [Path(override)] if override else [package.parent.parent / "target" / built, package / "lib" / folder / shipped]
    for candidate in looked if override else [looked[0], *sorted(looked[1].parent.glob(shipped))]:
        if candidate.is_file():
            return candidate
    raise OSError(
        f"lib{name} not found for {folder}; looked at " + ", ".join(map(str, looked))
        + f". Set {name.upper()}_LIBRARY to the shared library, or build it with `make lib{name}` (a `mvn clean` deletes it)."
    )


def bind(lib: C.CDLL) -> C.CDLL:
    """Types the toknroll_* symbols of a loaded library: libtoknroll, or libjinfer carrying them. The
    library's face must be this layer's toknroll: a function it lacks would surface as a missing symbol,
    a changed signature as corruption, so the versions are compared first."""
    from . import __version__  # at call time, once the package is complete

    try:
        lib.toknroll_version.restype = C.c_char_p
    except AttributeError:  # a shared library, not this one
        raise OSError(f"{lib._name} is not a toknroll library: it has no toknroll_version") from None
    found = lib.toknroll_version().decode()
    # the library spells a release candidate 0.4.0-rcN, Python 0.4.0rcN
    if found.replace("-", "") != __version__:  # ponytail: exact while the API is 0.x; same major once it is stable
        raise OSError(f"toknroll {__version__}'s Python layer cannot drive the toknroll {found} in {lib._name}")
    P, S, I32, I64 = C.c_void_p, C.c_char_p, C.c_int32, C.c_int64
    sig = {
        "toknroll_version": ([], S),
        "toknroll_last_error": ([C.POINTER(I32)], S),
        "toknroll_log_set": ([LogFn, P], I32),
        "toknroll_configure": ([C.POINTER(C.c_char_p), I32], I32),
        "toknroll_tokenizer_load": ([S], P),
        "toknroll_tokenizer_free": ([P], None),
        # the id buffers travel as addresses: an array('i') or a numpy array, in place
        "toknroll_encode": ([P, S, I64, C.c_uint32, P, I32], I32),
        "toknroll_decode": ([P, P, I32, I32, C.c_char_p, I64, P], I64),
        "toknroll_encode_batch": ([P, C.c_char_p, P, I32, C.c_uint32, P, I32, P], I32),
        "toknroll_vocabulary_size": ([P], I32),
        "toknroll_token_flags": ([P, I32], I64),
        "toknroll_token": ([P, I32, C.c_char_p, I64], I64),
        "toknroll_id": ([P, S, I64], I32),
    }
    for name, (argtypes, restype) in sig.items():
        try:
            fn = getattr(lib, name)
        except AttributeError:  # the version matched, the symbol table does not: a build that is not whole
            raise OSError(f"{lib._name} says toknroll {found} but has no {name}") from None
        fn.argtypes, fn.restype = argtypes, restype
    return lib


LogFn = C.CFUNCTYPE(None, C.c_int32, C.c_char_p, C.c_void_p)
_LEVELS = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING, 3: logging.ERROR}
_callbacks = globals().get("_callbacks", {})   # a reload keeps them: the library still calls them


def log_callback(name: str):
    """The C log callback that hands each line to logging.getLogger(name): made once per name and kept
    for the life of the process, as the library keeps it."""
    if name not in _callbacks:
        logger = logging.getLogger(name)

        def forward(level, message, _):
            try:
                logger.log(_LEVELS.get(level, logging.INFO), "%s", message.decode("utf-8", "replace"))
            except KeyboardInterrupt:  # Ctrl-C landed in the callback, where ctypes would drop it
                _thread.interrupt_main()

        _callbacks[name] = LogFn(forward)
    return _callbacks[name]


def open_library(path) -> C.CDLL:
    """ctypes.CDLL, with glibc's refusal of a CPU below the library's x86-64 level (2.33 and later check it before
    any of our code runs) given the requirement; the runtime says the same itself under an older glibc."""
    try:
        return C.CDLL(str(path))
    except OSError as e:
        if "ISA level" not in str(e):
            raise
        raise OSError(f"{e}: the library needs an x86-64-v3 CPU, with AVX2, FMA and BMI2; in a virtual machine, "
                      "pass the host's CPU through (QEMU: -cpu host)") from None


@functools.cache
def library() -> C.CDLL:
    """libtoknroll, loaded on first use so that importing the package loads nothing. Its log lines go to
    logging.getLogger("toknroll"), as many as TOKNROLL_LOG lets through; libjinfer's, toknroll's face
    included, are jinfer's to route."""
    lib = bind(open_library(locate("toknroll")))
    # before any other call, so it holds; EINVAL only for a library some other loader in the process
    # started first, whose lines then go where that one said
    lib.toknroll_log_set(log_callback("toknroll"), None)
    return lib


def config(name: str, argv: list[str] | None = None) -> int:
    """<name>-config, the C world's own convention (python3-config, numpy-config, pybind11-config): where this
    install's headers and library are, one complete compiler argument per line, without shell quoting.
    Preserve each line as one argument, including any spaces in a path:

        import subprocess
        flags = subprocess.check_output(["toknroll-config", "--cflags", "--libs"], text=True).splitlines()
        subprocess.run(["cc", "tokenize.c", *flags], check=True)

    Nothing else: the library is the API, jbang <name>@qxoticai the tool."""
    import argparse
    import importlib

    parser = argparse.ArgumentParser(prog=f"{name}-config", description=f"where this install's C headers and lib{name} are",
                                     epilog="Compiler flags are one argument per line, without shell quoting; preserve spaces within each line.")
    parser.add_argument("--version", action="store_true", help="the package's version, the library's too")
    parser.add_argument("--cflags", action="store_true", help="the include flag")
    parser.add_argument("--libs", action="store_true", help="the link flags")
    parser.add_argument("--pkgconfigdir", action="store_true", help=f"the directory holding {name}.pc, for PKG_CONFIG_PATH")
    args = parser.parse_args(argv)
    if not any(vars(args).values()):
        parser.print_help()
        return 2
    module = importlib.import_module(name)
    library = locate(name)
    libdir, include = library.parent, Path(module.__path__[0]).resolve() / "include"
    windows = sys.platform == "win32"
    if args.version:
        print(module.__version__)
    if args.cflags:
        print(f"/I{include}" if windows else f"-I{include}")
    if args.libs:
        print(f"/LIBPATH:{libdir}\n{name}.lib" if windows else f"{library}\n-Wl,-rpath,{libdir}")
    if args.pkgconfigdir:
        print(libdir / "pkgconfig")
    return 0
