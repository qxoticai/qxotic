"""ctypes bindings of include/toknroll.h. Nothing public lives here; see __init__.py."""
from __future__ import annotations

import ctypes as C
import functools
import os
import platform
import sys
from pathlib import Path

OK, ERROR, EINVAL, ENOTFOUND = 0, -1, -2, -3


def locate(name: str) -> Path:
    """The shared library called `name` (toknroll, jinfer): `<NAME>_LIBRARY`, then a source checkout's fresh
    build, then the library the package carries, laid out per platform as `make python` does."""
    system, machine = platform.system().lower(), platform.machine().lower()
    arch = {"x86_64": "x86-64", "amd64": "x86-64", "arm64": "aarch64"}.get(machine, machine)
    folder = f"{system}-{arch}"
    file = f"{name}.dll" if system == "windows" else f"lib{name}.dylib" if system == "darwin" else f"lib{name}.so"
    package = Path(sys.modules[name].__file__).resolve().parent
    override = os.environ.get(f"{name.upper()}_LIBRARY")
    candidates = [Path(override)] if override else []
    candidates += [package.parent.parent / "target" / file, package / "lib" / folder / file]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise OSError(
        f"lib{name} not found for {folder}; looked at " + ", ".join(map(str, candidates))
        + f". Set {name.upper()}_LIBRARY to the shared library, or build it with `make lib{name}`."
    )


def bind(lib: C.CDLL) -> C.CDLL:
    """Types the toknroll_* symbols of a loaded library: libtoknroll, or libjinfer carrying them. The
    library's face must be this layer's toknroll: a function it lacks would surface as a missing symbol,
    a changed signature as corruption, so the versions are compared first."""
    from . import __version__  # at call time, once the package is complete

    lib.toknroll_version.restype = C.c_char_p
    found = lib.toknroll_version().decode()
    if found != __version__:  # ponytail: exact while the API is 0.x; same major once it is stable
        raise OSError(f"toknroll {__version__}'s Python layer cannot drive the toknroll {found} in {lib._name}")
    P, S, I32 = C.c_void_p, C.c_char_p, C.c_int32
    sig = {
        "toknroll_version": ([], S),
        "toknroll_last_error": ([C.POINTER(I32)], S),
        "toknroll_configure": ([C.POINTER(C.c_char_p), I32], I32),
        "toknroll_load": ([S], P),
        "toknroll_free": ([P], None),
        # the id buffers travel as addresses: an array('i') or a numpy array, in place
        "toknroll_encode": ([P, S, C.c_bool, P, I32], I32),
        "toknroll_decode": ([P, P, I32, C.c_char_p, I32], I32),
        "toknroll_encode_batch": ([P, C.c_char_p, P, I32, C.c_bool, P, I32, P], I32),
        "toknroll_vocabulary_size": ([P], I32),
        "toknroll_token_flags": ([P, I32], C.c_int64),
        "toknroll_token": ([P, I32, C.c_char_p, I32], I32),
        "toknroll_id": ([P, S], I32),
    }
    for name, (argtypes, restype) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = argtypes, restype
    return lib


@functools.cache
def library() -> C.CDLL:
    """libtoknroll, loaded on first use so that importing the package loads nothing."""
    return bind(C.CDLL(str(locate("toknroll"))))
