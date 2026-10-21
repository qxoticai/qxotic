"""ctypes bindings of include/toknroll.h. Nothing public lives here; see __init__.py."""
from __future__ import annotations

import ctypes as C
import functools
import os
import platform
from pathlib import Path

OK, ERROR, EINVAL, ENOTFOUND = 0, -1, -2, -3


def _platform() -> tuple[str, str]:
    """(platform folder, library file name), as `make python` lays the library out."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = {"x86_64": "x86-64", "amd64": "x86-64", "arm64": "aarch64"}.get(machine, machine)
    if system == "windows":
        return f"windows-{arch}", "toknroll.dll"
    if system == "darwin":
        return f"darwin-{arch}", "libtoknroll.dylib"
    return f"{system}-{arch}", "libtoknroll.so"


def _locate() -> Path:
    """TOKNROLL_LIBRARY, then a source checkout's fresh build, then the library the package carries."""
    folder, name = _platform()
    here = Path(__file__).resolve().parent
    override = os.environ.get("TOKNROLL_LIBRARY")
    candidates = [Path(override)] if override else []
    candidates += [here.parent.parent / "target" / name, here / "lib" / folder / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise OSError(
        f"libtoknroll not found for {folder}; looked at " + ", ".join(map(str, candidates))
        + ". Set TOKNROLL_LIBRARY to the shared library, or build it with `make libtoknroll`."
    )


def bind(lib: C.CDLL) -> C.CDLL:
    """Types the toknroll_* symbols of a loaded library: libtoknroll, or libjinfer carrying them."""
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
    return bind(C.CDLL(str(_locate())))
