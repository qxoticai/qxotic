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
    file = f"{name}.dll" if system == "windows" else f"lib{name}.dylib" if system == "darwin" else f"lib{name}.so"
    package = Path(sys.modules[name].__file__).resolve().parent
    override = os.environ.get(f"{name.upper()}_LIBRARY")
    # an override is the one candidate: a typo in it must not run another build in silence
    candidates = [Path(override)] if override else [package.parent.parent / "target" / file, package / "lib" / folder / file]
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

    try:
        lib.toknroll_version.restype = C.c_char_p
    except AttributeError:  # a shared library, not this one
        raise OSError(f"{lib._name} is not a toknroll library: it has no toknroll_version") from None
    found = lib.toknroll_version().decode()
    if found != __version__:  # ponytail: exact while the API is 0.x; same major once it is stable
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
        fn = getattr(lib, name)
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


@functools.cache
def library() -> C.CDLL:
    """libtoknroll, loaded on first use so that importing the package loads nothing. Its log lines go to
    logging.getLogger("toknroll"), as many as TOKNROLL_LOG lets through; libjinfer's, toknroll's face
    included, are jinfer's to route."""
    lib = bind(C.CDLL(str(locate("toknroll"))))
    # before any other call, so it holds; EINVAL only for a library some other loader in the process
    # started first, whose lines then go where that one said
    lib.toknroll_log_set(log_callback("toknroll"), None)
    return lib
