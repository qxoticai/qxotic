"""ctypes bindings of include/jinfer.h. Nothing public lives here; see __init__.py."""
from __future__ import annotations

import ctypes as C
import os
import platform
from pathlib import Path

OK, ERROR, EINVAL, ENOTFOUND, EUNSUPPORTED, ECANCELLED = 0, -1, -2, -3, -4, -5

LogFn = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)


class ModelOptions(C.Structure):
    """jinfer_model_options; struct_size guards the layout, as the header says."""

    _fields_ = [
        ("struct_size", C.c_size_t),
        ("companions", C.POINTER(C.c_char_p)),
        ("companion_count", C.c_int32),
        ("prompt_cache", C.c_char_p),
        ("context_capacity", C.c_int32),
    ]


def _platform() -> tuple[str, str]:
    """(platform folder, library file name), as `make python` lays the library out."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = {"x86_64": "x86-64", "amd64": "x86-64", "arm64": "aarch64"}.get(machine, machine)
    if system == "windows":
        return f"windows-{arch}", "jinfer.dll"
    if system == "darwin":
        return f"darwin-{arch}", "libjinfer.dylib"
    return f"{system}-{arch}", "libjinfer.so"


def _locate() -> Path:
    """JINFER_LIBRARY, then a source checkout's fresh build, then the library the package carries."""
    folder, name = _platform()
    here = Path(__file__).resolve().parent
    override = os.environ.get("JINFER_LIBRARY")
    candidates = [Path(override)] if override else []
    candidates += [here.parent.parent / "target" / name, here / "lib" / folder / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise OSError(
        f"libjinfer not found for {folder}; looked at " + ", ".join(map(str, candidates))
        + ". Set JINFER_LIBRARY to the shared library, or build it with `make libjinfer`."
    )


def bind(lib: C.CDLL) -> C.CDLL:
    """Types the jinfer_* symbols of the loaded library."""
    P, S, I32, F = C.c_void_p, C.c_char_p, C.c_int32, C.POINTER(C.c_float)
    sig = {
        "jinfer_version": ([], S),
        "jinfer_last_error": ([C.POINTER(I32)], S),
        "jinfer_log_set": ([LogFn, P], None),
        "jinfer_configure": ([C.POINTER(C.c_char_p), I32], I32),
        "jinfer_model_options_init": ([C.POINTER(ModelOptions)], None),
        "jinfer_model_load": ([S, C.POINTER(ModelOptions)], P),
        "jinfer_model_free": ([P], None),
        "jinfer_model_property": ([P, C.c_int], I32),
        "jinfer_model_info": ([P], S),
        "jinfer_model_tokenizer": ([P], P),
        "jinfer_state_new": ([P, I32], P),
        "jinfer_state_free": ([P], None),
        "jinfer_state_reset": ([P], I32),
        "jinfer_state_position": ([P], I32),
        "jinfer_state_ingest": ([P, C.POINTER(I32), I32], I32),
        "jinfer_state_logits": ([P, F, I32], I32),
        "jinfer_state_embedding": ([P, F, I32], I32),
        "jinfer_stream_step": ([P], I32),
        "jinfer_stream_text": ([P], S),
        "jinfer_stream_samples": ([P, C.POINTER(I32)], F),
        "jinfer_stream_result": ([P], S),
        "jinfer_stream_cancel": ([P], None),
        "jinfer_stream_free": ([P], None),
        "jinfer_chat": ([P, S], P),
        "jinfer_speak": ([P, S], P),
        "jinfer_transcribe": ([P, F, I32, I32], P),
        "jinfer_embed": ([P, C.POINTER(S), I32, I32, F, I32], I32),
        "jinfer_rerank": ([P, S, C.POINTER(S), I32, S, F], I32),
        "jinfer_prepare": ([P, S], I32),
    }
    for name, (argtypes, restype) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = argtypes, restype
    return lib


lib = bind(C.CDLL(str(_locate())))
