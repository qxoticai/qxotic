"""Token-perfect LLM tokenization from Python, over libtoknroll.

    import toknroll

    tokenizer = toknroll.load("Qwen/Qwen3-0.6B")        # tokenizer.json from the hub, metadata only
    ids = tokenizer.encode("Hello, world!")              # array('i', [9707, 11, 1879, 0])
    tokenizer.decode(ids)                                # 'Hello, world!'
    tokenizer.count_tokens(text)                         # the budget, without the list
    tokenizer.encode_batch(texts)                        # on threads; the library releases the GIL
    tokenizer.encode("<|im_end|>", specials=True)        # spelled special tokens stay whole
    tokenizer.decode_bytes(ids)                          # the tokens' bytes, even half a character
    tokenizer.vocabulary_size, tokenizer.token(0), tokenizer.id("<|im_end|>")

`load` takes a tokenizer.json or the directory holding one, a GGUF file, or a hub reference
(`owner/repo`, `owner/repo:QUANT` for a GGUF's header, `hf:` or `ms:` for the hub); remote
sources fetch metadata only, never weights. A `Tokenizer` never changes, so it may be shared
between threads, and it pickles as its source. Errors are `ToknrollError` subclasses.

Ids are `array('i')`, four bytes each and nothing boxed: slice it, iterate it, `len` it, or hand
it to numpy with `numpy.frombuffer(ids, dtype=numpy.int32)` without a copy. `decode` takes any
int32 buffer the same way, so `decode(encode(text))` copies nothing in either direction.
"""
from __future__ import annotations

import ctypes as C
import os
from array import array
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import IntFlag
from itertools import accumulate

from . import _native as N

__all__ = ["load", "configure", "Tokenizer", "TokenFlags", "ToknrollError", "InvalidArgument", "NotFound"]
__version__ = "0.3.1"  # the toknroll release the library was built from; make check-version keeps the four spellings equal


class TokenFlags(IntFlag):
    """A token's flags: bit n is GGUF token type n, exactly one of these six today; higher bits
    are reserved. CONTROL and USER_DEFINED are the special tokens."""

    NORMAL = 1 << 1
    UNKNOWN = 1 << 2
    CONTROL = 1 << 3
    USER_DEFINED = 1 << 4
    UNUSED = 1 << 5
    BYTE = 1 << 6


class ToknrollError(Exception):
    """A failure in libtoknroll; `.code` is the TOKNROLL_E* code."""

    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


class InvalidArgument(ToknrollError, ValueError):
    """A bad argument: an id outside the vocabulary, a malformed source."""


class NotFound(ToknrollError, FileNotFoundError):
    """A file, repository or token that does not exist."""


def _error(lib: C.CDLL, code: int | None = None) -> ToknrollError:
    """The calling thread's last error; the code is read from the library when the call returned none."""
    box = C.c_int32()
    text = lib.toknroll_last_error(C.byref(box))
    message = text.decode("utf-8", "replace") if text else ""
    code = box.value if code is None else code
    return {N.EINVAL: InvalidArgument, N.ENOTFOUND: NotFound}.get(code, ToknrollError)(message, code)


def _check(lib: C.CDLL, rc: int) -> int:
    if rc < 0:
        raise _error(lib, rc)
    return rc


def _text(value: object, what: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{what}: expected str, got {type(value).__name__}")
    return value.encode()


def _int(value: object, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{what}: expected int, got {type(value).__name__}")
    return value


def _ids(value, what: str) -> tuple[object, int, int]:
    """(owner, address, count) of int32 ids: a writable int32 buffer in place, anything else copied once."""
    if isinstance(value, array) and value.itemsize == 4 and value.typecode in "il":
        return value, value.buffer_info()[0], len(value)
    try:
        view = memoryview(value)
    except TypeError:
        view = None  # an iterable of int
    if view is not None:
        if view.format in "il" and view.itemsize == 4 and view.c_contiguous and not view.readonly:
            raw = (C.c_char * view.nbytes).from_buffer(view)
            return raw, C.addressof(raw), view.nbytes // 4
        if view.format in ("B", "b", "c"):
            raise TypeError(f"{what}: bytes are ambiguous as ids; pass an array('i') or an int32 view")
        value = view.tolist()
    try:
        ids = array("i", value)
    except (TypeError, OverflowError):
        raise TypeError(f"{what}: expected int32 ids") from None
    return ids, ids.buffer_info()[0], len(ids)


class Tokenizer:
    """A loaded tokenizer: `load()` makes one. Immutable, so shareable between threads.

    Closed by `close()`, a `with` block or garbage collection. The handle belongs to the library
    that made it: libtoknroll, or libjinfer for a model's tokenizer.
    """

    __slots__ = ("_h", "_source", "_lib", "__weakref__")

    def __init__(self, handle: int, source: str, lib: C.CDLL | None = None):
        """Internal: wraps a handle the library returned. Use `load()`."""
        self._lib = lib or N.library()
        self._source = source
        self._h = handle

    @property
    def _handle(self) -> int:
        if self._h is None:
            raise ValueError("the tokenizer is closed")
        return self._h

    @property
    def source(self) -> str:
        """What `load()` was given: a path or a hub reference."""
        return self._source

    @property
    def vocabulary_size(self) -> int:
        """How many ids the vocabulary has; every id from `encode` is below it."""
        return _check(self._lib, self._lib.toknroll_vocabulary_size(self._handle))

    def encode(self, text: str, *, specials: bool = False) -> array[int]:
        """The text's token ids, an `array('i')`. Special tokens spelled in the text are plain
        text unless `specials=True`, which keeps them whole."""
        lib, data = self._lib, _text(text, "encode")
        if b"\0" in data:  # a C string ends at a NUL; the batch carries lengths
            return self.encode_batch([text], specials=specials)[0]
        # prose runs about four bytes a token: one pass almost always, a second when it does not
        capacity = len(data) // 2 + 16
        out = array("i", bytes(4 * capacity))
        n = _check(lib, lib.toknroll_encode(self._handle, data, specials, out.buffer_info()[0], capacity))
        if n > capacity:
            out = array("i", bytes(4 * n))
            _check(lib, lib.toknroll_encode(self._handle, data, specials, out.buffer_info()[0], n))
        del out[n:]
        return out

    def count_tokens(self, text: str, *, specials: bool = False) -> int:
        """`len(encode(text))` without building the list."""
        data = _text(text, "count_tokens")
        if b"\0" in data:
            return len(self.encode_batch([text], specials=specials)[0])
        return _check(self._lib, self._lib.toknroll_encode(self._handle, data, specials, None, 0))

    def encode_batch(self, texts: Sequence[str], *, specials: bool = False) -> list[array[int]]:
        """`encode` over many texts in one call; the library spreads them over the cores."""
        lib, handle = self._lib, self._handle
        data = [_text(text, "encode_batch") for text in texts]
        count = len(data)
        joined = b"".join(data)  # the texts travel as one buffer, text i ending at text_ends[i]
        text_ends = array("i", accumulate(map(len, data)))
        ends = array("i", bytes(4 * count))
        capacity = len(joined) // 2 + 16 * count
        out = array("i", bytes(4 * capacity))
        n = _check(lib, lib.toknroll_encode_batch(handle, joined, text_ends.buffer_info()[0], count, specials,
                                                  out.buffer_info()[0], capacity, ends.buffer_info()[0]))
        if n > capacity:
            out = array("i", bytes(4 * n))
            _check(lib, lib.toknroll_encode_batch(handle, joined, text_ends.buffer_info()[0], count, specials,
                                                  out.buffer_info()[0], n, ends.buffer_info()[0]))
        return [out[start:end] for start, end in zip([0, *ends[:-1]], ends)]

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        """The tokens' bytes: UTF-8, except where a token holds part of a character. Any int32
        buffer, an `array('i')` or a numpy array, is read in place."""
        lib = self._lib
        owner, address, count = _ids(ids, "decode_bytes")  # owner keeps the buffer alive through the calls
        capacity = 8 * count + 16  # a token is a few bytes; one pass almost always
        out = C.create_string_buffer(capacity)
        n = _check(lib, lib.toknroll_decode(self._handle, address, count, out, capacity))
        if n >= capacity:
            out = C.create_string_buffer(n + 1)
            _check(lib, lib.toknroll_decode(self._handle, address, count, out, n + 1))
        return out.raw[:n]

    def decode(self, ids: Iterable[int], *, errors: str = "replace") -> str:
        """The text. A token holding part of a character decodes per `errors`, as in tiktoken."""
        return self.decode_bytes(ids).decode("utf-8", errors)

    def decode_batch(self, batch: Sequence[Iterable[int]], *, errors: str = "replace") -> list[str]:
        """`decode` over many id lists, on threads; the library releases the GIL."""
        with ThreadPoolExecutor() as pool:
            return list(pool.map(lambda ids: self.decode(ids, errors=errors), batch))

    def token(self, id: int) -> str | None:
        """The spelling of an id; None when the vocabulary has no such id."""
        lib, id = self._lib, _int(id, "token")
        out = C.create_string_buffer(64)  # spellings are short; a longer one gets a second pass
        n = lib.toknroll_token(self._handle, id, out, 64)
        if n == N.ENOTFOUND:
            return None
        if _check(lib, n) >= 64:
            out = C.create_string_buffer(n + 1)
            _check(lib, lib.toknroll_token(self._handle, id, out, n + 1))
        return out.raw[:n].decode()

    def token_flags(self, id: int) -> TokenFlags | None:
        """An id's `TokenFlags`; None when the vocabulary has no such id."""
        rc = self._lib.toknroll_token_flags(self._handle, _int(id, "token_flags"))
        return None if rc == N.ENOTFOUND else TokenFlags(_check(self._lib, rc))

    def id(self, token: str) -> int | None:
        """The id of a spelling; None when the vocabulary has no such token."""
        data = _text(token, "id")
        if b"\0" in data:
            return None  # no vocabulary spells a token with a NUL in it
        rc = self._lib.toknroll_id(self._handle, data)
        return None if rc == N.ENOTFOUND else _check(self._lib, rc)

    def close(self) -> None:
        """Frees the tokenizer; idempotent. Every other method is refused afterwards. Close it
        when no other thread is using it, as with a file."""
        handle, self._h = getattr(self, "_h", None), None
        if handle is not None:
            self._lib.toknroll_free(handle)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # interpreter shutdown: the library may already be gone
            pass

    def __reduce__(self):
        return (load, (self._source,))  # pickling, multiprocessing: the other side loads the source

    def __repr__(self) -> str:
        if self._h is None:
            return "Tokenizer(closed)"
        return f"Tokenizer({self._source!r}, vocabulary_size={self.vocabulary_size})"


def configure(*options: str) -> None:
    """Runtime options as on a java command line, once and before the library's first use:
    `"-Dtoknroll.gguf.pre.my-model=alias:qwen2"` names a pre-tokenizer, `"-Dtoknroll.cache.root=..."`
    moves the hub cache, `"-XX:MaxHeapSize=512m"` caps the heap. A later call is refused, a rejected
    option fails this one, and `TOKNROLL_OPTIONS` in the environment overrides what is given here."""
    data = (C.c_char_p * len(options))(*(_text(option, "configure") for option in options))
    lib = N.library()
    _check(lib, lib.toknroll_configure(data, len(options)))


def load(source: str | os.PathLike) -> Tokenizer:
    """A tokenizer from a tokenizer.json or its directory, a GGUF file, or `owner/repo[:QUANT]`
    on a hub. Remote sources fetch tokenizer metadata only, never weights, and cache it on disk."""
    text = os.fspath(source)
    if not isinstance(text, str):
        raise TypeError(f"load: expected str or PathLike, got {type(source).__name__}")
    if os.path.exists(text):
        text = os.path.abspath(text)  # the source outlives the working directory: pickling, repr
    lib = N.library()
    handle = lib.toknroll_load(text.encode())
    if not handle:
        raise _error(lib)
    return Tokenizer(handle, text, lib)
