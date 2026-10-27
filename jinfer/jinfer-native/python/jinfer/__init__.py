"""jinfer: local AI inference from Python, over libjinfer's C API.

    import jinfer

    model = jinfer.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0")      # a LanguageModel; the file decides
    reply = model.chat("Why is the sky blue?", max_tokens=200)   # already generating
    for text in reply: print(text, end="", flush=True)          # stream it ...
    reply.text, reply.usage.prompt_ms                           # ... or just read it
    print(model.chat("Hi"))                                     # a reply prints as its text

    @dataclass
    class Person: name: str; born: int
    person = model.chat("Ada Lovelace, born 1815.", schema=Person).parsed   # a Person, guaranteed

    def weather(city: str) -> dict:
        \"\"\"Current weather for a city.\"\"\"
        return {"celsius": 18}
    conversation = model.conversation(system="Be brief.", tools=[weather])
    conversation.chat("Weather in Zurich?").text                # runs weather, then answers
    branch = conversation.fork(); conversation.rewind()         # free: nothing is recomputed

    model.chat(["What is this?", Path("photo.jpg")]).text       # media are values

    embedder = jinfer.load("LiquidAI/LFM2.5-Embedding-350M-GGUF:Q8_0")   # an Embedder
    vectors = embedder.embed(["a cat", "a dog"])                 # two float32 arrays

`load` returns the class the file calls for: LanguageModel, Embedder, Reranker, Speaker or
Transcriber, each with only the methods that apply. Options are keyword-only and `None` means
the model's own default. Embeddings and audio are float32 `array.array` buffers, zero-copy into
numpy or torch; scores are floats. The library's lines go to `logging.getLogger("jinfer")`; a download at a terminal
shows jinfer's progress board on stderr.
"""
from __future__ import annotations

import asyncio
import base64
import ctypes as C
import dataclasses
import enum
import functools
import inspect
import json
import logging
import mimetypes
import os
import signal
import sys
import threading
import types
import typing
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Literal, TypeVar

import array as _array

try:
    import toknroll._native  # fails on the monorepo's toknroll/ directory too, an empty namespace package
except ImportError:  # a source checkout before pip install: the sibling package
    sys.modules.pop("toknroll", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "toknroll" / "toknroll-native" / "python"))
    import toknroll

from . import _native as N  # finds the library through toknroll's locate

toknroll._native.bind(N.lib)  # libjinfer carries the toknroll_* symbols: its tokenizers belong to it

__all__ = ["load", "configure", "Model", "LanguageModel", "Embedder", "Reranker", "Speaker", "Transcriber",
           "Conversation", "Reply", "Speech", "Transcript", "Word", "ToolCall", "Usage",
           "JinferError", "InvalidArgument", "NotFound", "Cancelled"]

__version__ = N.lib.jinfer_version().decode()

T = TypeVar("T")
FinishReason = Literal["stop", "length", "tool_calls", "cancelled"]


# ---- errors ----

class JinferError(RuntimeError):
    """A failure libjinfer reports; `.code` is the JINFER_E* code."""

    code = N.ERROR

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class InvalidArgument(JinferError, ValueError):
    """A bad argument: an option out of range, a prompt over the context, a closed handle."""

    code = N.EINVAL


class NotFound(JinferError, FileNotFoundError):
    """A model, file or companion that does not exist, or is not cached in offline mode."""

    code = N.ENOTFOUND


class Cancelled(JinferError):
    """A stream another thread cancelled while this one was stepping it."""

    code = N.ECANCELLED


def _error(code: int | None = None) -> JinferError:
    """The calling thread's last error; the code is read from the library when the call returned none."""
    box = C.c_int32()
    text = N.lib.jinfer_last_error(C.byref(box))
    message = text.decode("utf-8", "replace") if text else ""
    code = box.value if code is None else code
    cls = {N.EINVAL: InvalidArgument, N.ENOTFOUND: NotFound, N.ECANCELLED: Cancelled}.get(code, JinferError)
    return cls(message.removeprefix("Invalid argument: "), code)


def _check(rc: int) -> int:
    if rc < 0:
        raise _error(rc)
    return rc


def _check_ptr(ptr):
    if not ptr:
        raise _error()
    return ptr


def _utf8(value: str, what: str) -> bytes:
    """A C string: one with a NUL inside is refused, where C would silently end it there."""
    if "\0" in value:
        raise InvalidArgument(f"{what}: contains a NUL character, where a C string would end")
    return value.encode()


def _strings(values, what: str):
    """A C array of UTF-8 strings; anything but str among them is a TypeError."""
    for i, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"{what}: expected str, got {type(value).__name__} at index {i}")
    return (C.c_char_p * len(values))(*(_utf8(value, f"{what}[{i}]") for i, value in enumerate(values)))


# ---- logging: the library's lines reach Python's logging, silent until the application configures it ----

_logger = logging.getLogger("jinfer")
_LEVELS = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING, 3: logging.ERROR}


def _log(level, message, _):
    _logger.log(_LEVELS.get(level, logging.INFO), "%s", message.decode("utf-8", "replace"))


_log_callback = N.LogFn(_log)  # kept alive for the life of the library
N.lib.jinfer_log_set(_log_callback, None)


# ---- messages: strings, Paths and bytes become the wire's content parts ----

def _data_url(mime: str, data: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}


def _audio(fmt: str, data: bytes) -> dict:
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": fmt}}


def _part(item) -> dict:
    """One item of a content list: text, an image or audio file, bytes of either, or a part as is."""
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return {"type": "text", "text": item}
    if isinstance(item, os.PathLike):
        path = Path(item)
        mime = mimetypes.guess_type(path.name)[0] or ""
        if mime.startswith("image/"):
            return _data_url(mime, path.read_bytes())
        if mime.startswith("audio/"):
            return _audio(path.suffix.lstrip(".").lower(), path.read_bytes())
        raise InvalidArgument(f"{path.name}: not an image or audio file")
    if isinstance(item, (bytes, bytearray)):
        data = bytes(item)
        if data[:4] == b"\x89PNG":
            return _data_url("image/png", data)
        if data[:3] == b"\xff\xd8\xff":
            return _data_url("image/jpeg", data)
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return _audio("wav", data)
        raise InvalidArgument("bytes of an unknown format: pass a Path so the type is known")
    raise InvalidArgument(f"content of type {type(item).__name__}: expected str, bytes, Path or a part dict")


def _content(content):
    return [_part(item) for item in content] if isinstance(content, (list, tuple)) else content


def _messages(messages, system: str | None) -> list[dict]:
    """OpenAI messages from a string, a content list (text with Paths and bytes) or messages."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    elif isinstance(messages, (list, tuple)):
        if messages and not all(isinstance(m, dict) and "role" in m for m in messages):
            messages = [{"role": "user", "content": list(messages)}]  # a content list: a part dict first is fine
    else:
        raise TypeError(f"messages: expected a string, a content list or a list of messages, got {type(messages).__name__}")
    out = [dict(m) for m in messages]
    for m in out:
        m["content"] = _content(m.get("content"))
    if system:
        if any(m.get("role") == "system" for m in out):
            raise InvalidArgument("system= given and the messages already have a system message; use one")
        out.insert(0, {"role": "system", "content": system})
    return out


# ---- schemas: Python types to JSON schema and back ----

def _schema(tp) -> dict:
    """A JSON schema for a dict (as is), a Pydantic class, a dataclass, a TypedDict, list[T],
    Literal, Enum, Optional and the scalars."""
    if isinstance(tp, dict):
        return tp
    if hasattr(tp, "model_json_schema"):
        return tp.model_json_schema()
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is Literal:
        return {"enum": list(args)}
    if origin in (typing.Union, types.UnionType):
        return {"anyOf": [{"type": "null"} if a is type(None) else _schema(a) for a in args]}
    if origin is list or tp is list:
        return {"type": "array", "items": _schema(args[0]) if args else {}}
    if origin is dict or tp is dict:
        if args and args[1] is not Any:
            # the grammar enforces named properties, not a value type on an open object
            raise InvalidArgument(f"schema: {tp!r} cannot be enforced; use dict for a free-form object, a dataclass or TypedDict for a typed one")
        return {"type": "object"}
    if dataclasses.is_dataclass(tp):
        hints = typing.get_type_hints(tp)
        fields = dataclasses.fields(tp)
        return {"type": "object", "properties": {f.name: _schema(hints[f.name]) for f in fields},
                "required": [f.name for f in fields if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]}
    if typing.is_typeddict(tp):
        hints = typing.get_type_hints(tp)
        return {"type": "object", "properties": {k: _schema(v) for k, v in hints.items()}, "required": sorted(tp.__required_keys__)}
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return {"enum": [m.value for m in tp]}
    if tp is Any:
        return {}
    scalar = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}.get(tp) if isinstance(tp, type) else None
    if scalar:
        return {"type": scalar}
    raise InvalidArgument(f"schema: cannot describe {tp!r}")


def _parse(value, tp):
    """The JSON value as the type the schema was made from."""
    if isinstance(tp, dict) or tp is Any:
        return value
    if hasattr(tp, "model_validate"):
        return tp.model_validate(value)
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is list and args:
        return [_parse(v, args[0]) for v in value]
    if origin in (typing.Union, types.UnionType):
        for a in args:
            if a is type(None):
                if value is None:
                    return None
                continue
            try:
                return _parse(value, a)
            except (TypeError, ValueError, KeyError):
                pass
        return value
    if dataclasses.is_dataclass(tp):
        hints = typing.get_type_hints(tp)
        return tp(**{f.name: _parse(value[f.name], hints[f.name]) for f in dataclasses.fields(tp) if f.name in value})
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return tp(value)
    return value


def _tool(fn: Callable) -> dict:
    """A function as a tool: its name, the first paragraph of its docstring, and its parameters
    from the type hints. ponytail: no per-parameter descriptions; name parameters well instead."""
    if fn.__name__ == "<lambda>":
        raise InvalidArgument("a tool needs a name: define it with def, not lambda")
    hints = typing.get_type_hints(fn)
    params = inspect.signature(fn).parameters
    return {"type": "function", "function": {
        "name": fn.__name__,
        "description": (inspect.getdoc(fn) or fn.__name__).split("\n\n")[0],
        "parameters": {"type": "object",
                       "properties": {name: _schema(hints.get(name, Any)) for name in params},
                       "required": [name for name, p in params.items() if p.default is inspect.Parameter.empty]}}}


def _tool_message(call: "ToolCall", result) -> dict:
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        result = dataclasses.asdict(result)
    content = result if isinstance(result, str) else json.dumps(result, default=str)  # anything else: its str, not a crash
    return {"role": "tool", "tool_call_id": call.id, "content": content}


# the chat options and the types each accepts; keyword-only on every call that takes them, `None` is the model's default
_OPTIONS = {"max_tokens": int, "seed": int, "temperature": (int, float), "top_p": (int, float),
            "stop": (str, list, tuple), "reasoning_effort": str, "tools": (list, tuple, Callable), "tool_choice": (str, dict, Callable)}


def _wrapped(spec: dict) -> bool:
    """Whether the schema's root is a scalar: it goes on the wire as {"value": ...}. An opening
    brace or bracket anchors the model's first token; a bare number or word after a reasoning span
    does not, and the model lands on a wrong digit. OpenAI refuses non-object roots for the same
    reason; arrays stay bare here because small models name list items worse inside a wrapper."""
    return spec.get("type") not in ("object", "array") and "properties" not in spec and "items" not in spec


def _request(messages: list[dict], schema, options: dict) -> dict:
    """The chat request: messages, the options that were given, tools as the wire wants them,
    and the schema as a strict response format."""
    for key, value in options.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, _OPTIONS[key])):  # no option is a bool
            raise TypeError(f"{key}: got {type(value).__name__}")
    request = {"messages": messages, **{k: v for k, v in options.items() if v is not None}}
    tools = request.get("tools")
    if tools:
        request["tools"] = [_tool(t) if callable(t) else t for t in ([tools] if callable(tools) else tools)]
    choice = request.get("tool_choice")
    if callable(choice) or (isinstance(choice, str) and choice not in ("auto", "none", "required")):
        request["tool_choice"] = {"type": "function", "function": {"name": choice.__name__ if callable(choice) else choice}}
    if schema is not None:
        spec = _schema(schema)
        name = getattr(schema, "__name__", None) or spec.get("title") or "response"
        if _wrapped(spec):
            spec = {"type": "object", "properties": {"value": spec}, "required": ["value"]}
        request["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "schema": spec}}
    return request


# ---- float buffers: stdlib float32 arrays, zero-copy into numpy or torch through the buffer protocol ----

def _floats(n: int):
    """A zeroed float32 array of n and its C pointer (None when empty, as the C API allows)."""
    buf = _array.array("f", bytes(4 * n))
    return buf, (C.c_float * n).from_buffer(buf) if n else None


def _copy_floats(ptr, n: int):
    buf = _array.array("f")
    if n:
        buf.frombytes(C.string_at(ptr, 4 * n))
    return buf


def _f32(samples):
    """`samples` as a float32 array: a float32 buffer (array, memoryview, numpy) in one copy, any
    other iterable of numbers element by element."""
    if isinstance(samples, _array.array) and samples.typecode == "f":
        return samples
    try:
        view = memoryview(samples)
    except TypeError:
        return _array.array("f", samples)
    if view.format != "f":
        return _array.array("f", samples)
    buf = _array.array("f")
    buf.frombytes(view.cast("B"))
    return buf


def _read_wav(path) -> tuple:
    """(float32 mono PCM, sample rate) of a PCM WAV file, first channel."""
    with wave.open(str(path), "rb") as f:
        rate, channels, width = f.getframerate(), f.getnchannels(), f.getsampwidth()
        frames = f.readframes(f.getnframes())
    code, scale, offset = {1: ("B", 128.0, 128), 2: ("h", 32768.0, 0), 4: ("i", 2147483648.0, 0)}.get(width, (None, 0, 0))
    if code is None:
        raise InvalidArgument(f"{width * 8}-bit WAV is not supported; 8, 16 or 32-bit PCM is")
    ints = _array.array(code, frames)[::channels]
    return _array.array("f", ((s - offset) / scale for s in ints)), rate


# ---- results ----

@dataclass(frozen=True)
class Usage:
    """What the reply cost on this machine. `prompt_tokens` is the whole prompt, `cached_tokens`
    the part the engine already had; `prompt_ms` is the time to the first token."""
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    prompt_ms: float
    completion_ms: float


@dataclass(frozen=True)
class ToolCall:
    """A tool the model asked for; `arguments` is the parsed JSON (a dict), or the raw string when
    it did not parse."""
    id: str
    name: str
    arguments: Any


@dataclass(frozen=True)
class Word:
    """A transcribed word with its time span in seconds and the model's confidence in it."""
    text: str
    start: float
    end: float
    confidence: float


@dataclass(frozen=True)
class Transcript:
    """What was said: the text, and the words with their timings."""
    text: str
    words: list[Word]


class _Stream:
    """The lifecycle of a jinfer_stream handle; Reply and Speech say what a chunk is."""

    def __init__(self, ptr, model: "Model"):
        self._ptr = _check_ptr(ptr)
        self._model = model
        self._done = False
        self._result = None

    @property
    def _h(self):
        if self._ptr is None:
            raise InvalidArgument(f"the {type(self).__name__.lower()} is closed")
        return self._ptr

    def _step(self) -> bool:
        """Advances one chunk; False once the stream is done. Raises the generation's failure, or
        Cancelled once when another thread cancelled; the result is there after either end."""
        if self._done:
            return False
        rc = N.lib.jinfer_stream_step(self._h)
        if rc == N.OK:
            return True
        self._done = True
        failure = _error(rc) if rc < 0 else None  # first: reading the result of a failed stream sets its own last error
        raw = N.lib.jinfer_stream_result(self._h)  # finished or cancelled: what was produced, counted
        if raw:
            self._result = json.loads(raw.decode())
        if failure:
            raise failure
        return False

    def _chunk(self):
        return None

    def __iter__(self):
        while self._step():
            yield self._chunk()

    async def __aiter__(self):
        # the work is CPU-bound and in-process: each step runs in a worker thread, the loop stays free
        loop = asyncio.get_running_loop()
        while await loop.run_in_executor(None, self._step):
            yield self._chunk()

    def _finish(self) -> dict:
        if self._result is None:
            for _ in self:
                pass
            if self._result is None:
                raise InvalidArgument(f"the {type(self).__name__.lower()} has no result")  # closed before it ended
        return self._result

    def close(self) -> None:
        """Frees the stream. A running generation is cancelled, and what it produced so far stays
        readable, with the finish reason "cancelled"."""
        if self._ptr:
            if self._result is None:
                N.lib.jinfer_stream_cancel(self._ptr)
                try:
                    self._finish()
                except JinferError:
                    pass  # the stream is gone either way
            N.lib.jinfer_stream_free(self._ptr)
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Reply(_Stream, Generic[T]):
    """One chat reply, already generating. Iterate it (or `async for`) for the text as it arrives,
    or read `.text`, `.tool_calls`, `.parsed`, `.usage`: those wait for the end. Both views are the
    same reply; iterating after a property read yields nothing more. Close a reply early to cancel
    it; what it produced stays readable, with `finish_reason` "cancelled"."""

    def __init__(self, ptr, model: "Model", schema=None):
        super().__init__(ptr, model)
        self._schema = schema

    def _chunk(self) -> str:
        return N.lib.jinfer_stream_text(self._h).decode("utf-8", "replace")

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        if self._result is None:
            return "jinfer.Reply(generating)"
        return f"jinfer.Reply({self.text!r}, finish_reason={self.finish_reason!r})"

    @property
    def _message(self) -> dict:
        return self._finish()["choices"][0]["message"]

    @property
    def text(self) -> str:
        """The whole reply's text, "" when the model only asked for tools; waits for the end."""
        return self._message.get("content") or ""

    @property
    def reasoning(self) -> str | None:
        """The model's reasoning before the answer; None when it did not reason."""
        return self._message.get("reasoning_content")

    @property
    def tool_calls(self) -> list[ToolCall]:
        """The tools the model asked for, in order; empty when it answered in words."""
        out = []
        for call in self._message.get("tool_calls") or []:
            raw = call["function"].get("arguments") or "{}"
            try:
                arguments = json.loads(raw)
            except ValueError:
                arguments = raw
            out.append(ToolCall(call.get("id", ""), call["function"]["name"], arguments))
        return out

    @property
    def finish_reason(self) -> FinishReason:
        """Why the reply ended: "stop", "length" (max_tokens), "tool_calls" or "cancelled"."""
        return self._finish()["choices"][0]["finish_reason"]

    @property
    def usage(self) -> Usage:
        """Tokens and milliseconds; for a conversation turn with tools, the whole turn's."""
        t = self._finish()["timings"]
        cached = t.get("cache_n", 0)
        return Usage(cached + t["prompt_n"], cached, t["predicted_n"], t["prompt_ms"], t["predicted_ms"])

    @property
    def parsed(self) -> T:
        """The reply as the `schema=` type it was asked for. The grammar guarantees it parses."""
        text = self.text
        try:
            value = json.loads(text)
        except ValueError:
            if self._schema is None:
                raise InvalidArgument("the reply is not JSON; ask for one with schema=") from None
            raise InvalidArgument(f"the reply was cut off before the JSON closed (finish_reason={self.finish_reason!r}); raise max_tokens") from None
        if _wrapped(_schema(self._schema)):
            value = value["value"]
        return _parse(value, self._schema)


class Speech(_Stream):
    """Synthesized speech, already generating. Iterate it for float32 mono PCM chunks (arrays) as
    they are produced, or read `.audio` for all of it; `.save(path)` writes a 16-bit WAV."""

    def __init__(self, ptr, model: "Model"):
        super().__init__(ptr, model)
        self._chunks = []
        self.sample_rate = model.sample_rate

    def __repr__(self) -> str:
        if self._result is None:
            return "jinfer.Speech(generating)"
        return f"jinfer.Speech({self._result['seconds']:.1f} s at {self.sample_rate} Hz)"

    def _chunk(self):
        count = C.c_int32()
        ptr = N.lib.jinfer_stream_samples(self._h, C.byref(count))
        chunk = _copy_floats(ptr, count.value)
        self._chunks.append(chunk)
        return chunk

    @property
    def audio(self):
        """All samples, a float32 array, mono at `sample_rate`; waits for the end."""
        self._finish()
        out = _array.array("f")
        for chunk in self._chunks:
            out.extend(chunk)
        return out

    def save(self, path) -> None:
        """Writes all of the audio as a 16-bit mono WAV."""
        pcm = self.audio
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(self.sample_rate)
            out.writeframes(_array.array("h", (int(max(-1.0, min(1.0, s)) * 32767) for s in pcm)).tobytes())


# ---- models ----

class Model:
    """A loaded model; `load` returns the subclass the file calls for. `ref` is what `load` was
    given, `name` the model file's name. Closed by `close()`, a `with` block or garbage collection."""

    def __init__(self, ptr, ref: str, info: dict):
        self._ptr = ptr
        self._info = info
        self.ref = ref
        self.name = info["name"]

    @property
    def _h(self):
        if self._ptr is None:
            raise InvalidArgument("the model is closed")
        return self._ptr

    def close(self) -> None:
        """Drops this reference; live replies and speech keep the weights alive."""
        if self._ptr:
            N.lib.jinfer_model_free(self._h)
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        # what is really running: the first kernel backend is the one the matmuls take
        kernels = ", ".join(self._info.get("kernels") or ["?"])
        return f"jinfer.{type(self).__name__}({self.ref!r}, kernels={kernels}, threads={self._info.get('threads', '?')})"


class _Context(Model):
    """A model whose states hold a context of tokens to budget by."""

    @property
    def context_capacity(self) -> int:
        """Tokens of context the model's states hold: how long a prompt plus its reply may be.
        `load(context_capacity=)` sets it."""
        return self._info["context_capacity"]

    @property
    def max_context_length(self) -> int:
        """What the model was trained for, the bound on `context_capacity`."""
        return self._info["max_context_length"]


class _Tokens(_Context):
    """A model with a tokenizer: language and embedding models."""

    @functools.cached_property
    def tokenizer(self) -> toknroll.Tokenizer:
        """The model's tokenizer, a `toknroll.Tokenizer` served by libjinfer; it outlives the model."""
        return toknroll.Tokenizer(_check_ptr(N.lib.jinfer_model_tokenizer(self._h)), self.ref, N.lib)


_TOOL_ROUNDS = 8  # ponytail: a constant; make it an option when a caller needs more


class _Turn(Reply):
    """A conversation turn with functions among the tools: streams the reply, runs the functions
    the model asks for, and streams the next reply, until the model answers in words or the rounds
    run out. One Reply to the caller; its usage is the whole turn's."""

    def __init__(self, conversation: "Conversation", schema, options: dict, functions: dict):
        self._conversation, self._options, self._functions = conversation, options, functions
        self._result = None
        self._schema = schema
        self._rounds = 1
        self._timings: dict = {}
        self._reply = conversation._model._chat(conversation._messages, schema, options)

    def _step(self) -> bool:
        while not self._reply._step():
            calls = self._reply.tool_calls
            if self._rounds == _TOOL_ROUNDS or not calls or not all(call.name in self._functions for call in calls):
                return False  # the model answered, asked for a tool that is yours to run, or ran out of rounds
            history = self._conversation._messages
            history.append(self._reply._message)
            for call in calls:
                try:
                    result = self._functions[call.name](**call.arguments)
                except Exception as e:  # the model hears what went wrong and may try again
                    result = f"{type(e).__name__}: {e}"
                history.append(_tool_message(call, result))
            for k, v in self._reply._finish()["timings"].items():
                self._timings[k] = self._timings.get(k, 0) + v
            self._options.pop("tool_choice", None)  # a forced choice is for the first round
            self._reply.close()
            self._reply = self._conversation._model._chat(history, self._schema, self._options)
            self._rounds += 1
        return True

    def _chunk(self) -> str:
        return self._reply._chunk()

    def _finish(self) -> dict:
        if self._result is None:
            while self._step():
                pass
            result = dict(self._reply._finish())
            result["timings"] = {k: self._timings.get(k, 0) + v for k, v in result["timings"].items()}
            self._result = result
        return self._result

    def close(self) -> None:
        self._reply.close()


class Conversation:
    """A chat with memory: `chat` adds your turn and returns the reply, which joins the history when
    it finishes. Tools given as functions run here, until the model answers in words. The engine
    keeps the state between turns, so a turn costs only its new tokens; `fork` branches that history
    for free and `rewind` drops turns without recomputing."""

    def __init__(self, model: "LanguageModel", messages: list[dict], options: dict):
        self._model = model
        self._messages = list(messages)
        self._options = options
        self._pending: Reply | None = None

    def _settle(self) -> None:
        if self._pending is not None:
            reply, self._pending = self._pending, None
            try:
                message = reply._message
            except JinferError:
                self._messages.pop()  # the turn failed: your message leaves the history with it
                raise
            if reply.finish_reason == "cancelled" and not message.get("content"):
                self._messages.pop()  # cancelled before a word: the turn never happened
            else:
                self._messages.append(message)

    def __repr__(self) -> str:
        return f"jinfer.Conversation({len(self._messages) + (self._pending is not None)} messages)"

    @property
    def messages(self) -> list[dict]:
        """The history so far, the last reply included once it has finished."""
        self._settle()
        return list(self._messages)

    def chat(self, content, *, schema: type[T] | dict | None = None, max_tokens: int | None = None,
             temperature: float | None = None, top_p: float | None = None, seed: int | None = None,
             stop: str | list[str] | None = None, tools: list | None = None, tool_choice=None,
             reasoning_effort: str | None = None) -> Reply[T]:
        """Your turn (a string, or a list of strings, Paths and bytes) and the model's reply, already
        generating: iterate it for the text as it arrives, or read it. Functions among the tools run
        as the reply is read, between the words, until the model answers; `schema` constrains that
        answer, as in `LanguageModel.chat`."""
        if not isinstance(content, (str, list, tuple)):
            raise TypeError(f"content: expected a string or a list of strings, Paths and bytes, got {type(content).__name__}")
        self._settle()
        self._messages.append({"role": "user", "content": _content(content)})
        options = {**self._options, **{k: v for k, v in locals().items() if k in _OPTIONS and v is not None}}  # this turn's override the conversation's
        tools = options.get("tools") or []
        tools = [tools] if callable(tools) else list(tools)
        if not all(callable(fn) for fn in tools):
            self._messages.pop()
            raise InvalidArgument("a conversation runs its tools, so they are functions; to run a tool yourself, use model.chat with your own messages")
        self._pending = _Turn(self, schema, options, {fn.__name__: fn for fn in tools}) if tools else self._model._chat(self._messages, schema, options)
        return self._pending

    def prepare(self) -> None:
        """Computes the state for the history so far, so the next turn starts from it at once."""
        _check(N.lib.jinfer_prepare(self._model._h, json.dumps(_request(self._messages, None, self._options)).encode()))

    def fork(self) -> "Conversation":
        """A second conversation with the same history; the cached prefix serves both."""
        return Conversation(self._model, self.messages, self._options)

    def rewind(self, turns: int = 1) -> "Conversation":
        """Forgets the last `turns` exchanges (your message and everything that followed it); the
        system prompt stays."""
        self._settle()
        for _ in range(turns):
            while self._messages and self._messages[-1]["role"] not in ("user", "system"):
                self._messages.pop()
            if self._messages and self._messages[-1]["role"] == "user":
                self._messages.pop()
        return self


class LanguageModel(_Tokens):
    """A chat model: `chat` for one reply, `conversation` for a chat with memory."""

    def chat(self, messages, *, system: str | None = None, schema: type[T] | dict | None = None,
             max_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None,
             seed: int | None = None, stop: str | list[str] | None = None, tools: list | None = None,
             tool_choice=None, reasoning_effort: str | None = None) -> Reply[T]:
        """Starts a reply. `messages` is a string (one user turn), a content list (strings, Paths,
        bytes) or a list of messages; `system` prepends a system message; `schema` constrains the
        reply to a type (a dataclass, TypedDict, Pydantic class, list[T], Literal, Enum or a JSON
        schema dict), read back with `reply.parsed`; `tools` are functions or tool dicts, and a
        reply asking for one has `tool_calls` (a Conversation runs them for you)."""
        options = {k: v for k, v in locals().items() if k in _OPTIONS}  # the eight, by name
        return self._chat(_messages(messages, system), schema, options)

    def _chat(self, messages: list[dict], schema, options: dict) -> Reply:
        request = _request(messages, schema, options)
        return Reply(N.lib.jinfer_chat(self._h, json.dumps(request).encode()), self, schema)

    def conversation(self, *, system: str | None = None, messages: list[dict] = (), max_tokens: int | None = None,
                     temperature: float | None = None, top_p: float | None = None, seed: int | None = None,
                     stop: str | list[str] | None = None, tools: list | None = None, tool_choice=None,
                     reasoning_effort: str | None = None) -> Conversation:
        """A chat with memory; the options apply to every turn."""
        options = {k: v for k, v in locals().items() if k in _OPTIONS and v is not None}
        return Conversation(self, _messages(messages, system), options)


class Embedder(_Tokens):
    """An embedding model: `embed` turns texts into vectors."""

    @property
    def dimension(self) -> int:
        """The length of a vector from `embed`, before any `dimension=` truncation."""
        return self._info["embedding_dimension"]

    def embed(self, texts, *, dimension: int | None = None):
        """A float32 array per text, in the texts' order; a single string is a list of one.
        `dimension` truncates (and renormalises) a Matryoshka model's vectors."""
        texts = [texts] if isinstance(texts, str) else list(texts)
        strings = _strings(texts, "embed")
        n = _check(N.lib.jinfer_embed(self._h, strings, len(texts), dimension or 0, None, 0))
        buf, ptr = _floats(n)
        _check(N.lib.jinfer_embed(self._h, strings, len(texts), dimension or 0, ptr, n))
        d = n // max(len(texts), 1)
        return [buf[i * d:(i + 1) * d] for i in range(len(texts))]


class Reranker(_Context):
    """A reranking model: `rerank` scores documents against a query."""

    def rerank(self, query: str, documents, *, instruction: str | None = None):
        """A relevance score per document, in the documents' order; higher is more relevant.
        A single string is a list of one; `instruction` replaces the model's default."""
        if not isinstance(query, str):
            raise TypeError(f"query: expected str, got {type(query).__name__}")
        documents = [documents] if isinstance(documents, str) else list(documents)
        buf, ptr = _floats(len(documents))
        _check(N.lib.jinfer_rerank(self._h, _utf8(query, "query"), _strings(documents, "rerank"), len(documents),
                                   _utf8(instruction, "instruction") if instruction else None, ptr))
        return buf.tolist()


class Speaker(Model):
    """A text-to-speech model: `speak` synthesizes a Speech."""

    @property
    def sample_rate(self) -> int:
        """Hz of the audio the model produces."""
        return self._info["sample_rate"]

    def speak(self, text: str, *, speed: float | None = None) -> Speech:
        """Synthesizes speech; iterate the Speech for PCM chunks, or `.save("out.wav")`. `speed`
        is a factor, 1.0 the model's pace."""
        request = {"input": text, **({"speed": speed} if speed is not None else {})}
        return Speech(N.lib.jinfer_speak(self._h, json.dumps(request).encode()), self)


class Transcriber(Model):
    """A speech-to-text model: `transcribe` turns audio into a Transcript."""

    @property
    def sample_rate(self) -> int:
        """Hz the model listens at; `transcribe` resamples any other rate to it."""
        return self._info["sample_rate"]

    def transcribe(self, audio, *, sample_rate: int | None = None) -> Transcript:
        """Transcribes a WAV file (a path) or mono PCM samples at `sample_rate`: a float32 buffer
        (array, numpy) or any iterable of numbers; any rate, the library resamples. The Transcript
        has the text and timed words."""
        if isinstance(audio, (str, os.PathLike)):
            audio, sample_rate = _read_wav(audio)
        elif sample_rate is None:
            raise InvalidArgument("sample_rate is required for PCM samples")
        buf = _f32(audio)
        ptr = (C.c_float * len(buf)).from_buffer(buf) if len(buf) else None  # copied by the call
        with _Stream(N.lib.jinfer_transcribe(self._h, ptr, len(buf), sample_rate), self) as s:
            result = s._finish()
        return Transcript(result["text"], [Word(w["text"], w["start"], w["end"], w["confidence"]) for w in result["words"]])


_KINDS = {"language": LanguageModel, "embedding": Embedder, "reranking": Reranker, "speech": Speaker, "transcription": Transcriber}


def configure(*options: str) -> None:
    """Runtime options as on a java command line, once and before the first load: `"-Djinfer.threads=8"`
    sizes the inference pool, `"-Djinfer.cache.sessions=16"` and `"-Djinfer.cache.budget=8g"` the prompt
    cache, `"-Djinfer.offline=true"` never downloads, any `jinfer.*` or `jam.*` property, `"-XX:MaxHeapSize=4g"`.
    A later call is refused, a rejected option fails this one, and `JINFER_OPTIONS` in the environment
    overrides what is given here. `-Dtoknroll.*` options reach the model's tokenizers too, which libjinfer
    serves itself; `TOKNROLL_OPTIONS` and `toknroll.configure` configure the standalone libtoknroll only."""
    _check(N.lib.jinfer_configure(_strings(options, "configure"), len(options)))


def load(ref: str | os.PathLike, *, companions: dict[str, str] | None = None,
         prompt_cache: str | os.PathLike | None = None, context_capacity: int | None = None) -> Model:
    """Loads a hub reference such as "LiquidAI/LFM2.5-350M-GGUF:Q8_0" or a local path, downloading
    once. The file decides what comes back: a LanguageModel, Embedder, Reranker, Speaker or
    Transcriber. A download shows its progress on stderr at a terminal and logs it otherwise; Ctrl-C
    during one ends the process, and the download resumes next time.

    Options: companions ({"media": ref} for vision and audio in, {"voice": ref} for speech),
    prompt_cache (a catalog file, read at load and extended at close, so prepared prompts survive
    the process), context_capacity (the tokens of state the model keeps, at most its
    `max_context_length`; None is 4096 or the maximum when smaller, an embedder or reranker takes
    the maximum; -1 is the maximum). The thread count is the process's: `configure("-Djinfer.threads=N")`.
    """
    ref = os.fspath(ref)
    if context_capacity is not None and (isinstance(context_capacity, bool) or not isinstance(context_capacity, int)):
        raise TypeError(f"context_capacity: got {type(context_capacity).__name__}")
    options = N.ModelOptions()
    N.lib.jinfer_model_options_init(C.byref(options))
    options.context_capacity = context_capacity or 0
    if companions:  # ctypes keeps what is assigned to a pointer field alive with the struct
        options.companions = _strings([f"{k}={v}" for k, v in companions.items()], "companions")
        options.companion_count = len(companions)
    if prompt_cache:
        options.prompt_cache = _utf8(os.fspath(prompt_cache), "prompt_cache")
    # a blocking C call cannot deliver KeyboardInterrupt, so Ctrl-C during a download ends the process,
    # as it would end the jinfer CLI; the partial file resumes next time
    previous = signal.signal(signal.SIGINT, signal.SIG_DFL) if threading.current_thread() is threading.main_thread() else None
    try:
        ptr = _check_ptr(N.lib.jinfer_model_load(_utf8(ref, "ref"), C.byref(options)))
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    info = json.loads(_check_ptr(N.lib.jinfer_model_info(ptr)).decode())
    return _KINDS.get(info["kind"], Model)(ptr, ref, info)


from ._openai import OpenAI  # noqa: E402,F401  (needs load above; a migration aid, not in __all__)
