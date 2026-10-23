"""The OpenAI SDK's surface over a jinfer Model, in-process: client.chat.completions.create(...)
and client.embeddings.create(...) with the same arguments and response shapes, no server.
A migration aid: new code uses `jinfer.load` and `model.chat`."""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Iterator

from . import Model, _OPTIONS, load


class _Obj(dict):
    """An OpenAI response: a dict with attribute access, as the SDK's models read."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(name) from None
        return _wrap(value)

    def model_dump(self) -> dict:
        return dict(self)


def _wrap(value):
    if isinstance(value, dict) and not isinstance(value, _Obj):
        return _Obj(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


class _Completions:
    def __init__(self, model):
        self._model = model

    def create(self, *, messages, model: str | None = None, stream: bool = False, stream_options=None,
               response_format: dict | None = None, **options):
        # the SDK's request fields that have a jinfer option; the rest (n, user, logprobs, ...) are ignored
        schema = None
        if response_format and response_format.get("type") == "json_schema":
            schema = response_format["json_schema"]["schema"]
        elif response_format and response_format.get("type") == "json_object":
            schema = {"type": "object"}
        if "max_completion_tokens" in options:  # the SDK's newer name
            options.setdefault("max_tokens", options.pop("max_completion_tokens"))
        messages = [{**m, "role": "system"} if m.get("role") == "developer" else m for m in messages]
        reply = self._model.chat(messages, schema=schema, **{k: v for k, v in options.items() if k in _OPTIONS})
        if not stream:
            with reply:
                return _Obj(reply._finish())
        return self._chunks(reply, include_usage=bool(stream_options and stream_options.get("include_usage")))

    def _chunks(self, reply, include_usage: bool) -> Iterator[_Obj]:
        head = {"id": "chatcmpl-stream", "object": "chat.completion.chunk", "created": int(time.time()), "model": self._model.name}
        try:
            first = True
            for text in reply:
                delta = {"content": text}
                if first:
                    delta["role"] = "assistant"
                    first = False
                yield _Obj({**head, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
            response = reply._finish()
            last = {**head, "id": response["id"], "choices": [{"index": 0, "delta": {}, "finish_reason": reply.finish_reason}]}
            if include_usage:
                last["usage"] = response["usage"]
            yield _Obj(last)
        finally:
            reply.close()


class _Embeddings:
    def __init__(self, model):
        self._model = model

    def create(self, *, input, model: str | None = None, dimensions: int | None = None, **_):
        texts = [input] if isinstance(input, str) else list(input)
        vectors = self._model.embed(texts, dimension=dimensions)
        data = [{"object": "embedding", "index": i, "embedding": vector.tolist()} for i, vector in enumerate(vectors)]
        return _Obj({"object": "list", "data": data, "model": self._model.name,
                     "usage": {"prompt_tokens": 0, "total_tokens": 0}})


class OpenAI:
    """`OpenAI(model_ref)` instead of `OpenAI(api_key=...)`; everything after that reads the same.

        client = jinfer.OpenAI("LiquidAI/LFM2.5-350M-GGUF:Q8_0")
        reply = client.chat.completions.create(messages=[{"role": "user", "content": "Hi"}])
        reply.choices[0].message.content
    """

    def __init__(self, model: str | Model, **load_options):
        self.model = model if isinstance(model, Model) else load(model, **load_options)
        self.chat = SimpleNamespace(completions=_Completions(self.model))
        self.embeddings = _Embeddings(self.model)

    def close(self) -> None:
        """Closes the model; a model passed in is closed too."""
        self.model.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
