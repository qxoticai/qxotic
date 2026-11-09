"""Encoding speed on text a tokenizer has never seen: one tokenizer, one model, one process.

Wikipedia articles from enwik8, one encode call each, as an application encodes documents. The
tokenizer first encodes a separate set of articles, so its caches hold common words as they would
in a running program; every measured article is new to it. Run it pinned to one core, in a fresh
process per measurement, and compare the ids' hash across tokenizers: the same hash, the same ids.

    taskset -c 3 python encode.py toknroll Qwen/Qwen3-0.6B
    taskset -c 3 python encode.py tokenizers Qwen/Qwen3-0.6B      # HF tokenizers 0.23 or 1.0
    taskset -c 3 python encode.py tiktoken openai/gpt-oss-20b --encoding o200k_harmony

Each run prints one JSON line; chart.py draws the lines it is given.
"""

import argparse
import array
import hashlib
import json
import re
import sys
import time
from pathlib import Path

WARM_UP = range(0, 300)  # articles encoded before measuring, never measured
MEASURED_FROM = 4000  # measured articles start here, far from the warm-up set


def articles(corpus: Path) -> list[str]:
    """enwik8's article texts, redirects and stubs left out."""
    raw = corpus.read_bytes().decode("utf-8", "replace")
    texts = (m.group(1) for m in re.finditer(r"<text[^>]*>(.*?)</text>", raw, re.S))
    return [t for t in texts if len(t) > 500 and not t.startswith("#REDIRECT")]


def encoder(engine: str, model: str, encoding: str | None):
    """A function from text to ids, and the library's name and version."""
    if engine == "toknroll":
        import toknroll

        tokenizer = toknroll.load(model)
        return (lambda text: tokenizer.encode(text)), f"toknroll {toknroll.__version__}"
    if engine == "tokenizers":
        import tokenizers
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained(model)
        if hasattr(tokenizer, "compile"):  # 1.0: encode lives on the compiled tokenizer
            tokenizer = tokenizer.compile()
        return (
            lambda text: tokenizer.encode(text, add_special_tokens=False).ids
        ), f"tokenizers {tokenizers.__version__}"
    if engine == "tiktoken":
        import tiktoken

        tokenizer = tiktoken.get_encoding(encoding)
        return (lambda text: tokenizer.encode_ordinary(text)), f"tiktoken {tiktoken.__version__}"
    raise SystemExit(f"unknown engine {engine}: toknroll, tokenizers or tiktoken")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("engine", choices=["toknroll", "tokenizers", "tiktoken"])
    parser.add_argument("model", help="the Hugging Face repository whose tokenizer is measured")
    parser.add_argument("--encoding", help="tiktoken's encoding for the model, such as o200k_harmony")
    parser.add_argument("--corpus", type=Path, default=Path.home() / ".cache/qxotic/toknroll/corpus/enwik8")
    parser.add_argument("--megabytes", type=float, default=8, help="how much text is measured")
    args = parser.parse_args()

    texts = articles(args.corpus)
    measured, size = [], 0
    for text in texts[MEASURED_FROM:]:
        measured.append(text)
        size += len(text.encode())
        if size >= args.megabytes * 1e6:
            break

    encode, library = encoder(args.engine, args.model, args.encoding)
    for i in WARM_UP:
        encode(texts[i])

    digest = hashlib.sha256()
    elapsed = 0.0
    for text in measured:
        start = time.perf_counter()
        ids = encode(text)
        elapsed += time.perf_counter() - start
        digest.update(array.array("i", ids).tobytes())

    json.dump(
        {
            "library": library,
            "model": args.model,
            "articles": len(measured),
            "bytes": size,
            "seconds": round(elapsed, 4),
            "mb_per_s": round(size / elapsed / 1e6, 2),
            "ids_sha256": digest.hexdigest(),
        },
        sys.stdout,
    )
    print()


if __name__ == "__main__":
    main()
