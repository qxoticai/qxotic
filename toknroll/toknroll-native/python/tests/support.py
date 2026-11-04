"""Shared fixtures: the local GGUF every test can use, and whether the hub may be touched."""
import os
import sys
import unittest
from pathlib import Path

if "TOKNROLL_INSTALLED" not in os.environ:  # the checkout's package; check-wheel tests the installed one
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import toknroll  # noqa: E402

_found = sorted(Path.home().glob(".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M-GGUF/snapshots/*/LFM2.5-350M-Q8_0.gguf"))
GGUF = os.environ.get("TOKNROLL_TEST_GGUF") or (str(_found[0]) if _found else None)
OFFLINE = os.environ.get("TOKNROLL_OFFLINE") == "1"
HUB_JSON = "Qwen/Qwen3-0.6B"               # a tokenizer.json repository
HUB_GGUF = "LiquidAI/LFM2.5-350M-GGUF:Q8_0"  # a GGUF whose header is fetched

needs_gguf = unittest.skipIf(GGUF is None, "no local GGUF: set TOKNROLL_TEST_GGUF")

# A tokenizer of the tiktoken family, the kind toknroll-vector's engine encodes: Qwen3's (the qwen2 pattern).
_json = sorted(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"))
TIKTOKEN_JSON = os.environ.get("TOKNROLL_TEST_TIKTOKEN_JSON") or (str(_json[0]) if _json else None)
needs_hub = unittest.skipIf(OFFLINE, "TOKNROLL_OFFLINE=1")

try:
    import numpy
except ImportError:  # the package has no dependency on it; the buffer tests do
    numpy = None
needs_numpy = unittest.skipIf(numpy is None, "numpy not installed")

# Text that has broken a tokenizer somewhere: every string here must survive encode -> decode.
EDGE_TEXTS = [
    "", " ", "  ", "\n", "\r\n", "\t", " ", "​", " ",
    "a", "Hello, world!", "Hello, Tok'n'Roll! 🎸",
    "   leading", "trailing   ", "double  space", "tab\tseparated\tcolumns",
    "line1\nline2\n\nline4", "CRLF\r\nline",
    "日本語のテキスト", "中文文本测试", "한국어 텍스트", "Ελληνικά", "Кириллица", "العربية", "עברית", "हिन्दी", "ไทย",
    "emoji 🎸🎹🎺", "zwj 👨‍👩‍👧‍👦", "flags 🇨🇭🇯🇵", "skin 👍🏽", "keycap 1️⃣",
    "combining é and ñ", "zalgo h̵̡e̷̠",
    "RTL mixed: hello שלום world", "fullwidth ＡＢＣ１２３",
    "code: def f(x): return x**2  # comment", "json: {\"a\": [1, 2, {\"b\": null}]}",
    "url https://example.com/a?b=c&d=e#f", "email a.b+c@example.com",
    "numbers 1234567890 3.14159 -42 1e10 0x1F", "math ∑∫∂√∞≈≠≤≥",
    "<|im_start|>", "<|im_start|>user", "<|im_end|>\n", "text <|endoftext|> more",
    "<s>", "</s>", "<unk>", "[CLS] [SEP]", "<|not_a_special|>",
    "repeat " * 50, "x" * 1000, "🎸" * 300, "\n" * 100,
    "mixed 日本 and English and 😀 and 123",
    "quotes ‘single’ “double” «guillemets»", "dashes - – — ―", "ellipsis … and …",
    "control \x01\x02\x7f chars", "nul in \x00 the middle", "\x00", "end nul\x00",
    "surrogate-free max \U0010ffff", "bom ﻿ text",
]


def tiny_gguf(path, pre="gpt2"):
    """A GGUF tokenizer, header only, as the C tests' fixtures.h writes it: byte b is id b, "he" 256."""
    import struct
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    spelling, n = {}, 0
    for b in range(256):
        spelling[b] = chr(b) if b in printable else chr(256 + n)
        n += b not in printable
    tokens = [spelling[b] for b in range(256)] + ["he"]
    s = lambda text: struct.pack("<Q", len(text.encode())) + text.encode()  # noqa: E731
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, 5)
    out += s("tokenizer.ggml.model") + struct.pack("<I", 8) + s("gpt2")
    out += s("tokenizer.ggml.pre") + struct.pack("<I", 8) + s(pre)
    out += s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(tokens)) + b"".join(map(s, tokens))
    out += s("tokenizer.ggml.token_type") + struct.pack("<IIQ", 9, 5, len(tokens)) + struct.pack(f"<{len(tokens)}i", *[1] * len(tokens))
    out += s("tokenizer.ggml.merges") + struct.pack("<IIQ", 9, 8, 1) + s("h e")
    Path(path).write_bytes(out)
    return str(path)
