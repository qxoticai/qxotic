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
