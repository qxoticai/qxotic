"""Shared fixtures: the local GGUF every test can use, and whether the hub may be touched."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

if "TOKNROLL_INSTALLED" not in os.environ:  # the checkout's package; check-wheel tests the installed one
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import toknroll  # noqa: E402

OFFLINE = os.environ.get("TOKNROLL_OFFLINE") == "1"
HUB_JSON = "Qwen/Qwen3-0.6B"               # a tokenizer.json repository
HUB_GGUF = "LiquidAI/LFM2.5-350M-GGUF:Q8_0"  # a GGUF whose header is fetched
needs_hub = unittest.skipIf(OFFLINE, "TOKNROLL_OFFLINE=1")
CLEARED = ("XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE", "JINFER_MODELS", "TOKNROLL_OPTIONS", "TOKNROLL_LOG", "TOKNROLL_PROGRESS")


def fresh(code: str, **env: str | None) -> subprocess.CompletedProcess:
    """`code` in a new interpreter with toknroll imported: the cache and toknroll variables cleared, then `env` set,
    a None left unset."""
    clean = {k: v for k, v in os.environ.items() if k not in CLEARED}
    clean.update({k: v for k, v in env.items() if v is not None})
    head = "" if "TOKNROLL_INSTALLED" in clean else f"import sys; sys.path.insert(0, {sys.path[0]!r})\n"
    return subprocess.run([sys.executable, "-c", head + "import toknroll\n" + code],
                          capture_output=True, text=True, timeout=300, env=clean)


_HUB = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface") / "hub")


def _local(env: str, *patterns: str) -> str | None:
    """A fixture on this machine: `env` names it, else the first match of the patterns under the HF hub cache."""
    found = sorted(path for pattern in patterns for path in _HUB.glob(pattern))
    return os.environ.get(env) or (str(found[0]) if found else None)


# The GGUF most tests use, LFM2.5-350M's and no other: its vocabulary size and special tokens are asserted.
# Required, so that the suite can never pass by skipping; the other fixtures are optional.
GGUF = _local("TOKNROLL_TEST_GGUF", "models--LiquidAI--LFM2.5-350M-GGUF/snapshots/*/LFM2.5-350M-Q8_0.gguf")
if GGUF is None:
    raise SystemExit("the tests need LFM2.5-350M-Q8_0.gguf: `huggingface-cli download LiquidAI/LFM2.5-350M-GGUF LFM2.5-350M-Q8_0.gguf`,"
                     " or TOKNROLL_TEST_GGUF=<path to that file>")
# A tokenizer of the tiktoken family, the kind toknroll-vector's engine encodes: Qwen3's (the qwen2 pattern).
TIKTOKEN_JSON = _local("TOKNROLL_TEST_TIKTOKEN_JSON", "models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json")
# A byte-level tokenizer.json whose added tokens are spelled outside ASCII: DeepSeek V3's, "<｜begin▁of▁sentence｜>".
DEEPSEEK_JSON = _local("TOKNROLL_TEST_DEEPSEEK_JSON", "models--deepseek-ai--DeepSeek-V3*/snapshots/*/tokenizer.json")
# A tiktoken checkout: a folder holding tiktoken.model and tokenizer_config.json, no tokenizer.json (Kimi's).
TIKTOKEN_DIR = _local("TOKNROLL_TEST_TIKTOKEN_DIR", "models--moonshotai--Kimi-K*/snapshots/*")
# A tokenizer with byte fallback, whose BYTE tokens spell part of a character: any Gemma 4's, GGUF or tokenizer.json.
BYTE_FALLBACK = _local("TOKNROLL_TEST_BYTE_FALLBACK", "models--google--gemma-4-*/snapshots/*/tokenizer.json",
                       "models--google--gemma-4-*/snapshots/*/*.gguf")

needs_deepseek = unittest.skipIf(DEEPSEEK_JSON is None, "no local DeepSeek V3 tokenizer.json: set TOKNROLL_TEST_DEEPSEEK_JSON")
needs_tiktoken_dir = unittest.skipIf(TIKTOKEN_DIR is None, "no local tiktoken checkout: set TOKNROLL_TEST_TIKTOKEN_DIR")
needs_byte_fallback = unittest.skipIf(BYTE_FALLBACK is None, "no local Gemma 4: set TOKNROLL_TEST_BYTE_FALLBACK")

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


def byte_level_json(path, merges=(), added=()):
    """A byte-level BPE tokenizer.json: GPT-2's 256-byte alphabet (byte b is id b), the merges' tokens after
    it in order, then the added tokens, each special; text is one piece (no regex)."""
    import json
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    extra = 0
    vocab = {}
    for b in range(256):
        vocab[chr(b) if b in printable else chr(256 + extra)] = b
        extra += b not in printable
    for merge in merges:
        vocab["".join(merge.split())] = len(vocab)
    tokens = [{"id": len(vocab) + i, "content": text, "special": True, "single_word": False, "lstrip": False, "rstrip": False,
               "normalized": False} for i, text in enumerate(added)]
    Path(path).write_text(json.dumps({
        "version": "1.0", "model": {"type": "BPE", "vocab": vocab, "merges": list(merges)}, "added_tokens": tokens,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False}, "decoder": {"type": "ByteLevel"}}))
    return str(path)


def sentencepiece_json(path, normalizer=None, decoder_replacement="▁", **model):
    """A SentencePiece-style tokenizer.json: ▁ is 0, "a" 1, "<unk>" 2, then the 256 byte tokens; Metaspace in
    and out, byte fallback on; `model` fields override the model's, `normalizer` is the normalizer's JSON."""
    import json
    vocab = {"▁": 0, "a": 1, "<unk>": 2, **{f"<0x{b:02X}>": 3 + b for b in range(256)}}
    model = {"type": "BPE", "vocab": vocab, "merges": [], "unk_token": "<unk>", "byte_fallback": True, **model}
    Path(path).write_text(json.dumps({
        "version": "1.0", "normalizer": normalizer, "model": model, "added_tokens": [],
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": True},
        "decoder": {"type": "Metaspace", "replacement": decoder_replacement, "prepend_scheme": "first", "split": True}}))
    return str(path)


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
