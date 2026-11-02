#!/usr/bin/env python3
"""Hugging Face tokenizers' answers on edge-case texts and the test suite's texts, the goldens of
HuggingFaceEdgeParityTest, in toknroll's two modes. No special tokens are added either way.

- "ids" and "decoded", for plain encode, raw text: the tokenizer.json without its added tokens, so
  Hugging Face runs only its normalizer, pre-tokenizer and model, and matches no added token.
- "specials", for Specials.encode, every added token matched: the tokenizer.json as it is.

Each supported model's tokenizer.json is fetched at its pinned revision (HF_TOKEN, or the token in
the Hugging Face home, for gated repositories). The goldens were made with tokenizers 0.23.2; the
script refuses another version.

Usage:
  python -m venv .venv && .venv/bin/pip install tokenizers==0.23.2
  .venv/bin/python scripts/hf_edge_parity.py [OUT.json]
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

TOKENIZERS_VERSION = "0.23.2"

# Every supported tokenizer.json model, pinned. phi-4 is not supported: its added tokens set
# lstrip/rstrip, which toknroll refuses.
MODELS = [
    ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    ("Xenova/gpt-4", "1d9f1f1b1fae88c0e4df1dab0a397f8de6229075"),
    ("Xenova/gpt-4o", "7956d98f2a83b2751a98ea7136fdf7fe6cf54e69"),
    ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
    ("unsloth/Llama-3.2-1B-Instruct", "5a8abab4a5d6f164389b1079fb721cfab8d7126c"),
    ("HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1"),
    ("ibm-granite/granite-4.0-h-1b", "d18cca4c121edb87d022116d281ce212c9136f57"),
    ("deepseek-ai/DeepSeek-V3.2", "a7e62ac04ecb2c0a54d736dc46601c5606cf10a6"),
    ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    ("google/gemma-4-e2b-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71"),
    ("mistralai/ministral-8b-instruct-2410", "2f494a194c5b980dfb9772cb92d26cbb671fce5a"),
    ("poolside/Laguna-XS-2.1", "c5f36269bbdbd3f27fddc9a9f9dbae0cf2cf57db"),
    ("openbmb/MiniCPM5-2B", "f97400052a43d642bbc6e9975e2397e3ae6a6b52"),
    ("inclusionAI/Ling-3.0-tiny", "9a98e35fe1c9ee255f78dd64771c7ae15a799481"),
    ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    ("LiquidAI/LFM2.5-350M", "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"),
    ("LiquidAI/LFM3-1.7B", "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"),
    ("deepseek-ai/DeepSeek-V4-Pro", "b5968e9190ef611bbf34a7229255be88a0e937c1"),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "314113f8a7dadb388215396ba5fbee2c610a04d6"),
    ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"),
    ("poolside/Laguna-S-2.1", "0f573140834b11cfac0c2af97a101a7a69a13e22"),
    ("inclusionAI/Ling-3.1-flash", "276cca72bbd8b95693032848bfd638460b529b8b"),
]

EDGE_TEXTS = [
    "", " ", "  ", "\n", "\n\n", "\r\n", "\r", "a\r\nb", "a\rb", "a\n", "a\n\n", "\na",
    " leading space", "  two leading", "trailing space ", "trailing\n", "\ttab", "x  y   z",
    "▁", "▁literal metaspace", "a▁b", "Hello world", "hello\nworld\n",
    "123 4567 89012", "1,234.56", "2026-11-02T06:04:55", "don't you'll we'VE I'M",
    "NÁT ESTAÇÃO GARCÍA FÜR", "АвтоВАЗ Москва", "日本語のテキスト", "中文混合English",
    "한국어 텍스트", "ภาษาไทย", "العربية mixed", "हिन्दी", "😀🫨👍🏽 family 👨‍👩‍👧",
    "café naïve", "é combining", " nbsp em　ideo", "​zero‍width",
    "<|im_start|>user", "<|endoftext|>", "<s>x</s>", "[CLS] [SEP] [MASK]", "<unk>", "<0x41>",
    "<think>hi</think>", "/think /no_think", "<start_of_turn>", "<|begin_of_text|>",
    "def f(x):\n    return x  # c\n", "    indented\n\tmixed", "http://example.com/a?b=c&d=e",
    "a" * 300, "ab " * 100, " " * 50, "\n" * 20, "🫨" * 30, "\x00\x01\x7f control",
    "� replacement", "x" + "́" * 20,
]

# then every text of the test suite's own lists (suite_texts.json says how they were harvested)
with open(Path(__file__).resolve().parent / "suite_texts.json", encoding="utf-8") as _f:
    TEXTS = EDGE_TEXTS + json.load(_f)["texts"]


def fetch(model, revision, path):
    request = urllib.request.Request(f"https://huggingface.co/{model}/resolve/{revision}/{path}")
    token = os.environ.get("HF_TOKEN") or read_token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def read_token():
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    return path.read_text().strip() if path.is_file() else None


def main() -> None:
    import tokenizers
    from tokenizers import Tokenizer

    if tokenizers.__version__ != TOKENIZERS_VERSION:
        sys.exit(f"tokenizers {tokenizers.__version__}: the goldens need {TOKENIZERS_VERSION}")
    default = Path(__file__).resolve().parents[1] / "toknroll-hf/src/test/resources/hf_edge_parity.json"
    out_path = sys.argv[1] if len(sys.argv) > 1 else default
    models = []
    for ref, revision in MODELS:
        source = fetch(ref, revision, "tokenizer.json")
        raw = json.loads(source)
        raw["added_tokens"] = []
        plain, full = Tokenizer.from_str(json.dumps(raw)), Tokenizer.from_str(source)
        cases = []
        for text in TEXTS:
            ids = plain.encode(text, add_special_tokens=False).ids
            cases.append({"ids": ids, "decoded": plain.decode(ids, skip_special_tokens=False),
                          "specials": full.encode(text, add_special_tokens=False).ids})
        models.append({"ref": ref, "revision": revision, "cases": cases})
        print(f"{ref:40} {len(cases)} cases")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tokenizers": tokenizers.__version__, "texts": TEXTS, "models": models}, f,
                  ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
