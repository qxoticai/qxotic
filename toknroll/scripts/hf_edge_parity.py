#!/usr/bin/env python3
"""Hugging Face tokenizers' answers on edge-case texts and the test suite's texts, the goldens of
HuggingFaceEdgeParityTest.

Encodes as toknroll does by default: no special tokens added, special tokens spelled in the text
are text (encode_special_tokens). Hugging Face still matches non-special added tokens, where
toknroll encodes them as text; such a case keeps Hugging Face's ids, so the test fails on it, and
names the cause in "known".

Usage:
  python scripts/hf_edge_parity.py OUT.json owner/repo=path/to/tokenizer.json ...
"""

import json
import os
import sys

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

# then every text of the test suite's own lists, harvested once into suite_texts.json
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "suite_texts.json"),
          encoding="utf-8") as _f:
    TEXTS = EDGE_TEXTS + json.load(_f)["texts"]


def main() -> None:
    import tokenizers
    from tokenizers import Tokenizer

    out_path, specs = sys.argv[1], sys.argv[2:]
    models = []
    for spec in specs:
        ref, path = spec.split("=", 1)
        tok = Tokenizer.from_file(path)
        if hasattr(tok, "encode_special_tokens"):  # 0.2x: a setting; 1.0: an encode argument
            tok.encode_special_tokens = True
            encode = lambda text: tok.encode(text, add_special_tokens=False)
        else:
            encode = lambda text: tok.encode(text, add_special_tokens=False, encode_special_tokens=True)
        config = json.load(open(path, encoding="utf-8"))
        added = {a["id"]: a["content"] for a in config.get("added_tokens", []) if not a["special"]}
        cases = []
        for text in TEXTS:
            ids = encode(text).ids
            case = {"ids": ids, "decoded": tok.decode(ids, skip_special_tokens=False)}
            matched = sorted({added[i] for i in ids if i in added})
            if matched:
                case["known"] = ("Hugging Face matches non-special added tokens "
                                 + ", ".join(matched) + "; toknroll encodes them as text")
            cases.append(case)
        models.append({"ref": ref, "cases": cases})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tokenizers": tokenizers.__version__, "texts": TEXTS, "models": models}, f,
                  ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
