#!/usr/bin/env python3
"""llama.cpp's answers on the texts of hf_edge_parity.py (edge cases, then the test suite's), the
goldens of GGUFEdgeParityTest, in toknroll's two modes. No BOS or EOS either way.

- "ids", for plain encode, raw text: llama.cpp matches every user-defined token even without
  parse_special, so it runs on a copy of the file whose user-defined tokens are typed normal; its
  control tokens stay text, as parse_special is off.
- "specials", for Specials.encode, every added token matched: the file itself, parse_special on.

Ids come from llama_tokenize.c (build it first, see its header); the copies need the gguf package
(llama.cpp's gguf-py). A text llama.cpp refuses fails the run.

Usage:
  python scripts/gguf_edge_parity.py path/to/llama_tokenize GGUF_DIR [OUT.json]
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_edge_parity import TEXTS  # noqa: E402

# The supported GGUFs, llama.cpp's vocab-only test files and real models (their first shard),
# by file name in GGUF_DIR.
FILES = [
    "gemma-4-E2B-it-Q8_0.gguf",
    "ggml-vocab-gemma-4.gguf",
    "ggml-vocab-gpt-2.gguf",
    "ggml-vocab-llama-bpe.gguf",
    "ggml-vocab-llama-spm.gguf",
    "ggml-vocab-qwen2.gguf",
    "ggml-vocab-qwen35.gguf",
    "gpt-oss-20b-Q8_0.gguf",
    "granite-4.2-3b-Q4_0.gguf",
    "Laguna-XS-2.1-Q4_K_M.gguf",
    "LFM2.5-230M-Q4_0.gguf",
    "Ling-3.0-tiny-Q4_K_M.gguf",
    "Mellum2-12B-A2.5B-Instruct-MXFP4_MOE.gguf",
    "MiniCPM5-1B-Q4_K_M.gguf",
    "MiniCPM5-2B-Q4_K_M.gguf",
    "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf",
    "nvidia_Nemotron-Cascade-2-30B-A3B-Q4_K_M.gguf",
    "Qwen3.5-0.8B-Q4_K_M.gguf",
    "Qwen3-Embedding-0.6B-Q8_0.gguf",
    "SmolLM3-Q4_K_M.gguf",
    "LFM3-1.7B-Q8_0.gguf",
    "Ling-3.1-flash-Q4_0-00001-of-00009.gguf",
    "Qwen3.6-27B-Q4_K_M.gguf",
    "laguna-s-2.1-Q4_K_M.gguf",
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf",
    "DeepSeek-V4-Pro-0813-UD-Q4_K_XL-00001-of-00020.gguf",
    "Kimi-K2.6-UD-Q4_K_XL-00001-of-00014.gguf",
    "Kimi-K3-UD-IQ1_M-00001-of-00015.gguf",
]


def raw_text_copy(gguf, into):
    """A copy of the file whose user-defined tokens are typed normal: its header, the rest left a
    hole, beside links to the file's other shards, which llama.cpp opens by name."""
    import gguf as gguf_py

    copy = into / gguf.name
    with open(gguf, "rb") as src, open(copy, "wb") as dst:
        dst.write(src.read(gguf_py.GGUFReader(gguf).data_offset))
        dst.truncate(gguf.stat().st_size)
    reader = gguf_py.GGUFReader(copy, "r+")
    field = reader.fields["tokenizer.ggml.token_type"]
    for i in field.data:
        if field.parts[i][0] == gguf_py.TokenType.USER_DEFINED:
            field.parts[i][0] = gguf_py.TokenType.NORMAL
    reader.data.flush()
    if "-00001-of-" in gguf.name:
        prefix = gguf.name.split("-00001-of-")[0]
        for shard in gguf.parent.glob(prefix + "-0*-of-*.gguf"):
            if shard.name != gguf.name:
                (into / shard.name).symlink_to(shard.resolve())
    return copy


def answers(tokenize, gguf, request, *flags):
    lines = subprocess.run([tokenize, gguf, *flags], input=request, capture_output=True,
                           check=True).stdout.decode().splitlines()
    if len(lines) != len(TEXTS):
        sys.exit(f"{gguf.name}: {len(lines)} answers for {len(TEXTS)} texts")
    refused = [i for i, line in enumerate(lines) if line == "ERR"]
    if refused:
        sys.exit(f"{gguf.name}: llama.cpp refuses texts {refused}")
    return [[int(x) for x in line.split()] for line in lines]


def main() -> None:
    tokenize, ggufs = sys.argv[1], Path(sys.argv[2])
    default = Path(__file__).resolve().parents[1] / "toknroll-gguf/src/test/resources/gguf_edge_parity.json"
    out_path = sys.argv[3] if len(sys.argv) > 3 else default
    request = b"".join(b"%d\n" % len(t.encode()) + t.encode() for t in TEXTS)
    models = []
    for name in FILES:
        with tempfile.TemporaryDirectory() as tmp:
            plain = answers(tokenize, raw_text_copy(ggufs / name, Path(tmp)), request)
        specials = answers(tokenize, ggufs / name, request, "special")
        cases = [{"ids": ids, "specials": sp} for ids, sp in zip(plain, specials)]
        models.append({"file": name, "cases": cases})
        print(f"{name:56} {len(cases)} cases")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"texts": TEXTS, "models": models}, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
