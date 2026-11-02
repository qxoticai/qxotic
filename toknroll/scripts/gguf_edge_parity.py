#!/usr/bin/env python3
"""llama.cpp's answers on the texts of hf_edge_parity.py (edge cases, then the test suite's), the
goldens of GGUFEdgeParityTest.

Pure tokenization, as toknroll and llama.cpp's test-tokenizer-0: no BOS or EOS, special tokens
spelled in the text are text. Ids come from llama_tokenize.c (build it first, see its header).
A text llama.cpp refuses fails the run.

Usage:
  python scripts/gguf_edge_parity.py path/to/llama_tokenize GGUF_DIR [OUT.json]
"""

import json
import os
import subprocess
import sys
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


def main() -> None:
    tokenize, ggufs = sys.argv[1], Path(sys.argv[2])
    default = Path(__file__).resolve().parents[1] / "toknroll-gguf/src/test/resources/gguf_edge_parity.json"
    out_path = sys.argv[3] if len(sys.argv) > 3 else default
    request = b"".join(b"%d\n" % len(t.encode()) + t.encode() for t in TEXTS)
    models = []
    for name in FILES:
        lines = subprocess.run([tokenize, ggufs / name], input=request, capture_output=True,
                               check=True).stdout.decode().splitlines()
        if len(lines) != len(TEXTS):
            sys.exit(f"{name}: {len(lines)} answers for {len(TEXTS)} texts")
        refused = [i for i, line in enumerate(lines) if line == "ERR"]
        if refused:
            sys.exit(f"{name}: llama.cpp refuses texts {refused}")
        models.append({"file": name, "cases": [[int(x) for x in line.split()] for line in lines]})
        print(f"{name:56} {len(lines)} cases")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"texts": TEXTS, "models": models}, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
