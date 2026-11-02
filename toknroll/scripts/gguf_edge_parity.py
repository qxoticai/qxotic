#!/usr/bin/env python3
"""llama.cpp's answers on the texts of hf_edge_parity.py (edge cases, then the test suite's), the
goldens of GGUFEdgeParityTest.

Pure tokenization, as toknroll and llama.cpp's test-tokenizer-0: no BOS or EOS, special tokens
spelled in the text are text. Ids come from llama_tokenize.c (build it first, see its header).

Usage:
  python scripts/gguf_edge_parity.py OUT.json path/to/llama_tokenize a.gguf b.gguf ...
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_edge_parity import TEXTS  # noqa: E402


def main() -> None:
    out_path, tokenize, ggufs = sys.argv[1], sys.argv[2], sys.argv[3:]
    request = b"".join(b"%d\n" % len(t.encode()) + t.encode() for t in TEXTS)
    models = []
    for gguf in ggufs:
        lines = subprocess.run([tokenize, gguf], input=request, capture_output=True,
                               check=True).stdout.decode().splitlines()
        assert len(lines) == len(TEXTS), (gguf, len(lines))
        models.append({"file": os.path.basename(gguf),
                       "cases": [None if line == "ERR" else [int(x) for x in line.split()]
                                 for line in lines]})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"texts": TEXTS, "models": models}, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
