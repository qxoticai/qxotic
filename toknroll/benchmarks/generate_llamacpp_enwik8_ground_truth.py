#!/usr/bin/env python3
"""Generate enwik8 tokenizer ground truth using llama.cpp llama-tokenize.

This script writes files compatible with TokenizerParityHarness:
  - chunks.json (when generated locally)
  - llamacpp_<family>_ground_truth.json

Example:
  python benchmarks/generate_llamacpp_enwik8_ground_truth.py \
    --llama-tokenize /home/mukel/Desktop/playground/llama.cpp/build/bin/llama-tokenize \
    --families unsloth_llama3_2,google_gemma4,openai_gpt_oss \
    --gguf-paths /models/Llama-3.2-1B-Instruct-Q8_0.gguf,/models/gemma-4-E2B-it-Q8_0.gguf,/models/gpt-oss-20b-Q8_0.gguf
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from generate_enwik8_ground_truth import (
    compute_chunk_hash,
    download_enwik8,
    generate_chunks,
    get_cache_dir,
)


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_or_generate_chunks(
    chunks_path: Path,
    cache_root_override: str | None,
    chunk_sizes: List[int],
    samples_per_size: int,
) -> List[Dict[str, Any]]:
    if chunks_path.exists():
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise RuntimeError(f"chunks file must be a JSON list: {chunks_path}")
        return data

    cache_dir = get_cache_dir(cache_root_override=cache_root_override)
    enwik8_path = download_enwik8(cache_dir)
    corpus_bytes = enwik8_path.read_bytes()
    chunks = generate_chunks(corpus_bytes, chunk_sizes, samples_per_size)

    rows: List[Dict[str, Any]] = []
    for offset, size, chunk_data in chunks:
        rows.append(
            {
                "offset": offset,
                "size": size,
                "hash": compute_chunk_hash(offset, size),
                "text": chunk_data.decode("utf-8", errors="replace"),
            }
        )

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


def tokenize_with_llamacpp(
    llama_tokenize: Path, model_path: Path, text: str
) -> List[int]:
    cmd = [
        str(llama_tokenize),
        "--model",
        str(model_path),
        "--stdin",
        "--ids",
        "--no-escape",
        "--no-bos",
        "--no-parse-special",
        "--log-disable",
    ]
    proc = subprocess.run(
        cmd,
        input=text,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "llama-tokenize failed for "
            + str(model_path)
            + f" (exit={proc.returncode}): {proc.stderr.strip()}"
        )
    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError("llama-tokenize produced empty output")
    parsed = json.loads(stdout)
    if not isinstance(parsed, list):
        raise RuntimeError(f"llama-tokenize output is not a JSON list: {stdout[:120]}")
    return [int(v) for v in parsed]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate enwik8 ground truth files using llama.cpp tokenizer"
    )
    parser.add_argument(
        "--llama-tokenize",
        required=True,
        help="Path to llama-tokenize binary",
    )
    parser.add_argument(
        "--output-dir",
        default="toknroll-core/src/test/resources/golden/enwik8",
        help="Output directory for ground truth files",
    )
    parser.add_argument(
        "--chunks-file",
        default="",
        help="Path to chunks.json (defaults to <output-dir>/chunks.json)",
    )
    parser.add_argument(
        "--families",
        required=True,
        help="Comma-separated family IDs (used in output filenames)",
    )
    parser.add_argument(
        "--gguf-paths",
        required=True,
        help="Comma-separated local GGUF model file paths",
    )
    parser.add_argument(
        "--chunk-sizes",
        default="256,1024,4096,16384",
        help="Used only when chunks file does not exist",
    )
    parser.add_argument(
        "--samples-per-size",
        type=int,
        default=20,
        help="Used only when chunks file does not exist",
    )
    parser.add_argument(
        "--cache-root",
        default="",
        help="Optional cache root override for enwik8 download",
    )

    args = parser.parse_args()
    llama_tokenize = Path(args.llama_tokenize).expanduser().resolve()
    if not llama_tokenize.exists():
        raise RuntimeError(f"llama-tokenize binary not found: {llama_tokenize}")

    families = parse_csv(args.families)
    gguf_paths = [Path(p).expanduser().resolve() for p in parse_csv(args.gguf_paths)]
    if len(families) != len(gguf_paths):
        raise RuntimeError("--families and --gguf-paths must have the same item count")

    for gguf_path in gguf_paths:
        if not gguf_path.exists():
            raise RuntimeError(f"GGUF model file not found: {gguf_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_file = (
        Path(args.chunks_file).expanduser().resolve()
        if args.chunks_file.strip()
        else output_dir / "chunks.json"
    )

    chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(",") if x.strip()]
    chunks = load_or_generate_chunks(
        chunks_file,
        args.cache_root.strip() or None,
        chunk_sizes,
        args.samples_per_size,
    )
    print(f"Loaded {len(chunks)} chunks from {chunks_file}")

    for family_id, gguf_path in zip(families, gguf_paths):
        print(
            f"Generating llama.cpp ground truth for {family_id} ({gguf_path.name})..."
        )
        results = []
        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            chunk_hash = chunk.get("hash")
            if not isinstance(chunk_hash, str) or not chunk_hash:
                raise RuntimeError(f"Invalid chunk hash at index {idx}")

            tokens = tokenize_with_llamacpp(llama_tokenize, gguf_path, text)
            results.append(
                {
                    "chunk_hash": chunk_hash,
                    "tokens": tokens,
                    "token_count": len(tokens),
                }
            )

        out_file = output_dir / f"llamacpp_{family_id}_ground_truth.json"
        out_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote {len(results)} chunks -> {out_file}")


if __name__ == "__main__":
    main()
