#!/usr/bin/env python3
"""Generate enwik8 tokenizer ground truth using llama.cpp llama-tokenize.

This script writes files compatible with TokenizerParityHarness:
  - chunks.json (when generated locally)
  - llamacpp_<family>_ground_truth.json

Pass --gguf-py with GGUF metadata-prefix files downloaded by Tok'n'Roll's test
cache. The script rewrites each prefix as a vocabulary-only GGUF, so neither
model weights nor the remaining shards are needed.

Example:
  python toknroll-benchmarks/generate_llamacpp_enwik8_ground_truth.py \
    --llama-tokenize /path/to/llama.cpp/build/bin/llama-tokenize \
    --families unsloth_llama3_2,google_gemma4,openai_gpt_oss \
    --gguf-paths /models/Llama-3.2-1B-Instruct-Q8_0.gguf,/models/gemma-4-E2B-it-Q8_0.gguf,/models/gpt-oss-20b-Q8_0.gguf
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List

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


@contextmanager
def vocabulary_only_gguf(
    metadata_path: Path, gguf_py_path: Path | None
) -> Iterator[Path]:
    """Turn a downloaded GGUF metadata prefix into a tiny llama.cpp-loadable GGUF."""
    if metadata_path.suffix != ".metadata":
        yield metadata_path
        return
    if gguf_py_path is None:
        raise RuntimeError("--gguf-py is required for .metadata inputs")

    if not gguf_py_path.is_dir():
        raise RuntimeError(f"llama.cpp gguf-py directory not found: {gguf_py_path}")
    sys.path.insert(0, str(gguf_py_path))
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError(f"Cannot import gguf from {gguf_py_path}") from exc
    finally:
        sys.path.pop(0)

    with tempfile.TemporaryDirectory(prefix="toknroll-vocab-gguf-") as temp_dir:
        tensorless_input = Path(temp_dir) / "metadata.gguf"
        shutil.copyfile(metadata_path, tensorless_input)
        with tensorless_input.open("r+b") as output:
            if output.read(4) != b"GGUF":
                raise RuntimeError(f"Not a GGUF file: {metadata_path}")
            output.seek(8)
            output.write(b"\0" * 8)

        reader = gguf.GGUFReader(tensorless_input, "r")
        architecture = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
        model_path = Path(temp_dir) / "vocabulary.gguf"
        writer = gguf.GGUFWriter(
            model_path, arch=architecture, endianess=reader.endianess
        )
        alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
        if alignment is not None:
            writer.data_alignment = alignment.contents()

        for field in reader.fields.values():
            if (
                field.name == gguf.Keys.General.ARCHITECTURE
                or field.name.startswith("GGUF.")
                or field.name.startswith("split.")
            ):
                continue
            value_type = field.types[0]
            sub_type = (
                field.types[-1]
                if value_type == gguf.GGUFValueType.ARRAY
                else None
            )
            writer.add_key_value(
                field.name, field.contents(), value_type, sub_type=sub_type
            )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        writer.close()
        yield model_path


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
        help="Comma-separated local GGUF or metadata-prefix file paths",
    )
    parser.add_argument(
        "--gguf-py",
        default="",
        help=(
            "Path to llama.cpp/gguf-py; converts metadata prefixes into "
            "vocabulary-only GGUFs"
        ),
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
    gguf_py_path = (
        Path(args.gguf_py).expanduser().resolve() if args.gguf_py.strip() else None
    )
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
        with vocabulary_only_gguf(gguf_path, gguf_py_path) as model_path:
            for idx, chunk in enumerate(chunks):
                text = chunk.get("text", "")
                chunk_hash = chunk.get("hash")
                if not isinstance(chunk_hash, str) or not chunk_hash:
                    raise RuntimeError(f"Invalid chunk hash at index {idx}")

                tokens = tokenize_with_llamacpp(llama_tokenize, model_path, text)
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
