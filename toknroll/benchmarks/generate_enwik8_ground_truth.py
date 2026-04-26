#!/usr/bin/env python3
"""
Generate ground truth tokenization outputs for enwik8 corpus chunks.

Usage:
    python generate_enwik8_ground_truth.py [--chunks-dir DIR] [--output-dir DIR] [--families FAMILY1,FAMILY2] [--encodings ENC1,ENC2]

This script:
1. Loads enwik8 corpus (downloads if needed)
2. Generates chunks of various sizes
3. Tokenizes each chunk using both tiktoken and HuggingFace tokenizers
4. Saves ground truth JSON files for Java comparison tests

Requirements:
    pip install tiktoken transformers torch
"""

import argparse
import json
import hashlib
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cache_paths import resolve_under_test_artifacts


def get_cache_dir(cache_root_override: Optional[str] = None) -> Path:
    """Get the cache directory for enwik8 data."""
    cache = resolve_under_test_artifacts("corpus", override=cache_root_override)
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def download_enwik8(cache_dir: Path) -> Path:
    """Download and extract enwik8 if not already cached."""
    zip_path = cache_dir / "enwik8.zip"
    raw_path = cache_dir / "enwik8"

    if raw_path.exists() and raw_path.stat().st_size == 100_000_000:
        print(f"Using cached enwik8 at {raw_path}")
        return raw_path

    if not zip_path.exists():
        url = "https://www.mattmahoney.net/dc/enwik8.zip"
        print(f"Downloading enwik8 from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(cache_dir)

    if raw_path.exists() and raw_path.stat().st_size == 100_000_000:
        print(f"Extracted successfully to {raw_path}")
        return raw_path
    else:
        raise RuntimeError(
            f"Extraction failed or wrong size: {raw_path.stat().st_size if raw_path.exists() else 'missing'}"
        )


def generate_chunks(
    data: bytes, chunk_sizes: List[int], samples_per_size: int
) -> List[Tuple[int, int, bytes]]:
    """Generate chunks from enwik8 data.

    Returns list of (offset, size, bytes) tuples.
    Uses deterministic offsets based on data size.
    """
    chunks = []
    total_size = len(data)

    for size in chunk_sizes:
        stride = max(1, (total_size - size) // samples_per_size)
        for i in range(samples_per_size):
            offset = min(i * stride, total_size - size)
            # Ensure we don't split UTF-8 sequences - find valid start
            while offset > 0 and (data[offset] & 0xC0) == 0x80:
                offset -= 1
            # Ensure we end at a valid UTF-8 boundary
            end = offset + size
            while end < total_size and (data[end] & 0xC0) == 0x80:
                end += 1
            chunks.append((offset, end - offset, data[offset:end]))

    return chunks


def get_tiktoken_encodings() -> Dict[str, Any]:
    """Get available tiktoken encodings."""
    import tiktoken

    return {
        "r50k_base": tiktoken.get_encoding("r50k_base"),
        "cl100k_base": tiktoken.get_encoding("cl100k_base"),
        "o200k_base": tiktoken.get_encoding("o200k_base"),
    }


def tokenize_with_tiktoken(text: str, encoding) -> Tuple[List[int], str]:
    """Tokenize text using tiktoken and return tokens + decoded."""
    tokens = encoding.encode(text)
    decoded = encoding.decode(tokens)
    return tokens, decoded


def get_hf_tokenizer(
    model_name: str, revision: Optional[str] = None, cache_root: Optional[Path] = None
):
    """Get HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    kwargs: Dict[str, object] = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    try:
        return AutoTokenizer.from_pretrained(model_name, **kwargs)
    except Exception as e:
        print(f"  Warning: Failed to load tokenizer via AutoTokenizer: {e}")
        print(f"  Falling back to tokenizer.json direct load...")
        return load_tokenizer_from_json(model_name, revision, cache_root)


def load_tokenizer_from_json(
    model_name: str, revision: Optional[str] = None, cache_root: Optional[Path] = None
):
    """Load tokenizer directly from tokenizer.json file."""
    import requests

    rev = revision or "main"
    url = f"https://huggingface.co/{model_name}/resolve/{rev}/tokenizer.json"

    headers = {}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    effective_cache_root = cache_root or resolve_under_test_artifacts()
    tokenizer_json_cache = (
        effective_cache_root / "ground-truth" / "downloads" / "tokenizer-json"
    )
    tokenizer_json_cache.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
    json_path = tokenizer_json_cache / f"{url_hash}.json"

    if not json_path.exists():
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download tokenizer.json from {url}: {response.status_code}"
            )
        json_path.write_bytes(response.content)

    from tokenizers import Tokenizer as HFTokenizer

    tokenizer = HFTokenizer.from_file(str(json_path))

    # Wrap to match AutoTokenizer interface
    class TokenizerWrapper:
        def __init__(self, inner):
            self.inner = inner

        def encode(self, text, add_special_tokens=False):
            encoded = self.inner.encode(text)
            return encoded.ids

        def decode(self, tokens, clean_up_tokenization_spaces=False):
            return self.inner.decode(tokens)

    return TokenizerWrapper(tokenizer)


def tokenize_with_hf(text: str, tokenizer) -> Tuple[List[int], str]:
    """Tokenize text using HuggingFace tokenizer and return tokens + decoded."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    # Use clean_up_tokenization_spaces=False to get raw BPE output,
    # matching llama.cpp behavior and avoiding model-specific post-processing differences.
    decoded = tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
    return tokens, decoded


def compute_chunk_hash(offset: int, size: int) -> str:
    """Compute a unique hash for a chunk."""
    return hashlib.sha256(f"{offset}:{size}".encode()).hexdigest()[:16]


def generate_ground_truth(
    enwik8_path: Path,
    output_dir: Path,
    encodings: List[str],
    hf_models: Dict[str, Tuple[str, Optional[str]]],
    chunk_sizes: List[int],
    samples_per_size: int,
    cache_root: Optional[Path] = None,
) -> None:
    """Generate ground truth files for all tokenizers."""

    print("Loading enwik8...")
    with open(enwik8_path, "rb") as f:
        data = f.read()

    print(f"Loaded {len(data)} bytes")

    # Generate chunks
    print("Generating chunks...")
    chunks = generate_chunks(data, chunk_sizes, samples_per_size)
    print(f"Generated {len(chunks)} chunks")

    # Save chunks metadata
    chunks_meta = []
    for offset, size, chunk_data in chunks:
        chunk_hash = compute_chunk_hash(offset, size)
        try:
            text = chunk_data.decode("utf-8", errors="replace")
        except:
            text = chunk_data.decode("utf-8", errors="replace")
        chunks_meta.append(
            {"offset": offset, "size": size, "hash": chunk_hash, "text": text}
        )

    chunks_file = output_dir / "chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks metadata to {chunks_file}")

    # Tiktoken ground truth
    if encodings:
        print("\nGenerating tiktoken ground truth...")
        import tiktoken

        tiktoken_encodings = get_tiktoken_encodings()

        for enc_name in encodings:
            if enc_name not in tiktoken_encodings:
                print(f"  Warning: Unknown tiktoken encoding '{enc_name}', skipping")
                continue

            encoding = tiktoken_encodings[enc_name]
            results = []

            for i, (offset, size, chunk_data) in enumerate(chunks):
                text = chunks_meta[i]["text"]
                try:
                    tokens, decoded = tokenize_with_tiktoken(text, encoding)
                    results.append(
                        {
                            "chunk_hash": chunks_meta[i]["hash"],
                            "tokens": tokens,
                            "decoded": decoded,
                            "token_count": len(tokens),
                        }
                    )
                except Exception as e:
                    print(f"  Error tokenizing chunk {i} with {enc_name}: {e}")
                    results.append(
                        {"chunk_hash": chunks_meta[i]["hash"], "error": str(e)}
                    )

            output_file = output_dir / f"tiktoken_{enc_name}_ground_truth.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(
                f"  Saved {enc_name} ground truth ({len(results)} chunks) to {output_file}"
            )

    # HuggingFace ground truth
    if hf_models:
        print("\nGenerating HuggingFace ground truth...")

        for family_id, (model_name, revision) in hf_models.items():
            print(f"  Loading {family_id} ({model_name})...")
            try:
                tokenizer = get_hf_tokenizer(
                    model_name, revision, cache_root=cache_root
                )
                results = []

                for i, (offset, size, chunk_data) in enumerate(chunks):
                    text = chunks_meta[i]["text"]
                    try:
                        tokens, decoded = tokenize_with_hf(text, tokenizer)
                        results.append(
                            {
                                "chunk_hash": chunks_meta[i]["hash"],
                                "tokens": tokens,
                                "decoded": decoded,
                                "token_count": len(tokens),
                            }
                        )
                    except Exception as e:
                        print(f"    Error tokenizing chunk {i}: {e}")
                        results.append(
                            {"chunk_hash": chunks_meta[i]["hash"], "error": str(e)}
                        )

                safe_name = family_id.replace(".", "_")
                output_file = output_dir / f"hf_{safe_name}_ground_truth.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(
                    f"  Saved {family_id} ground truth ({len(results)} chunks) to {output_file}"
                )

            except Exception as e:
                print(f"  Error loading tokenizer for {family_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate enwik8 ground truth tokenization data"
    )
    parser.add_argument(
        "--output-dir",
        default="toknroll-core/src/test/resources/golden/enwik8",
        help="Output directory for ground truth files",
    )
    parser.add_argument(
        "--chunk-sizes",
        default="256,1024,4096,16384",
        help="Comma-separated chunk sizes",
    )
    parser.add_argument(
        "--samples-per-size",
        type=int,
        default=20,
        help="Number of samples per chunk size",
    )
    parser.add_argument(
        "--encodings",
        default="r50k_base,cl100k_base,o200k_base",
        help="Comma-separated tiktoken encodings",
    )
    parser.add_argument(
        "--families",
        default="meta.llama3,alibaba.qwen3_5,google.gemma4",
        help="Comma-separated HF model families",
    )
    parser.add_argument(
        "--hf-model-refs",
        default="meta-llama/Llama-3.2-1B-Instruct,Qwen/Qwen3.5-0.8B,google/gemma-4-e2b-it",
        help="Comma-separated HF model references",
    )
    parser.add_argument(
        "--hf-revisions",
        default=",,",
        help="Comma-separated HF revisions (empty for default)",
    )
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("TOKNROLL_TEST_CACHE_ROOT", ""),
        help="Cache root override (defaults to OS-specific qxotic/toknroll/test-artifacts)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(",")]
    encodings = [x.strip() for x in args.encodings.split(",") if x.strip()]
    families = [x.strip() for x in args.families.split(",") if x.strip()]
    model_refs = [x.strip() for x in args.hf_model_refs.split(",") if x.strip()]
    revisions = [x.strip() if x.strip() else None for x in args.hf_revisions.split(",")]

    if len(families) != len(model_refs):
        print(
            "Error: Number of families must match number of model refs", file=sys.stderr
        )
        sys.exit(1)

    # Pad revisions if needed
    while len(revisions) < len(families):
        revisions.append(None)

    hf_models = {}
    for family, model_ref, revision in zip(families, model_refs, revisions):
        hf_models[family] = (model_ref, revision)

    cache_root_override = args.cache_root.strip() or None
    cache_root = resolve_under_test_artifacts(override=cache_root_override)

    cache_dir = get_cache_dir(cache_root_override=cache_root_override)
    enwik8_path = download_enwik8(cache_dir)

    generate_ground_truth(
        enwik8_path=enwik8_path,
        output_dir=output_dir,
        encodings=encodings,
        hf_models=hf_models,
        chunk_sizes=chunk_sizes,
        samples_per_size=args.samples_per_size,
        cache_root=cache_root,
    )

    print("\nGround truth generation complete!")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
