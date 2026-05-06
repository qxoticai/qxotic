#!/usr/bin/env python3
"""Tokenize text with a HuggingFace tokenizer. Outputs one token ID per line.

Usage:
  python scripts/hf.py -s unsloth/Gemma-4-E2B-it -t "Hello World"
  python scripts/hf.py -s unsloth/Gemma-4-E2B-it --decode < ids.txt
  echo "Hello World" | python scripts/hf.py -s unsloth/Gemma-4-E2B-it
"""

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize text with a HuggingFace tokenizer")
    parser.add_argument(
        "-s", "--source", required=True, help="Model: org/repo or hf:org/repo"
    )
    parser.add_argument("-t", "--text", help="Text to tokenize (reads stdin by default)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-e", "--encode", action="store_true",
                      help="Encode text to token IDs (default)")
    mode.add_argument("-d", "--decode", action="store_true",
                      help="Decode token IDs to text")
    mode.add_argument("-c", "--count", action="store_true",
                      help="Print token count only")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print timing information to stderr")
    args = parser.parse_args()

    model_id = args.source.removeprefix("hf:").removeprefix("huggingface:").strip()
    text = args.text if args.text is not None else sys.stdin.read()

    from transformers import AutoTokenizer

    t_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    t_load = time.perf_counter() - t_load

    t_op = time.perf_counter()
    if args.decode:
        ids = [int(x) for x in text.strip().split()]
        if not ids:
            print("error: no token IDs provided for decode", file=sys.stderr)
            sys.exit(1)
        output = tokenizer.decode(ids)
        t_op = time.perf_counter() - t_op
        print(output)
        op = "decode"
    elif args.count:
        ids = tokenizer.encode(text, add_special_tokens=False)
        t_op = time.perf_counter() - t_op
        print(len(ids))
        op = "count"
    else:
        ids = tokenizer.encode(text, add_special_tokens=False)
        t_op = time.perf_counter() - t_op
        for tid in ids:
            print(tid)
        op = "encode"

    if args.verbose:
        print(f"load= {t_load * 1000:5.0f}ms  {op}= {t_op * 1000:5.0f}ms  "
              f"tokens= {len(ids)}", file=sys.stderr)


if __name__ == "__main__":
    main()
