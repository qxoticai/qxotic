#!/usr/bin/env python3
"""Download OpenAI tiktoken fixture files used by tests.

Writes files to test-fixtures/tiktoken by default.
"""

import argparse
import pathlib
import urllib.error
import urllib.request

BASE_URL = "https://openaipublic.blob.core.windows.net/encodings"
FILES = ("r50k_base.tiktoken", "p50k_base.tiktoken", "cl100k_base.tiktoken", "o200k_base.tiktoken")


def download_one(target_dir: pathlib.Path, name: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    tmp = target.with_suffix(target.suffix + ".part")
    url = f"{BASE_URL}/{name}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"ERROR: failed to download {name} from {url}: {exc}") from exc

    if not data:
        raise SystemExit(f"ERROR: downloaded empty fixture for {name} from {url}")

    tmp.write_bytes(data)
    tmp.replace(target)
    print(f"downloaded {name} ({len(data)} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download tiktoken test fixtures")
    parser.add_argument(
        "--output-dir",
        default="test-fixtures/tiktoken",
        help="Directory to write .tiktoken files (default: test-fixtures/tiktoken)",
    )
    args = parser.parse_args()

    out = pathlib.Path(args.output_dir)
    for name in FILES:
        download_one(out, name)

    missing = [name for name in FILES if not (out / name).is_file()]
    if missing:
        raise SystemExit(f"ERROR: missing downloaded fixtures: {missing}")


if __name__ == "__main__":
    main()
