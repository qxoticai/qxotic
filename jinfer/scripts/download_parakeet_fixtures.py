#!/usr/bin/env python3
"""Download the speech clips the Parakeet end-to-end tests read, from their public sources:
sherpa-onnx's Parakeet test speech and three LibriSpeech test-clean utterances, each pinned to a
repository revision and checked against its SHA-256 (the golden transcripts depend on the exact
audio).

Writes files to test-fixtures/parakeet by default.
"""

import argparse
import hashlib
import pathlib
import urllib.error
import urllib.request

HF = "https://huggingface.co/csukuangfj"
V3 = f"{HF}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/2bda32ec70b097a55adaa07d9a7173915b43cc78/test_wavs"
V2 = f"{HF}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2/resolve/86891485dd8ad7cb28cb1aade45c3e23d0197c30/test_wavs"
ZIPFORMER = f"{HF}/sherpa-onnx-zipformer-en-2023-06-26/resolve/11a70a47ec5fd969c17b8d0818681d222a147781/test_wavs"

# file name -> (url, sha256)
CLIPS = {
    "en.wav": (f"{V3}/en.wav", "148b936b43ce7c546a866e64da059f0458aee2d65e617f16e9d94f06e8d99ed6"),
    "de.wav": (f"{V3}/de.wav", "36d3c4845b9808a1656a2a2e92d884590e2db94389e6fe559643291ae0cd3710"),
    "es.wav": (f"{V3}/es.wav", "49fd2cfa4b62db7068143c582b35de9d31ec2733495ece3611105131d21de06c"),
    "fr.wav": (f"{V3}/fr.wav", "b59be4349b92d344fb903677165eaf4694025d1ab119c608726ecbcb3164b528"),
    "en0.wav": (f"{V2}/0.wav", "5fceacff0315d49cb59fcc505bcecf1ed5f2f35c2897b1e65a59f30e5d922150"),
    "ls0.wav": (f"{ZIPFORMER}/0.wav", "6bc58a4efdf20daac252b6b1502632601a71efe0308f6757dc1eda34891a7e4f"),
    "ls1.wav": (f"{ZIPFORMER}/1.wav", "5143a6ba93c4b274e2c4ac22deb75c2c48936c853f0519add1de828b6c79cc5a"),
    "ls8k.wav": (f"{ZIPFORMER}/8k.wav", "f6f3c8b33e2534cdc154fe773ad2750f1f6a2ca5096179cdf037ae782456613e"),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_one(target_dir: pathlib.Path, name: str) -> None:
    url, expected = CLIPS[name]
    target = target_dir / name
    if target.is_file() and sha256(target.read_bytes()) == expected:
        print(f"present {name}")
        return
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"ERROR: failed to download {name} from {url}: {exc}") from exc
    actual = sha256(data)
    if actual != expected:
        raise SystemExit(f"ERROR: {name} from {url} has sha256 {actual}, expected {expected}")
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(target)
    print(f"downloaded {name} ({len(data)} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Parakeet test speech")
    parser.add_argument(
        "--output-dir",
        default="test-fixtures/parakeet",
        help="Directory to write the clips to (default: test-fixtures/parakeet)",
    )
    args = parser.parse_args()
    out = pathlib.Path(args.output_dir)
    for name in CLIPS:
        download_one(out, name)


if __name__ == "__main__":
    main()
