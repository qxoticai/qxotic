#!/usr/bin/env python3
"""Download the media the jinfer test suites read (pictures, video frames, audio clips) from the
qxoticai/assets repository, pinned to a commit and checked against their SHA-256: parity and
decision tests compare against numbers measured on these exact bytes.

Writes files to test-fixtures by default, at the paths the tests name (TestModels.fixture).
"""
import argparse
import pathlib

from download_parakeet_fixtures import download_one

ASSETS = "https://raw.githubusercontent.com/qxoticai/assets/65c4fd89947d2061a4716f9af366d96f517190f6/jinfer"

# path under test-fixtures -> sha256
MEDIA = {
    "lfm3/images/cat.png": "83aff88e5e663b65e51d869daf87037ad4949dd987f7da861f908cb59f547371",
    "lfm3/images/dice.png": "c4e870d358d64ce677e958c156d0e3fba466993d8e915fc81c5dc64937a9b696",
    "lfm3/images/small.png": "7674a0564fe8823b0ae1b12e4a6ec0ea7289295c84797b56b16843133a4982d8",
    "lfm3/images/wide.png": "5b1f7ce044bc0487e5259fd23e20d2a428d22648664e16093e3094db11b7746a",
    "lfm3/video/clip-00.png": "49363466408b9584defd2ac4fd86fa33849560610968bd23c4eb3ea79c7f4feb",
    "lfm3/video/clip-02.png": "b155a3dce98dbdbb5597b3504435bb037a86c7542d451b43d1e0d3449007ed14",
    "lfm3/video/clip-04.png": "fd9acde72f415876f99ceef0827fbf4188f03930497e832fe1f81468982df4a4",
    "lfm3/video/clip-07.png": "c305b109f08db1a6cf8557116df2c4c9a05768ba91e808699cd1a2cdea937ffd",
    "lfm3/video/clip-09.png": "5ad17f44e5b132aefc17d38ff2cc0c11142df14ffa2aa085aa274bc1ce2f7167",
    "lfm3/video/clip-11.png": "5457a828e0396d72560eee5f52016b44544f942562f86e6753a3c005ecb94231",
    "audio/speech-tts.wav": "66f50258f3673f4ed0801af0375f8c2ed30041081641a483bc05568ddb47c70b",
    "audio/chirp-5s.wav": "17e91d196063152b35cf76fda0db32d6498fe25570f266c20ed924e7a8fde8c2",
    "audio/sine440-3s.wav": "3b4c0e2b37a4a676cd88a1c867cfa6f1b02f8a5a7ce1dcd4e7265e7f54b337d9",
    "audio/odd-1p37s.wav": "623ba7acd3f59b6db71d038199d157a8035093a5841c018a195d71a24a0852d0",
    "audio/long-35s.wav": "9d26eb2eccc57ffdec616856c452b60ad980ef957751b2057cfc0f87e6a23ffb",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the jinfer test media")
    parser.add_argument(
        "--output-dir",
        default="test-fixtures",
        help="Directory to write the media to (default: test-fixtures)",
    )
    args = parser.parse_args()
    out = pathlib.Path(args.output_dir)
    for name, expected in MEDIA.items():
        download_one(out, name, f"{ASSETS}/{name}", expected)


if __name__ == "__main__":
    main()
