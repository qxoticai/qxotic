#!/usr/bin/env python3
"""Score speech-recognition runs: WER against the reference, RTFx, and agreement between engines.

Reads what an engine heard, from either shape:

  *.tsv   TranscriptionBench --dump: id, audio seconds, decode seconds, hypothesis, reference
  *.json  parakeet-cli bench --json: files[] of path, audio_sec, proc_ms/proc_s, text

Normalization matches parakeet.cpp's scripts/asr_metrics.py (NFKC, lowercase, punctuation to
spaces, collapse whitespace), so every engine is scored on equal terms. WER is the word-level edit
distance over the reference length; agreement is the same distance against the first run given,
which is 0 when two engines heard exactly the same words.

  score_asr.py run.tsv [more.tsv ...]              # WER and RTFx
  score_asr.py --against reference.tsv run.json    # plus agreement with that run
  score_asr.py --references manifest.tsv run.json  # references for runs that carry none
"""

import argparse
import json
import pathlib
import re
import sys
import unicodedata

PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text):
    """NFKC, lowercase, punctuation to spaces, collapsed: the words WER counts."""
    folded = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"\s+", " ", PUNCTUATION.sub(" ", folded)).strip().split()


def errors(reference, hypothesis):
    """Word-level edit distance and the reference's word count."""
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for i, want in enumerate(reference, 1):
        current = [i]
        for j, heard in enumerate(hypothesis, 1):
            substitute = previous[j - 1] + (want != heard)
            current.append(min(substitute, previous[j] + 1, current[j - 1] + 1))
        previous = current
    return previous[-1], len(reference)


def read(path, references=None):
    """One run as {id: (audio seconds, decode seconds, heard, reference)}."""
    path = pathlib.Path(path)
    if path.suffix == ".json":
        rows = {}
        for file in json.loads(path.read_text())["files"]:
            name = pathlib.Path(file["path"]).stem
            seconds = file.get("proc_s", file.get("proc_ms", 0) / 1000)
            rows[name] = (file["audio_sec"], seconds, file["text"], (references or {}).get(name, ""))
        return rows
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        name, audio, decode, heard, *rest = line.split("\t")
        reference = rest[0] if rest else (references or {}).get(name, "")
        rows[name] = (float(audio), float(decode), heard, reference)
    return rows


def report(name, rows, against=None):
    wrong = words = agreed_wrong = agreed_words = 0
    audio = decode = 0.0
    for key, (seconds, spent, heard, reference) in rows.items():
        wrong_here, words_here = errors(reference, heard)
        wrong += wrong_here
        words += words_here
        audio += seconds
        decode += spent
        if against and key in against:
            differ, counted = errors(against[key][2], heard)
            agreed_wrong += differ
            agreed_words += counted
    line = f"{name:<34} WER {100 * wrong / max(1, words):5.2f}%   RTFx {audio / max(1e-9, decode):6.1f}"
    if against:
        line += f"   agreement {100 * agreed_wrong / max(1, agreed_words):5.2f}%"
    print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="dumps to score (*.tsv or *.json)")
    parser.add_argument("--against", help="the run to measure agreement with")
    parser.add_argument("--references", help="manifest of <audio path>\\t<reference> per line")
    arguments = parser.parse_args()

    references = None
    if arguments.references:
        references = {}
        for line in pathlib.Path(arguments.references).read_text().splitlines():
            if line.strip():
                audio, reference = line.split("\t", 1)
                references[pathlib.Path(audio).stem] = reference
    against = read(arguments.against, references) if arguments.against else None
    for run in arguments.runs:
        report(pathlib.Path(run).stem, read(run, references), against)
    return 0


if __name__ == "__main__":
    sys.exit(main())
