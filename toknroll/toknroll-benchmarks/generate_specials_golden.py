#!/usr/bin/env python3
"""
Generate the special-token goldens (toknroll-hf/src/test/resources/specials/<org>__<repo>.json):
Hugging Face `tokenizers` ids for text full of each model's added tokens, which
HuggingFaceSpecialsGoldenTest checks against Specials.of(tokenizer).encode, toknroll's
special-aware path.

    python3 -m venv .venv && .venv/bin/pip install tokenizers==0.23.3
    .venv/bin/python toknroll-benchmarks/generate_specials_golden.py [--out DIR] [--model ORG/REPO]

The goldens were made with tokenizers 0.23.3; the script refuses another version.

Hugging Face matches every added token whole (special or not) with encode(text,
add_special_tokens=False): the ids recorded here. Each model is pinned to a commit. The texts are
deterministic: prose, a sample of the added tokens in tricky contexts, every added token at least
once, and for DeepSeek V4.1 the reference prompts its repository ships.
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

TOKENIZERS_VERSION = "0.23.3"

# Every supported tokenizer.json model, pinned. phi-4 is not supported: its added tokens set
# lstrip/rstrip, which toknroll refuses. Nor is Gemma 3: its added tokens spelled as runs of U+2581
# match only as their decoded text, spaces.
MODELS = [
    ("deepseek-ai/DeepSeek-V3.2", "a7e62ac04ecb2c0a54d736dc46601c5606cf10a6"),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "314113f8a7dadb388215396ba5fbee2c610a04d6"),
    ("deepseek-ai/DeepSeek-V4-Pro", "b5968e9190ef611bbf34a7229255be88a0e937c1"),
    ("google/gemma-4-e2b-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"),
    ("HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1"),
    ("ibm-granite/granite-4.0-h-1b", "d18cca4c121edb87d022116d281ce212c9136f57"),
    ("inclusionAI/Ling-3.0-tiny", "9a98e35fe1c9ee255f78dd64771c7ae15a799481"),
    ("inclusionAI/Ling-3.1-flash", "276cca72bbd8b95693032848bfd638460b529b8b"),
    ("LiquidAI/LFM3-1.7B", "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"),
    ("MiniMaxAI/MiniMax-M2.7", "0a6536713b0b03f3be97494e5cf8d22f668a3164"),
    ("mistralai/ministral-8b-instruct-2410", "2f494a194c5b980dfb9772cb92d26cbb671fce5a"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71"),
    ("mistralai/Mistral-Medium-3.5-128B", "22b2b868a15677cfa6061277ed2f653d1349a9ab"),
    ("mistralai/Mistral-Small-24B-Instruct-2501", "9527884be6e5616bdd54de542f9ae13384489724"),
    ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
    ("poolside/Laguna-S-2.1", "0f573140834b11cfac0c2af97a101a7a69a13e22"),
    ("poolside/Laguna-XS-2.1", "c5f36269bbdbd3f27fddc9a9f9dbae0cf2cf57db"),
    ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"),
    ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    ("unsloth/Llama-3.2-1B-Instruct", "5a8abab4a5d6f164389b1079fb721cfab8d7126c"),
    ("unsloth/NVIDIA-Nemotron-3-Nano-4B", "b604b1379531cfac85a3cb195c387cc698aabdeb"),
    ("Xenova/gpt-4", "1d9f1f1b1fae88c0e4df1dab0a397f8de6229075"),
    ("XiaomiMiMo/MiMo-V2-Flash", "8b60b288e8325b7890fcdfe32bbb1d172c591a2d"),
    ("zai-org/GLM-5.1", "26e1bd6e011feb778d25ae34b09b07074139d92d"),
]

# Real prompts, rendered by the repository's own encoder, under encoding/tests/.
REFERENCE_PROMPTS = {
    "deepseek-ai/DeepSeek-V4.1-Flash": [f"encoding/tests/test_output_{i}.txt" for i in range(1, 6)],
}

PROSE = [
    "The quick brown fox jumps over the lazy dog.",
    "def f(x):\n    return x * 2  # double\n",
    "Hello,   world!\n\n\tTabs and  spaces.",
    "Unicode: naïve café, 東京, emoji 🙂, fullwidth ｜ and ▁ marks.",
    "",
]
CONTEXTS = ["{t}", " {t} ", "a{t}b", "{t}\n", "\n{t}", "  {t}  ", "{t}{t}", "word {t}word", "x{t} y", "{t}."]
SEPARATORS = ["", " ", "\n", "x", "hi "]


def texts_for(model, added):
    rng = random.Random(model)
    texts = list(PROSE)
    sample = added if len(added) <= 120 else rng.sample(added, 120)
    for t in sample:
        for c in rng.sample(CONTEXTS, 4):
            texts.append(rng.choice(PROSE[:4])[:20] + c.format(t=t) + rng.choice(PROSE[:4])[-12:])
    for _ in range(20):  # several in one text, as a rendered chat template has them
        texts.append(" ".join(rng.choice(sample) + rng.choice(["hi", " ", "\n", "x"]) for _ in range(5)))
    for i in range(0, len(added), 16):  # every added token at least once
        texts.append("".join(t + rng.choice(SEPARATORS) for t in added[i : i + 16]))
    return texts


def fetch(model, revision, path):
    request = urllib.request.Request(f"https://huggingface.co/{model}/resolve/{revision}/{path}")
    token = os.environ.get("HF_TOKEN") or read_token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def read_token():
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    return path.read_text().strip() if path.is_file() else None


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(root / "toknroll-hf" / "src" / "test" / "resources" / "specials"))
    p.add_argument("--model", action="append", help="only this ORG/REPO (repeatable)")
    args = p.parse_args()

    import tokenizers
    from tokenizers import Tokenizer

    if tokenizers.__version__ != TOKENIZERS_VERSION:
        sys.exit(f"tokenizers {tokenizers.__version__}: the goldens need {TOKENIZERS_VERSION}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for model, revision in MODELS:
        if args.model and model not in args.model:
            continue
        tokenizer_json = fetch(model, revision, "tokenizer.json")
        tokenizer = Tokenizer.from_str(tokenizer_json)
        added = sorted({a["content"] for a in json.loads(tokenizer_json)["added_tokens"]})
        cases = [{"text": t} for t in texts_for(model, added)]
        for path in REFERENCE_PROMPTS.get(model, []):
            cases.append({"text": fetch(model, revision, path), "source": path})
        for case in cases:
            case["ids"] = tokenizer.encode(case["text"], add_special_tokens=False).ids
        golden = {
            "model": model,
            "revision": revision,
            "tokenizers": tokenizers.__version__,
            "cases": cases,
        }
        target = out / (model.replace("/", "__") + ".json")
        target.write_text(json.dumps(golden, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"{model:44} {len(cases):5} cases -> {target}")


if __name__ == "__main__":
    sys.exit(main())
