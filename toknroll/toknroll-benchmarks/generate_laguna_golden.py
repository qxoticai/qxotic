#!/usr/bin/env python3
"""
Generate toknroll-gguf/src/test/resources/laguna_golden_tokens.json: reference token ids for
LagunaGoldenParityTest, produced by Hugging Face `tokenizers` from the model's own tokenizer.json.

    python3 -m venv .venv && .venv/bin/pip install tokenizers
    .venv/bin/python toknroll-benchmarks/generate_laguna_golden.py [--model poolside/Laguna-XS-2.1] [--out PATH]

Encoding uses add_special_tokens=False, the contract of Tokenizer.encode (the non-special-aware
path), so the cases must not contain special-token spellings such as 〈|EOS|〉.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

CASES = {
    "empty": "",
    "hello": "Hello world",
    "leading_space": " Hello world",
    "contractions": "I'm sure they'd say we're fine, isn't it? You'll see, I've got 'em. IT'S DONE, SHE'D KNOW",
    "digits_run": "The year 2026, phone 5551234567, pi 3.14159, minus -42, big 1,000,000.00",
    "alnum_mix": "abc123def456 x2y3 CamelCaseWord snake_case_name kebab-case-name",
    "punct_runs": "wow!!! ?![]{}()<>#$%^&*~`|\\/ ... ---> <<== ::",
    "spaces_run": "a    b        c\t\td   \t e",
    "trailing_spaces": "trailing spaces   ",
    "newline_single": "line one\nline two",
    "newline_double": "para one\n\npara two",
    "newline_runs": "a\n\n\nb\n",
    "newline_leading": "\n\nstarts with newlines",
    "newline_trailing": "ends with newlines\n\n",
    "crlf": "windows\r\nline\r\n\r\nendings\r\n",
    "newline_then_spaces": "def f():\n    return 1\n\n\nclass A:\n\tpass\n",
    "code_java": 'public static void main(String[] args) {\n    System.out.println("Hello, " + args[0]);\n}\n',
    "code_python": "import os\n\nfor i in range(10):\n    print(f\"{i:03d}\", end=' ')\n",
    "code_c": '#include <stdio.h>\nint main(void) { return printf("%d\\n", 0x1F & 0b1010); }\n',
    "json": '{"key": [1, 2.5, -3e10, true, null], "nested": {"a": "b"}}',
    "url": "See https://qxotic.ai/jinfer?x=1&y=2#anchor and mailto:dev@example.com",
    "markdown": "# Title\n\n- item **bold** _it_ `code`\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
    "unicode_mixed": "Whitespace\n\tand unicode 😀 café naïve résumé Ærø",
    "emoji_seq": "👨‍👩‍👧‍👦 🇩🇪🇪🇸 ✨🚀🔥 ☕️",
    "cjk": "日本語のテキストと中文混合，还有한국어。",
    "thai": "ไทยภาษาไทย without spaces",
    "arabic": "العربية mixed English 123",
    "cyrillic_greek": "Привет мир, Γειά σου κόσμε",
    "tamil_combining": "க்க ம் க் க மதியிறுக்கம் அரிஸ்டாட்டில்",
    "fullwidth": "ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３",
    "zero_width": "zero\u200bwidth\u200djoiner\ufeffbom",
    "control_chars": "tab\there\x0bvt\x0cff and nul\x00end",
    "long_word": "a" * 300,
    "long_digits": "9" * 40,
    "long_spaces": " " * 64 + "x",
    "prose": "In the beginning the Universe was created. This has made a lot of people very angry and been widely regarded as a bad move.",
    "angle_brackets_plain": "〈not|special〉 ‹›«»",
}


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="poolside/Laguna-XS-2.1", help="Hugging Face repo holding tokenizer.json")
    p.add_argument("--revision", default="main")
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "toknroll-gguf/src/test/resources/laguna_golden_tokens.json"),
    )
    args = p.parse_args()

    import tokenizers
    from tokenizers import Tokenizer

    base = f"https://huggingface.co/{args.model}"
    sha = json.loads(fetch(f"https://huggingface.co/api/models/{args.model}/revision/{args.revision}"))["sha"]
    tok_json = fetch(f"{base}/resolve/{sha}/tokenizer.json")
    tok = Tokenizer.from_str(tok_json.decode("utf-8"))

    cases = []
    for case_id, text in CASES.items():
        ids = tok.encode(text, add_special_tokens=False).ids
        decoded = tok.decode(ids, skip_special_tokens=False)
        if decoded != text:
            sys.exit(f"{case_id}: reference decode is not the identity; goldens assume a lossless byte-level decoder")
        cases.append({"id": case_id, "text": text, "tokens": ids, "decoded": decoded})

    out = {
        "model_ref": args.model,
        "revision": sha,
        "tokenizer_file": "tokenizer.json",
        "generator": "tokenizers " + tokenizers.__version__,
        "encode": "add_special_tokens=False",
        "cases": cases,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(cases)} cases, {sum(len(c['tokens']) for c in cases)} tokens -> {args.out}")


if __name__ == "__main__":
    main()
