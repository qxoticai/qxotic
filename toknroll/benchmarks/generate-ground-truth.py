#!/usr/bin/env python3
"""
Generate ground_truth_tokens.json for tokenizer golden tests.
This creates a subset of test cases for each tiktoken encoding.
"""

import json
import base64
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

# Add parent directory to path to import tiktoken
try:
    import tiktoken
except ImportError:
    print("Error: tiktoken not installed. Run: pip install tiktoken")
    print("Or use the system Python with tiktoken installed.")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
except ImportError:
    MistralTokenizer = None

try:
    from gemma import gm
except ImportError:
    gm = None

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


def get_encoding(name):
    """Get a tiktoken encoding by name."""
    try:
        return tiktoken.get_encoding(name)
    except Exception as e:
        print(f"Error loading encoding {name}: {e}")
        return None


def package_version(name):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def resolve_hf_revision(model_id, revision):
    if HfApi is None:
        return revision
    try:
        info = HfApi().model_info(model_id, revision=revision)
        return info.sha or revision
    except Exception:
        return revision


class HFAdapter:
    def __init__(self, model_candidates, revision="main", split_special_tokens=False):
        self.model_candidates = model_candidates
        self.revision = revision
        self.split_special_tokens = split_special_tokens
        self.tokenizer = None
        self.model_id = None
        self.resolved_revision = None

    def available(self):
        if AutoTokenizer is None:
            return False
        for model_id in self.model_candidates:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id, revision=self.revision, trust_remote_code=True
                )
                self.model_id = model_id
                self.resolved_revision = resolve_hf_revision(model_id, self.revision)
                return True
            except Exception:
                continue
        return False

    def encode(self, text):
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
            split_special_tokens=self.split_special_tokens,
        )

    def decode(self, tokens):
        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def decode_bytes(self, tokens):
        # HF tokenizers may not expose canonical token-by-token byte decode consistently.
        return None


class MistralTekkenAdapter:
    def __init__(self, repo_candidates, revision="main"):
        self.repo_candidates = repo_candidates
        self.revision = revision
        self.repo_id = None
        self.resolved_revision = None
        self._base = None

    def available(self):
        if MistralTokenizer is None:
            return False
        for repo_id in self.repo_candidates:
            try:
                mtok = MistralTokenizer.from_hf_hub(
                    repo_id=repo_id, revision=self.revision
                )
                self._base = mtok.instruct_tokenizer.tokenizer
                self.repo_id = repo_id
                self.resolved_revision = resolve_hf_revision(repo_id, self.revision)
                return True
            except Exception:
                continue
        return False

    def encode(self, text):
        return self._base.encode(text, bos=False, eos=False)

    def decode(self, tokens):
        return self._base.decode(tokens)

    def decode_bytes(self, tokens):
        # Tekken decode-bytes API is not uniform across versions in mistral-common.
        return None


class Gemma3Adapter:
    def __init__(self, hf_model_candidates, revision="main"):
        self.hf_model_candidates = hf_model_candidates
        self.revision = revision
        self.model_ref = None
        self.resolved_revision = None
        self._gemma_tokenizer = None
        self._hf = None

    def available(self):
        # Prefer the official Gemma tokenizer package when available.
        if gm is not None:
            try:
                self._gemma_tokenizer = gm.text.Gemma3Tokenizer()
                self.model_ref = "gemma:gm.text.Gemma3Tokenizer"
                self.resolved_revision = package_version("gemma")
                return True
            except Exception:
                pass

        # Fallback to HF-compatible Gemma tokenizer repos.
        self._hf = HFAdapter(self.hf_model_candidates, revision=self.revision)
        if self._hf.available():
            self.model_ref = self._hf.model_id
            self.resolved_revision = self._hf.resolved_revision
            return True
        return False

    def encode(self, text):
        if self._gemma_tokenizer is not None:
            return self._gemma_tokenizer.encode(text)
        return self._hf.encode(text)

    def decode(self, tokens):
        if self._gemma_tokenizer is not None:
            return self._gemma_tokenizer.decode(tokens)
        return self._hf.decode(tokens)

    def decode_bytes(self, tokens):
        return None


# Single source of truth for both tiktoken and model-family fixtures.
GOLDEN_TEST_STRINGS = [
    "",
    "Hello",
    "Hello, World!",
    "Hello World",
    "Hello   World",
    "Hello\n\nWorld",
    "The quick brown fox jumps over the lazy dog",
    "'Hello' \"World\"",
    "Hello (World) [Test] {Data}",
    "12345",
    "3.14159 -42 +7e10 1,000,000",
    "2024-01-15",
    "$1,234.56",
    "0xDEADBEEF",
    "www.example.com",
    "user@example.com",
    "<|endoftext|>",
    "<|fim_prefix|>Hello<|fim_suffix|>World<|fim_middle|>",
    "[THINK]T1[/THINK]",
    '[TOOL_CALLS]{"name":"F1","arguments":{}}',
    '[TOOL_RESULTS]{"content":"R1","call_id":"123"}',
    "   ",
    "\t\t\t",
    "\n\n\n",
    "Hello\r\nWorld",
    "line1\rline2\r\nline3\nline4",
    "NBSP only: a\u00a0b\u00a0c",
    "NEL only: a\u0085b\u0085c",
    "Ogham space mark: a\u1680b\u1680c",
    "en/em quads: a\u2000b\u2001c",
    "figure space: a\u2007b\u2007c",
    "thin/hair spaces: a\u2009b\u200ac",
    "line separator: a\u2028b",
    "paragraph separator: a\u2029b",
    "narrow no-break space: a\u202fb\u202fc",
    "medium mathematical space: a\u205fb",
    "ideographic space: a\u3000b",
    "mixed unicode spaces: a\u00a0\u2007\u202f\u3000b",
    "CR LF CRLF NEL LS PS: a\rb\nc\r\nd\u0085e\u2028f\u2029g",
    "Hello\u0000World",
    "zero-width controls: a\u200bb\u200cc\u200dd\ufeffe",
    "bidi marks: A\u200eB\u200fC",
    "directional isolates: A\u2066B\u2069 C\u2067D\u2069",
    "word joiner + zwnbsp: a\u2060b\ufeffc",
    "café",
    "naïve",
    "e\u0301 != é",
    "\u00f1 n\u0303",
    "العربية",
    "arabic detailed: مرحبًا بالعالم، كيف الحال؟ الإصدار ٢٫٥ جاهز.",
    "arabic with tatweel: العـــــربية",
    "arabic harakat: العَرَبِيَّةُ",
    "arabic numerals: ٠١٢٣٤٥٦٧٨٩",
    "שלום עולם",
    "hebrew detailed: שלום עולם, מה שלומך? גרסה 2.5 מוכנה.",
    "rtl mix العربية English עברית 123",
    "हैलो वर्ल्ड",
    "क्‍ष",
    "ไทยภาษาไทย without spaces",
    "thai detailed: สวัสดีชาวโลก วันนี้อากาศดีมาก ไปเดินเล่นกันไหม",
    "thai with digits: เวอร์ชัน 2.5 เปิดตัววันที่ 03/04/2026",
    "thai tone marks: ก่า ก้า ก๊า ก๋า",
    "vietnamese: Xin chào thế giới, hôm nay trời đẹp quá!",
    "vietnamese combining: tiếng Việt có dấu",
    "日本語",
    "中文",
    "한국어",
    "Hello世界",
    "🎉🎊",
    "Hello🌍World",
    "👋👨‍👩‍👧‍👦👨‍💻",
    "👩🏽‍💻👨🏻‍🚀🧑🏿‍🔬",
    "🏳️‍🌈🏴‍☠️",
    "🏳️‍⚧️",
    "🇺🇳🇺🇸🇯🇵",
    "1️⃣2️⃣3️⃣#️⃣*️⃣",
    "print('Hello World')",
    "function test() { return 42; }",
    "<html><body>Hello</body></html>",
    '<div class="x">\n  <p>Hi</p>\n</div>',
    "<!doctype html><title>x</title><p>tiny</p>",
    '{"key": "value"}',
    "# Heading",
    "**bold**",
    "`code`",
    "[link](http://example.com)",
    "But ird and ปี   ird   ด",
    "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
    "E=mc²",
    "H₂O",
    "∞ ± × ÷",
    "$100 €200 £300 ¥400",
    "© ® ™",
    "plain ascii words",
    "It's we're THEY'LL I'd",
    "don't I'M WE'RE THEY'LL",
    "digits 0 1 12 123 1234 12345 123456",
    "long digits 123456789012345678901234567890",
    "numeric mix v1beta2 build42 x86_64 arm64",
    "versions: v1 v2.5 v10.12.003 build-20260403",
    "dates and times: 2026-04-03 23:59:59 03/04/2026",
    "unicode digits ١٢٣ ۱۲۳ १२३ １２３",
    "tabs\t\tspaces  end",
    "leading separators ,,,hello",
    "trailing separators world!!!",
    "adjacent separators !!!???...,,;;::",
    "boundary punctuation (start) [token] (end)",
    "path/to/api/v2.5.1?x=1&y=22",
    "slashes and CRLF: a/b\\c\r\nnext",
    'json: {"a":1,"b":[true,false,null]}',
    "code: for(i=0;i<10;i++){sum+=i;} // done",
    "a\u0000b\u0001c\u001fd",
    "emoji 😀 fallback",
    "emoji 👨‍👩‍👧‍👦 and flags 🇺🇳🇯🇵",
    "ZWJ family 👨‍👩‍👧‍👦 + kiss 👩‍❤️‍💋‍👨",
    "skin tones 👍🏻👍🏽👍🏿",
    "keycaps 1️⃣ 2️⃣ #️⃣ *️⃣",
    "variation selectors ✊︎ ✊️",
    "combining e\u0301 vs é, n\u0303 vs ñ",
    "combining without base: \u0301\u0308\u0304",
    "normalization pairs: é e\u0301 Å A\u030a Å",
    "rtl عربي עברית فارسی",
    "rtl punctuation mix: العربية؛ עברית، فارسی؟ yes/no.",
    "cjk 你好 日本語 한글",
    "japanese mixed: 今日は良い天気です。カタカナとひらがな、そして漢字。",
    "japanese numbers: 第3版は2026年4月3日に公開されました。",
    "arabic with nbsp: مرحبًا\u00a0بالعالم",
    "thai with nbsp: สวัสดี\u00a0ชาวโลก",
    "vietnamese with nbsp: Xin\u00a0chào\u00a0thế\u00a0giới",
    "emoji + unicode spaces: 😀\u00a0😀\u202f😀\u3000😀",
    "spaces  \t\n\r\n  end",
]


def generate_test_cases(encoding_name, max_cases=100):
    """Generate test cases for an encoding."""
    enc = get_encoding(encoding_name)
    if not enc:
        return {}
    test_strings = GOLDEN_TEST_STRINGS

    cases = {}

    for i, text in enumerate(test_strings):
        if len(cases) >= max_cases:
            break
        try:
            # Handle null bytes for JSON
            safe_text = text.replace("\x00", "\ufffd")

            # Tokenize as raw text (no special-token auto handling)
            tokens = enc.encode(safe_text, disallowed_special=())

            # Decode
            decoded = enc.decode(tokens)

            # Decode to bytes - flatten to single list of byte values
            decoded_bytes = enc.decode_tokens_bytes(tokens)
            byte_list = []
            for b in decoded_bytes:
                if len(b) == 1:
                    byte_list.append(b[0])
                else:
                    byte_list.extend(b)

            # Create case
            case_id = f"case_{len(cases):03d}"
            cases[case_id] = {
                "text": safe_text,
                "decoded": decoded,
                "tokens": tokens,
                "decoded_bytes": byte_list,
                "token_count": len(tokens),
            }
        except Exception as e:
            print(f"  Warning: Failed to process case {i}: {e}")

    return cases


def generate_cases_with_adapter(adapter, max_cases=100):
    test_strings = GOLDEN_TEST_STRINGS

    cases = {}
    for i, text in enumerate(test_strings):
        if len(cases) >= max_cases:
            break
        try:
            safe_text = text.replace("\x00", "\ufffd")
            tokens = adapter.encode(safe_text)
            decoded = adapter.decode(tokens)
            decoded_bytes = adapter.decode_bytes(tokens)
            case = {
                "text": safe_text,
                "decoded": decoded,
                "tokens": tokens,
                "token_count": len(tokens),
            }
            if decoded_bytes is not None:
                case["decoded_bytes"] = decoded_bytes
            case_id = f"case_{len(cases):03d}"
            cases[case_id] = case
        except Exception as e:
            print(f"  Warning: Failed to process case {i}: {e}")
    return cases


def generate_model_family_ground_truth():
    families = [
        {
            "family_id": "google.gemma3",
            "backend": "gemma3",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "google/gemma-3-4b-it",
                "google/gemma-3-1b-it",
            ],
        },
        {
            "family_id": "alibaba.qwen3_5",
            "backend": "hf",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "Qwen/Qwen3.5-0.8B",
                "Qwen/Qwen3.5-0.6B",
                "Qwen/Qwen3-0.6B",
            ],
        },
        {
            "family_id": "meta.llama3",
            "backend": "hf",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.1-8B-Instruct",
                "unsloth/Llama-3.2-1B-Instruct",
                "NousResearch/Llama-3.2-1B",
            ],
        },
        {
            "family_id": "moonshot.kimi2_5",
            "backend": "hf",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "moonshotai/Kimi-K2.5",
                "moonshotai/Kimi-K2-Instruct-0905",
                "moonshotai/Kimi-K2-Instruct",
            ],
        },
        {
            "family_id": "ibm.granite4_0",
            "backend": "hf",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "ibm-granite/granite-4.0-h-1b",
                "ibm-granite/granite-4.0-1b",
            ],
        },
        {
            "family_id": "huggingface.smollm3",
            "backend": "hf",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "HuggingFaceTB/SmolLM3-3B",
                "HuggingFaceTB/SmolLM3-3B-Base",
            ],
        },
        {
            "family_id": "mistral.gpt2_pretekken",
            "backend": "hf",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "mistralai/Mistral-Small-24B-Instruct-2501",
                "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                "mistralai/Mistral-Nemo-Instruct-2407",
            ],
        },
        {
            "family_id": "mistral.v0_3",
            "backend": "hf",
            "revision": "main",
            "split_special_tokens": True,
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "mistralai/Mistral-7B-Instruct-v0.3",
            ],
        },
        {
            "family_id": "deepseek.v3_2",
            "backend": "hf",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "deepseek-ai/DeepSeek-V3.1",
                "deepseek-ai/DeepSeek-V3",
                "deepseek-ai/DeepSeek-R1-0528",
            ],
        },
        {
            "family_id": "microsoft.phi4",
            "backend": "hf",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": ["microsoft/phi-4", "microsoft/Phi-4-mini-instruct"],
        },
        {
            "family_id": "mistral.tekken",
            "backend": "mistral-common",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "mistralai/ministral-8b-instruct-2410",
                "mistralai/open-mistral-nemo-2407",
            ],
        },
    ]

    result = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "benchmarks/generate-ground-truth.py",
            "libraries": {
                "tiktoken": package_version("tiktoken"),
                "transformers": package_version("transformers"),
                "tokenizers": package_version("tokenizers"),
                "mistral-common": package_version("mistral-common"),
            },
        },
        "families": {},
    }

    for family in families:
        family_id = family["family_id"]
        backend = family["backend"]
        print(f"Generating family ground truth for {family_id} ({backend})...")
        if backend == "hf":
            adapter = HFAdapter(
                family["model_candidates"],
                revision=family["revision"],
                split_special_tokens=family.get("split_special_tokens", False),
            )
        elif backend == "gemma3":
            adapter = Gemma3Adapter(
                family["model_candidates"], revision=family["revision"]
            )
        elif backend == "mistral-common":
            adapter = MistralTekkenAdapter(
                family["model_candidates"], revision=family["revision"]
            )
        else:
            continue

        if not adapter.available():
            print(f"  Warning: could not initialize backend for {family_id}, skipping")
            continue

        cases = generate_cases_with_adapter(adapter, family["max_cases"])
        result["families"][family_id] = {
            "backend": backend,
            "model_ref": getattr(adapter, "model_ref", None)
            or getattr(adapter, "model_id", None)
            or getattr(adapter, "repo_id", None),
            "revision": getattr(adapter, "resolved_revision", None)
            or family["revision"],
            "cases": cases,
        }
        print(f"  ✓ Generated {len(cases)} cases")

    return result


def main():
    """Generate ground truth file."""
    print("Generating ground_truth_tokens.json...")
    print()

    # Encodings to generate
    encodings = {
        "r50k_base": len(GOLDEN_TEST_STRINGS),  # GPT-2
        "cl100k_base": len(GOLDEN_TEST_STRINGS),  # GPT-4, ChatGPT
        "o200k_base": len(GOLDEN_TEST_STRINGS),  # GPT-4o
    }

    ground_truth = {}

    for enc_name, max_cases in encodings.items():
        print(f"Generating {max_cases} test cases for {enc_name}...")
        cases = generate_test_cases(enc_name, max_cases)
        if cases:
            ground_truth[enc_name] = cases
            print(f"  ✓ Generated {len(cases)} cases")
        else:
            print(f"  ✗ Failed to generate cases")

    # Write output
    output_path = Path("src/test/resources/ground_truth_tokens.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    # Print stats
    total_cases = sum(len(cases) for cases in ground_truth.values())
    print()
    print(f"✓ Generated ground truth file with {total_cases} total test cases")
    print(f"  - r50k_base: {len(ground_truth.get('r50k_base', {}))} cases")
    print(f"  - cl100k_base: {len(ground_truth.get('cl100k_base', {}))} cases")
    print(f"  - o200k_base: {len(ground_truth.get('o200k_base', {}))} cases")
    print()
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Also generate modern model-family fixtures (best effort, optional deps).
    print()
    print("Generating model family ground truth...")
    family_truth = generate_model_family_ground_truth()
    family_output = Path("src/test/resources/ground_truth_model_families.json")
    family_output.parent.mkdir(parents=True, exist_ok=True)
    with open(family_output, "w", encoding="utf-8") as f:
        json.dump(family_truth, f, indent=2, ensure_ascii=False)
    print(f"✓ Wrote model family fixture to {family_output}")


if __name__ == "__main__":
    main()
