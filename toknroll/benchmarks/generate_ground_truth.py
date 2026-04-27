#!/usr/bin/env python3
"""
Generate ground_truth_tokens.json for tokenizer golden tests.
This creates a subset of test cases for each tiktoken encoding.
"""

import json
import base64
import re
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

# Add parent directory to path to import tiktoken
try:
    import tiktoken
    from tiktoken import load as tiktoken_load
except ImportError:
    print("Error: tiktoken not installed. Run: pip install tiktoken")
    print("Or use the system Python with tiktoken installed.")
    sys.exit(1)

try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
except ImportError:
    MistralTokenizer = None

try:
    from gemma import gm
except ImportError:
    gm = None

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    HfApi = None
    hf_hub_download = None

try:
    from tokenizers import Tokenizer as RawTokenizer
except ImportError:
    RawTokenizer = None


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
    def __init__(self, model_candidates, revision="main"):
        self.model_candidates = model_candidates
        self.revision = revision
        self.tokenizer = None
        self.tiktoken_encoding = None
        self.model_id = None
        self.resolved_revision = None
        self.failures = []

    def available(self):
        if hf_hub_download is None or RawTokenizer is None:
            self.failures.append(
                "huggingface_hub and tokenizers are required for raw tokenizer.json loading"
            )
            return False
        for model_id in self.model_candidates:
            try:
                tokenizer_path = hf_hub_download(
                    repo_id=model_id,
                    filename="tokenizer.json",
                    revision=self.revision,
                )
                self.tokenizer = RawTokenizer.from_file(tokenizer_path)
                self.model_id = model_id
                self.resolved_revision = resolve_hf_revision(model_id, self.revision)
                return True
            except Exception as exc:
                self.failures.append(f"{model_id}: {exc}")

            try:
                tiktoken_path = hf_hub_download(
                    repo_id=model_id,
                    filename="tiktoken.model",
                    revision=self.revision,
                )
                mergeable_ranks = tiktoken_load.load_tiktoken_bpe(tiktoken_path)
                tokenizer_config = _try_load_hf_json(
                    model_id, "tokenizer_config.json", self.revision
                )
                pat_str = _resolve_tiktoken_pat_str(
                    model_id, tokenizer_config, self.revision
                )
                special_tokens = _extract_special_tokens(tokenizer_config)
                self.tiktoken_encoding = tiktoken.Encoding(
                    name=model_id,
                    pat_str=pat_str,
                    mergeable_ranks=mergeable_ranks,
                    special_tokens=special_tokens,
                )
                self.model_id = model_id
                self.resolved_revision = resolve_hf_revision(model_id, self.revision)
                return True
            except Exception as exc:
                self.failures.append(f"{model_id} (tiktoken.model): {exc}")
        return False

    def encode(self, text):
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=False).ids
        if self.tiktoken_encoding is not None:
            return self.tiktoken_encoding.encode(text, disallowed_special=())
        raise RuntimeError("HFAdapter.encode called before tokenizer initialization")

    def decode(self, tokens):
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens, skip_special_tokens=False)
        if self.tiktoken_encoding is not None:
            return self.tiktoken_encoding.decode(tokens)
        raise RuntimeError("HFAdapter.decode called before tokenizer initialization")

    def decode_bytes(self, tokens):
        # HF tokenizers may not expose canonical token-by-token byte decode consistently.
        return None


def _try_load_hf_json(model_id, filename, revision):
    try:
        path = hf_hub_download(repo_id=model_id, filename=filename, revision=revision)
    except Exception:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_special_tokens(tokenizer_config):
    if not tokenizer_config:
        return {}
    out = {}
    added_tokens_decoder = tokenizer_config.get("added_tokens_decoder")
    if not isinstance(added_tokens_decoder, dict):
        return out
    for id_str, token_spec in added_tokens_decoder.items():
        try:
            token_id = int(id_str)
        except Exception:
            continue
        if isinstance(token_spec, dict):
            content = token_spec.get("content")
            if isinstance(content, str):
                out[content] = token_id
    return out


def _resolve_tiktoken_pat_str(model_id, tokenizer_config, revision):
    if isinstance(tokenizer_config, dict):
        pat_str = tokenizer_config.get("pat_str")
        if isinstance(pat_str, str) and pat_str:
            return pat_str

        auto_map = tokenizer_config.get("auto_map")
        if isinstance(auto_map, dict):
            auto_tokenizer = auto_map.get("AutoTokenizer")
            class_ref = None
            if isinstance(auto_tokenizer, list) and auto_tokenizer:
                class_ref = auto_tokenizer[0]
            elif isinstance(auto_tokenizer, str):
                class_ref = auto_tokenizer

            if isinstance(class_ref, str) and "." in class_ref:
                module_name = class_ref.split(".", 1)[0]
                module_path = hf_hub_download(
                    repo_id=model_id,
                    filename=f"{module_name}.py",
                    revision=revision,
                )
                source = Path(module_path).read_text(encoding="utf-8")
                extracted = _parse_pat_str_from_python_module(source)
                if extracted:
                    return extracted

    raise RuntimeError(
        f"Could not resolve pat_str for tiktoken.model in {model_id}; tokenizer_config.json"
        " must include pat_str or auto_map.AutoTokenizer module with pat_str"
    )


def _parse_pat_str_from_python_module(source):
    start = source.find('pat_str = "|".join([')
    if start < 0:
        return None
    end = source.find("])", start)
    if end < 0:
        return None
    block = source[start:end]
    parts = re.findall(r'r"""(.*?)"""', block, flags=re.DOTALL)
    if not parts:
        return None
    return "|".join(parts)


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
    "separators x100: " + "!?,.;:-_" * 100,
    "nested separators ((([[[{{{<<<>>>}}}]]])))",
    "long pipes and slashes ||||||| ////// \\\\\\ ---- ____",
    "csv separators: a,,,b;;c::d||e\\tf",
    "thai corner: ภาษาไทยไม่มีเว้นวรรคแต่มี,เครื่องหมาย!และ?ตัวเลข12345",
    "thai emoji mix: สวัสดี😀โลก🚀ภาษาไทย✨",
    "vietnamese corner: Trời ơi, tiếng Việt có dấu: ắằẳẵặ ấầẩẫậ",
    "vietnamese html: <p>Tiếng Việt: Cộng hòa xã hội chủ nghĩa Việt Nam</p>",
    "chinese punctuation: 你好，世界！这是一个测试：标点、数字123、符号#@$。",
    "chinese fullwidth: ＡＢＣ１２３，。！？【】（）％＃＠＆",
    "mixed cjk: 中文日本語한국어 together-with-separators---///***",
    "html attrs: <a href=\"https://example.com?q=你好&lang=vi\" data-x='1,2,3'>link</a>",
    "html script style: <script>const x='😀';</script><style>.x{white-space:pre;}</style>",
    "html malformed: <div><span>broken</div></span>",
    'xml-ish: <node key="值">内容<![CDATA[<x>&y;]]></node>',
    "emoji heavy: 😀😃😄😁😆😅😂🤣😊😇🙂🙃😉😌😍🥰😘😗😙😚",
    "emoji zwj chain: 👨‍👩‍👧‍👦👩‍❤️‍💋‍👨👩‍❤️‍👩👨‍❤️‍👨",
    "emoji professions: 🧑‍⚕️🧑‍🍳🧑‍🏫🧑‍🚀🧑‍🔬🧑‍💻",
    "emoji skin tones extended: 👍🏻👍🏼👍🏽👍🏾👍🏿",
    "flags set: 🇨🇳🇻🇳🇹🇭🇯🇵🇰🇷🇮🇳🇫🇷🇩🇪🇧🇷",
    "unicode separators run: a\u2028\u2029\u0085\u202f\u00a0b",
    "zero-width run: x\u200b\u200c\u200d\u2060\ufeffy",
    "bidi stress: abc\u202e123\u202cdef \u2067RTL\u2069",
    "combining stress: a\u0301\u0323\u0308 b\u0338 c\u20dd",
    "math unicode: ∑_{i=1}^{n} i = n(n+1)/2; αβγδεζηθ λ→∞",
    "url corner: https://example.com/a%20b/c?x=%E4%BD%A0%E5%A5%BD&y=1,2,3#片段",
    "email corner: first.last+tag+测试@example-domain.co.uk",
    'json deep: {"a":[1,{"b":[2,{"c":"😀"}]}],"d":null}',
    "markdown table: | 列1 | Col2 | 😀 |\\n|---|---:|:---:|\\n| 值 | 123 | ok |",
    'code mixed: if(a<=b&&c!=d){printf("สวัสดี 你好 😀\\n");}',
    "pathological spaces: start" + (" " * 64) + "end",
    "pathological newlines: a" + ("\n" * 16) + "b",
    "pathological tabs: a" + ("\t" * 16) + "b",
    # Full Unicode White_Space coverage set used by many tokenizers:
    # HT, LF, VT, FF, CR, SPACE, NEL, NBSP, OGHAM SPACE MARK,
    # EN QUAD..HAIR SPACE, LS, PS, NNBSP, MMSP, IDEOGRAPHIC SPACE.
    "all-whitespace-run:"
    + "\u0009\u000a\u000b\u000c\u000d\u0020\u0085\u00a0\u1680"
    + "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    + "\u2028\u2029\u202f\u205f\u3000"
    + ":end",
    "all-whitespace-separated:"
    + "A\u0009B\u000aC\u000bD\u000cE\u000dF\u0020G\u0085H\u00a0I\u1680J"
    + "\u2000K\u2001L\u2002M\u2003N\u2004O\u2005P\u2006Q\u2007R\u2008S\u2009T\u200aU"
    + "\u2028V\u2029W\u202fX\u205fY\u3000Z",
    "nel-only-stress:" + ("A\u0085" * 16) + "B",
    "ls-ps-stress:" + ("L\u2028P\u2029" * 12) + "END",
    "alternating-ws-punct:" + (" \t,\u00a0;\u202f|" * 20) + "END",
    'json-escaped-controls:{"s":"\\u0000\\u0001\\u001f\\u007f\\u0085"}',
    "base64-and-hex: AAECAwQFBgcICQ== deadbeefCAFEBABE00112233",
    "unicode-normalization-long: " + ("e\u0301 " * 20) + "| " + ("é " * 20),
    "bidi-isolates-stress:" + ("\u2066abc\u2069\u2067DEF\u2069" * 12),
    "zwj-zwnj-stress:" + ("क\u094d\u200dष क\u094d\u200cष " * 12),
    "cjk-fullwidth-mix:" + ("ＡＢＣ１２３，。！？「」『』（）［］｛｝" * 8),
    "emoji-vs-text-variation:" + ("✊︎/✊️ ♀/♀️ ♂/♂️ " * 12),
    "pathological-mixed-separators:" + (":::,,,;;;|||///\\\\\n\r\t" * 16) + "fin",
    "long-a-run:" + ("a" * 4096),
    "long-space-run:" + (" " * 4096),
    "long-newline-run:" + ("\n" * 2048),
    "long-emoji-run:" + ("😀" * 1024),
    "alternating-ab:" + ("ab" * 2048),
    "alternating-01:" + ("01" * 2048),
    "alternating-case:" + ("aA" * 2048),
    "alternating-punct:" + ("._" * 2048),
    "boundary-leading-space: hello",
    "boundary-trailing-space:hello ",
    "boundary-leading-newline:\nhello",
    "boundary-trailing-newline:hello\n",
    "boundary-double-wrap:  hello  ",
    "apostrophes: I'm I'M I’m i’m we'd WE'D we’d they'dn't",
    "url-encoded-stress:https://x.example/%F0%9F%98%80?q=%E4%BD%A0%E5%A5%BD&x=%00%01&y=a%2Fb",
    'escaped-literals:\\n \\t \\" \\\\u2028 \\\\ud83d\\\\ude00',
    "confusables: LatinA=A GreekAlpha=Α CyrillicA=А fullwidthA=Ａ",
    "combining-order-a:\u0061\u0301\u0323",
    "combining-order-b:\u0061\u0323\u0301",
    "delimiters: <<SYS>> <|assistant|> [INST]x[/INST] {{tool_call}}",
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
            "family_id": "google.gemma4",
            "backend": "hf",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "google/gemma-4-e2b-it",
                "google/gemma-4-9b-it",
            ],
        },
        {
            "family_id": "nvidia.nemotron3_nano4b",
            "backend": "hf",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "unsloth/NVIDIA-Nemotron-3-Nano-4B",
                "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
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
                "moonshotai/Kimi-K2.6",
                "moonshotai/Kimi-K2.5",
                "moonshotai/Kimi-K2-Instruct-0905",
                "moonshotai/Kimi-K2-Instruct",
                "moonshotai/Kimi-Dev-72B",
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
                "deepseek-ai/DeepSeek-V3.2",
                "deepseek-ai/DeepSeek-V3.1",
                "deepseek-ai/DeepSeek-V3",
                "deepseek-ai/DeepSeek-R1-0528",
            ],
        },
        {
            "family_id": "deepseek.v4_pro",
            "backend": "hf",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "deepseek-ai/DeepSeek-V4-Pro",
                "deepseek-ai/DeepSeek-V3.1",
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
        {
            "family_id": "zai.glm5_1",
            "backend": "hf",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "zai-org/GLM-5.1",
            ],
        },
        {
            "family_id": "minimax.m2_7",
            "backend": "hf",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "MiniMaxAI/MiniMax-M2.7",
            ],
        },
        {
            "family_id": "xiaomi.mimo_v2_flash",
            "backend": "hf",
            "revision": "main",
            "max_cases": len(GOLDEN_TEST_STRINGS),
            "model_candidates": [
                "XiaomiMiMo/MiMo-V2-Flash",
            ],
        },
    ]

    result = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "benchmarks/generate_ground_truth.py",
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
            details = getattr(adapter, "failures", None)
            detail_msg = ""
            if details:
                detail_msg = "\n    " + "\n    ".join(details)
            raise RuntimeError(
                f"Failed to initialize backend for {family_id} ({backend}){detail_msg}"
            )

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
    output_path = Path("toknroll-core/src/test/resources/ground_truth_tokens.json")
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
    family_output = Path(
        "toknroll-core/src/test/resources/ground_truth_model_families.json"
    )
    family_output.parent.mkdir(parents=True, exist_ok=True)
    with open(family_output, "w", encoding="utf-8") as f:
        json.dump(family_truth, f, indent=2, ensure_ascii=False)
    print(f"✓ Wrote model family fixture to {family_output}")


if __name__ == "__main__":
    main()
