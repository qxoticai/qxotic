#!/usr/bin/env python3
"""
Generate the tokenizer goldens: ground_truth_tokens.json (tiktoken's r50k_base, cl100k_base and
o200k_base) and toknroll-core/src/test/resources/ground_truth_model_families.json (each family's
reference tokenizer at its pinned revision), from GOLDEN_TEST_STRINGS.

Run it from toknroll/ with the versions pinned in requirements.txt (the script refuses others):

    python3 -m venv .venv && .venv/bin/pip install -r toknroll-benchmarks/requirements.txt
    .venv/bin/python toknroll-benchmarks/generate_ground_truth.py
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import tiktoken
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tiktoken import load as tiktoken_load
from tokenizers import Tokenizer as RawTokenizer

# The libraries whose answers the goldens record, at the versions requirements.txt pins.
PINNED = ("tiktoken", "tokenizers", "huggingface_hub", "mistral-common")


def package_version(name):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def check_versions():
    pins = {}
    requirements = Path(__file__).resolve().parent / "requirements.txt"
    for line in requirements.read_text(encoding="utf-8").splitlines():
        name, _, version = line.strip().partition("==")
        pins[name] = version
    wrong = [
        f"{name} {package_version(name)} (requirements.txt: {pins[name]})"
        for name in PINNED
        if package_version(name) != pins[name]
    ]
    if wrong:
        sys.exit("the goldens need the pinned versions: " + ", ".join(wrong))


def resolve_hf_revision(model_id, revision):
    return HfApi().model_info(model_id, revision=revision).sha


class HFAdapter:
    """A repository's tokenizer.json, or its tiktoken.model when it has none (Kimi)."""

    def __init__(self, model_id, revision):
        self.model_id = model_id
        self.resolved_revision = resolve_hf_revision(model_id, revision)
        try:
            tokenizer_path = hf_hub_download(
                repo_id=model_id, filename="tokenizer.json", revision=self.resolved_revision
            )
        except EntryNotFoundError:
            self.tokenizer = None
            self.tiktoken_encoding = self._tiktoken_encoding()
        else:
            # toknroll's encode is raw text: the tokenizer.json without its added tokens, so no
            # added token is matched, as with tiktoken's encode_ordinary below
            with open(tokenizer_path, encoding="utf-8") as f:
                raw = json.load(f)
            raw["added_tokens"] = []
            self.tokenizer = RawTokenizer.from_str(json.dumps(raw))
            self.tiktoken_encoding = None

    def _tiktoken_encoding(self):
        tiktoken_path = hf_hub_download(
            repo_id=self.model_id, filename="tiktoken.model", revision=self.resolved_revision
        )
        tokenizer_config = _try_load_hf_json(
            self.model_id, "tokenizer_config.json", self.resolved_revision
        )
        return tiktoken.Encoding(
            name=self.model_id,
            pat_str=_resolve_tiktoken_pat_str(
                self.model_id, tokenizer_config, self.resolved_revision
            ),
            mergeable_ranks=tiktoken_load.load_tiktoken_bpe(tiktoken_path),
            special_tokens=_extract_special_tokens(tokenizer_config),
        )

    def encode(self, text):
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=False).ids
        return self.tiktoken_encoding.encode_ordinary(text)

    def decode(self, tokens):
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens, skip_special_tokens=False)
        return self.tiktoken_encoding.decode(tokens)


def _try_load_hf_json(model_id, filename, revision):
    try:
        path = hf_hub_download(repo_id=model_id, filename=filename, revision=revision)
    except EntryNotFoundError:
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
    def __init__(self, repo_id, revision):
        self.model_id = repo_id
        self.resolved_revision = resolve_hf_revision(repo_id, revision)
        mtok = MistralTokenizer.from_hf_hub(repo_id=repo_id, revision=self.resolved_revision)
        self._base = mtok.instruct_tokenizer.tokenizer

    def encode(self, text):
        return self._base.encode(text, bos=False, eos=False)

    def decode(self, tokens):
        return self._base.decode(tokens)


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
    # Keep fixtures in plain-text mode: avoid strings that are commonly
    # intercepted as special/control tokens across tokenizer backends.
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
    "delimiters: <<SYS>> assistant INSTx/INST {{tool_call}}",
]


def generate_test_cases(encoding_name, max_cases=100):
    """Generate test cases for an encoding."""
    enc = tiktoken.get_encoding(encoding_name)
    test_strings = GOLDEN_TEST_STRINGS

    cases = {}

    for text in test_strings[:max_cases]:
        safe_text = text.replace("\x00", "\ufffd")  # JSON-safe
        # Raw text: special-token spellings are text
        tokens = enc.encode_ordinary(safe_text)
        # text and tokens are the whole truth: tiktoken round-trips, so the decoded text is the
        # text, its bytes are its UTF-8, the count is len(tokens)
        assert enc.decode(tokens) == safe_text, (encoding_name, safe_text)
        cases[f"case_{len(cases):03d}"] = {"text": safe_text, "tokens": tokens}
    return cases


def generate_cases_with_adapter(adapter):
    cases = {}
    for text in GOLDEN_TEST_STRINGS:
        safe_text = text.replace("\x00", "\ufffd")  # JSON-safe
        tokens = adapter.encode(safe_text)
        case = {"text": safe_text, "tokens": tokens}
        decoded = adapter.decode(tokens)
        if decoded != safe_text:  # e.g. a SentencePiece decoder drops the leading space
            case["decoded"] = decoded
        cases[f"case_{len(cases):03d}"] = case
    return cases


# Each family's reference tokenizer: (family id, backend, repository, pinned revision).
FAMILIES = [
    ("google.gemma4", "hf", "google/gemma-4-e2b-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"),
    ("nvidia.nemotron3_nano4b", "hf", "unsloth/NVIDIA-Nemotron-3-Nano-4B", "b604b1379531cfac85a3cb195c387cc698aabdeb"),
    ("alibaba.qwen3_5", "hf", "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    ("meta.llama3", "hf", "unsloth/Llama-3.2-1B-Instruct", "5a8abab4a5d6f164389b1079fb721cfab8d7126c"),
    ("moonshot.kimi2_5", "hf", "moonshotai/Kimi-K2.6", "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"),
    ("ibm.granite4_0", "hf", "ibm-granite/granite-4.0-h-1b", "d18cca4c121edb87d022116d281ce212c9136f57"),
    ("huggingface.smollm3", "hf", "HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1"),
    ("mistral.gpt2_pretekken", "hf", "mistralai/Mistral-Small-24B-Instruct-2501", "9527884be6e5616bdd54de542f9ae13384489724"),
    ("mistral.v0_3", "hf", "mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71"),
    ("deepseek.v3_2", "hf", "deepseek-ai/DeepSeek-V3.2", "a7e62ac04ecb2c0a54d736dc46601c5606cf10a6"),
    ("deepseek.v4_pro", "hf", "deepseek-ai/DeepSeek-V4-Pro", "b5968e9190ef611bbf34a7229255be88a0e937c1"),
    ("mistral.tekken", "mistral-common", "mistralai/ministral-8b-instruct-2410", "2f494a194c5b980dfb9772cb92d26cbb671fce5a"),
    ("zai.glm5_1", "hf", "zai-org/GLM-5.1", "26e1bd6e011feb778d25ae34b09b07074139d92d"),
    ("minimax.m2_7", "hf", "MiniMaxAI/MiniMax-M2.7", "0a6536713b0b03f3be97494e5cf8d22f668a3164"),
    ("xiaomi.mimo_v2_flash", "hf", "XiaomiMiMo/MiMo-V2-Flash", "8b60b288e8325b7890fcdfe32bbb1d172c591a2d"),
    ("moonshot.kimi3", "hf", "moonshotai/Kimi-K3", "f831ab66814297da540d832a5235f8e904f29d06"),
    ("deepseek.v4_1", "hf", "deepseek-ai/DeepSeek-V4.1-Flash", "314113f8a7dadb388215396ba5fbee2c610a04d6"),
    ("alibaba.qwen3_6", "hf", "Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"),
    ("poolside.laguna_xs2_1", "hf", "poolside/Laguna-XS-2.1", "c5f36269bbdbd3f27fddc9a9f9dbae0cf2cf57db"),
    ("poolside.laguna_s2_1", "hf", "poolside/Laguna-S-2.1", "0f573140834b11cfac0c2af97a101a7a69a13e22"),
    ("inclusion.ling3_0", "hf", "inclusionAI/Ling-3.0-tiny", "9a98e35fe1c9ee255f78dd64771c7ae15a799481"),
    ("inclusion.ling3_1", "hf", "inclusionAI/Ling-3.1-flash", "276cca72bbd8b95693032848bfd638460b529b8b"),
    ("liquid.lfm3", "hf", "LiquidAI/LFM3-1.7B", "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"),
    ("openai.gpt-oss", "hf", "openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
]


def generate_model_family_ground_truth():
    result = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "toknroll-benchmarks/generate_ground_truth.py",
            "libraries": {
                "tiktoken": package_version("tiktoken"),
                "transformers": package_version("transformers"),
                "tokenizers": package_version("tokenizers"),
                "mistral-common": package_version("mistral-common"),
            },
        },
        "families": {},
    }
    adapters = {"hf": HFAdapter, "mistral-common": MistralTekkenAdapter}
    for family_id, backend, model_id, revision in FAMILIES:
        print(f"Generating family ground truth for {family_id} ({backend})...")
        adapter = adapters[backend](model_id, revision)
        cases = generate_cases_with_adapter(adapter)
        result["families"][family_id] = {
            "backend": backend,
            "model_ref": adapter.model_id,
            "revision": adapter.resolved_revision,
            "cases": cases,
        }
        print(f"  generated {len(cases)} cases")
    return result


def main():
    """Generate ground truth file."""
    parser = argparse.ArgumentParser(description="Generate tokenizer ground truth fixtures")
    parser.add_argument(
        "--skip-model-families",
        action="store_true",
        help="Only generate ground_truth_tokens.json and skip model family fixture generation",
    )
    args = parser.parse_args()
    check_versions()

    print("Generating ground_truth_tokens.json...")
    ground_truth = {}
    for enc_name in ("r50k_base", "cl100k_base", "o200k_base"):  # GPT-2, GPT-4, GPT-4o
        ground_truth[enc_name] = generate_test_cases(enc_name, len(GOLDEN_TEST_STRINGS))
        print(f"  {enc_name}: {len(ground_truth[enc_name])} cases")

    # Write output: compact, where `make test-fixtures` downloads it. The published copy lives in
    # the qxoticai/assets repository as toknroll/ground_truth_tokens.json - push it there after
    # regenerating, so every clone fetches the same truth.
    output_path = Path("../test-fixtures/tiktoken/ground_truth_tokens.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, separators=(",", ":"), ensure_ascii=False)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    if args.skip_model_families:
        print("Skipping model family fixture generation (--skip-model-families).")
        return

    print("Generating model family ground truth...")
    family_truth = generate_model_family_ground_truth()
    family_output = Path("toknroll-core/src/test/resources/ground_truth_model_families.json")
    family_output.parent.mkdir(parents=True, exist_ok=True)
    with open(family_output, "w", encoding="utf-8") as f:
        json.dump(family_truth, f, separators=(",", ":"), ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {family_output}")


if __name__ == "__main__":
    main()
