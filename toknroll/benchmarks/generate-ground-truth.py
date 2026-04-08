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


class TiktokenAdapter:
    def __init__(self, encoding_name):
        self.encoding_name = encoding_name
        self.enc = get_encoding(encoding_name)

    def available(self):
        return self.enc is not None

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, tokens):
        return self.enc.decode(tokens)

    def decode_bytes(self, tokens):
        decoded_bytes = self.enc.decode_tokens_bytes(tokens)
        flat = []
        for b in decoded_bytes:
            if len(b) == 1:
                flat.append(b[0])
            else:
                flat.extend(b)
        return flat


class HFAdapter:
    def __init__(self, model_candidates, revision="main"):
        self.model_candidates = model_candidates
        self.revision = revision
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
        return self.tokenizer.encode(text, add_special_tokens=False)

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


def generate_test_cases(encoding_name, max_cases=100):
    """Generate test cases for an encoding."""
    enc = get_encoding(encoding_name)
    if not enc:
        return {}

    # Test strings covering various scenarios
    test_strings = [
        # Basic ASCII and whitespace
        "",  # Empty string
        "Hello",  # Simple ASCII
        "Hello, World!",  # With punctuation
        "Hello World",  # With space
        "   ",  # Only spaces
        "Hello\nWorld",  # With newline
        "Hello\tWorld",  # With tab
        "12345",  # Numbers
        "Hello123",  # Mixed alphanumeric
        "!@#$%",  # Special chars
        "Hello   World",  # Multiple spaces
        "Hello\n\nWorld",  # Multiple newlines
        "The quick brown fox jumps over the lazy dog",  # Classic pangram
        "This is a longer test string with multiple words to test tokenization",  # Long string
        "<|endoftext|>",  # Special token
        # International text
        "日本語",  # Japanese
        "中文",  # Chinese
        "العربية",  # Arabic
        "🎉🎊",  # Emojis
        "café",  # Accented characters
        "naïve",  # Special characters
        "Hello\u0000World",  # Null byte (will be replaced in JSON)
        "a" * 100,  # Long repeated character
        "ab" * 50,  # Long repeated pattern
        "<|fim_prefix|>Hello<|fim_suffix|>World<|fim_middle|>",  # FIM tokens (if applicable)
        # Additional comprehensive test strings
        "Hello  World",  # Double space
        "Hello\r\nWorld",  # Windows line ending
        " Hello ",  # Leading/trailing spaces
        "\t\t\t",  # Only tabs
        "\n\n\n",  # Only newlines
        "A",  # Single character
        "AB",  # Two characters
        "ABC",  # Three characters
        "hello world",  # Lowercase
        "HELLO WORLD",  # Uppercase
        "Hello World!",  # With exclamation
        "Hello, World?",  # With question mark
        "Hello; World.",  # With semicolon and period
        "'Hello' \"World\"",  # With quotes
        "Hello (World) [Test] {Data}",  # With brackets
        "Hello+World=Test",  # With operators
        "Hello/World\\Test",  # With slashes
        "Hello: World;",  # With colon
        "www.example.com",  # URL-like
        "user@example.com",  # Email-like
        "192.168.1.1",  # IP-like
        "2024-01-15",  # Date-like
        "12:34:56",  # Time-like
        "$1,234.56",  # Currency
        "50%",  # Percentage
        "Hello_World",  # Underscores
        "Hello-World",  # Hyphens
        "Hello--World",  # Double hyphens
        "__init__",  # Python dunder
        "test_function_name",  # Snake case
        "TestClassName",  # Camel case
        "CONSTANT_NAME",  # Constant case
        # Numbers and special numeric formats
        "0",  # Single digit
        "00",  # Leading zeros
        "007",  # James Bond
        "3.14159",  # Pi
        "-42",  # Negative
        "+42",  # Positive
        "1e10",  # Scientific notation
        "0xDEADBEEF",  # Hexadecimal
        "0b101010",  # Binary
        "0o755",  # Octal
        "123-456-7890",  # Phone number-like
        "(123) 456-7890",  # Formatted phone
        "1/2/2024",  # Date format
        "12:00 AM",  # Time format
        # Control and special characters
        "Hello\x01World",  # SOH control char
        "Hello\x1fWorld",  # US control char
        "Hello\x7fWorld",  # DEL char
        "\x00\x01\x02",  # Null and control chars
        "\u00a0",  # Non-breaking space
        " \u00850",  # NEL control character edge
        "\u200b",  # Zero width space
        "\ufeff",  # BOM (Byte Order Mark)
        "Hello\u3000World",  # Ideographic space
        "\ufffd",  # Replacement char
        # Extended Unicode and mixed scripts
        "Héllo Wörld",  # Mixed European accents
        "Привет мир",  # Russian
        "Γειά σου Κόσμε",  # Greek
        "שלום עולם",  # Hebrew (RTL)
        "हैलो वर्ल्ड",  # Hindi
        "வணக்கம்",  # Tamil
        "한국어",  # Korean
        "ไทย",  # Thai
        "Tiếng Việt",  # Vietnamese
        "עבריתالعربية",  # Mixed RTL scripts
        "Hello世界",  # Mixed English-Chinese
        "Hello🌍World",  # Earth emoji
        "👋👨‍👩‍👧‍👦👨‍💻",  # Complex emojis
        "🏳️‍🌈🏴‍☠️",  # Flag emojis
        "\U0001f600\U0001f601\U0001f602",  # Emojis by codepoint
        "👩🏽‍💻👨🏻‍🚀🧑🏿‍🔬",  # Emoji ZWJ + skin tones
        "1️⃣2️⃣3️⃣#️⃣*️⃣",  # Keycap emojis
        "❤️🧡💛💚💙💜🖤🤍🤎",  # Variation selector heart emojis
        "🏳️‍⚧️",  # Trans flag sequence
        "🇺🇳🇺🇸🇯🇵",  # Regional indicator flags
        "🫠🫡🪿",  # Newer emoji code points
        # Zalgo and special Unicode
        "H̷̛̪e̷̛̪l̷̛̪l̷̛̪ơ̷̪",  # Zalgo text (combining marks)
        "ℌ𝔢𝔩𝔩𝔬",  # Mathematical fraktur
        "Ｈｅｌｌｏ",  # Fullwidth ASCII
        "🄷🄴🄻🄻🄾",  # Enclosed alphanumerics
        "ⓗⓔⓛⓛⓞ",  # Circled letters
        "🅷🅴🅻🅻🅾",  # Negative circled
        "e\u0301 != é",  # Combining mark vs precomposed
        "\u00f1 n\u0303",  # Alternate composition forms
        "क्‍ष",  # Devanagari with ZWJ
        "\u200fabc\u200edef",  # RTL/LTR marks in text
        "\u2066left-to-right isolate\u2069",  # Bidi isolate markers
        "\u2028line-separator\u2029paragraph",  # Unicode line separators
        # Code and markup
        "print('Hello World')",  # Python code
        "function test() { return 42; }",  # JavaScript code
        "<html><body>Hello</body></html>",  # HTML
        "<?xml version='1.0'?>",  # XML
        "SELECT * FROM table WHERE id = 1",  # SQL
        "int main() { return 0; }",  # C code
        "# This is a comment",  # Python comment
        "// This is a comment",  # JS comment
        "/* Block comment */",  # Block comment
        "`SELECT * FROM users`",  # Backtick string
        "Hello ${name}",  # Template literal
        "${variable}",  # Variable interpolation
        "{{template}}",  # Jinja/template syntax
        "{% if true %}",  # Django template
        # JSON and data formats
        '{"key": "value"}',  # Simple JSON
        '{"a": 1, "b": 2}',  # JSON with numbers
        "[1, 2, 3, 4, 5]",  # JSON array
        '{"nested": {"key": "value"}}',  # Nested JSON
        "true false null",  # JSON literals
        # Markdown and formatting
        "# Heading",  # Markdown heading
        "## Subheading",  # Markdown subheading
        "**bold**",  # Bold markdown
        "*italic*",  # Italic markdown
        "`code`",  # Inline code
        "```code block```",  # Code block
        "[link](http://example.com)",  # Markdown link
        "> quote",  # Blockquote
        "- list item",  # List item
        "1. ordered item",  # Ordered list
        "---",  # Horizontal rule
        # Mixed and edge cases
        "Hello" + "\t" * 10 + "World",  # Many tabs
        "Hello\n\n\n\nWorld",  # Multiple newlines
        "today\n ",  # tiktoken regex whitespace edge
        "today\n \n",  # tiktoken newline-space-newline edge
        "today\n  \n",  # tiktoken double-space newline edge
        " ",  # Single space
        "  ",  # Double space
        "   ",  # Triple space
        "\n",  # Single newline
        "\r",  # Single carriage return
        "\r\n",  # CRLF
        "a b c d e f g h i j",  # Many single chars
        "word " * 20,  # Repeated word
        "sentence one. sentence two. sentence three.",  # Multiple sentences
        "UPPER lower MiXeD",  # Mixed case
        # Special tokens and markers
        "[PAD] [UNK] [CLS] [SEP] [MASK]",  # BERT special tokens
        "<s> </s> <pad> <unk>",  # T5 special tokens
        "<|user|>Hello<|assistant|>Hi there<|end|>",  # Chat format
        "<|system|>You are helpful<|end|>",  # System prompt
        "<|im_start|>user<|im_end|>",  # IM tokens
        "[INST] Hello [/INST]",  # Instruction tokens
        "<<SYS>> System <</SYS>>",  # System tokens
        "<|startoftext|><|endoftext|>",  # Text markers
        "<|endoftext|> hello <|fim_prefix|>",  # tiktoken special token edge
        "<|endoftext|> hello <|fim_prefix|> there <|fim_middle|>",  # mixed specials
        "[THINK]T1[/THINK]",  # reasoning delimiters
        '[TOOL_CALLS]{"name":"F1","arguments":{}}',  # tool-call marker JSON
        '[TOOL_RESULTS]{"content":"R1","call_id":"123"}',  # tool-result marker JSON
        "But ird and ปี   ird   ด",  # llama/qwen integration spacing + Thai edge
        # Long and repetitive patterns
        "A" * 1000,  # Very long single char
        "Hello World " * 100,  # Repeated phrase
        "abc" * 100,  # Repeated pattern
        "0123456789" * 20,  # Repeated digits
        "word1 word2 word3 word4 word5 " * 50,  # Many different words
        # Boundary and edge cases
        "Hello\x00",  # Ends with null
        "\x00Hello",  # Starts with null
        "\x00",  # Only null
        "Hello\xff",  # Ends with high byte
        "\xffHello",  # Starts with high byte
        "\xff",  # Only high byte
        # Complex mixed content
        "Check out https://example.com/path?query=value&other=123",  # URL with params
        "Email me at user.name+tag@example.co.uk",  # Complex email
        "Price: $1,299.99 (was $1,599.99)",  # Price with parentheses
        "Ref: #123 @user mention",  # Social media refs
        "C++ Java C# F# A++",  # Programming languages
        "AT&T 3M 7-Eleven",  # Company names with symbols
        "It's " + '"working"' + " they're",  # Mixed quotes
        "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",  # Multiple lines
        "Tab\tSeparated\tValues\tHere",  # TSV-like
        "Column 1    Column 2    Column 3",  # Aligned columns
        # Unicode categories test
        "ⅧⅨⅩ",  # Roman numerals
        "①②③",  # Circled numbers
        "⑴⑵⑶",  # Parenthesized numbers
        "⒈⒉⒊",  # Digit full stops
        "«Hello»",  # Guillemets
        '"Hello"',  # German quotes
        "'Hello'",  # Single quotes
        '"Hello"',  # Double quotes
        "`Hello'",  # Mixed quotes
        # Mathematical and scientific
        "E=mc²",  # Einstein
        "H₂O",  # Water formula
        "∑∏∫∂",  # Math symbols
        "αβγδε",  # Greek letters
        "∞ ± × ÷",  # Math operators
        "≤ ≥ ≠ ≈",  # Comparisons
        "→ ← ↑ ↓",  # Arrows
        "✓ ✗ ✔ ✘",  # Check marks
        "★ ☆ ✡ ✦",  # Stars
        "♠ ♥ ♦ ♣",  # Card suits
        # Currency and commercial
        "$100 €200 £300 ¥400",  # Major currencies
        "₹500 ₹",  # Rupee
        "₽600",  # Ruble
        "₩700",  # Won
        "₺800",  # Lira
        "© ® ™ ℠",  # Intellectual property
        "§ ¶ † ‡",  # Legal/footnote
        "№ №.",  # Numero
        "°C °F",  # Temperature
        "㎏ ㎎ ㎖ ㎗",  # Units
        # Rare and unusual
        "𐍈𐍉𐍊",  # Gothic (if supported)
        "𒀀𒀁𒀂",  # Cuneiform (if supported)
        "𓀀𓀁𓀂",  # Egyptian hieroglyphs (if supported)
        "𝄞𝄢𝄪",  # Musical notation (if supported)
        "⏰⏱⏲⏳",  # Clock symbols
        "☀☁☂☃",  # Weather
        "✈✉✊✋",  # Transport and signs
    ]

    cases = {}

    for i, text in enumerate(test_strings):
        if len(cases) >= max_cases:
            break
        try:
            # Handle null bytes for JSON
            safe_text = text.replace("\x00", "\ufffd")

            # Tokenize (keep default special-token checks)
            tokens = enc.encode(safe_text)

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
    test_strings = [
        # Basic ASCII and whitespace
        "",
        "Hello",
        "Hello, World!",
        "Hello World",
        "   ",
        "Hello\nWorld",
        "Hello\tWorld",
        "12345",
        "Hello123",
        "!@#$%",
        "Hello   World",
        "Hello\n\nWorld",
        "today\n ",
        "today\n \n",
        "today\n  \n",
        "The quick brown fox jumps over the lazy dog",
        "This is a longer test string with multiple words to test tokenization",
        "<|endoftext|>",
        "日本語",
        "中文",
        "العربية",
        "🎉🎊",
        "café",
        "naïve",
        "Hello\u0000World",
        "a" * 100,
        "ab" * 50,
        "<|fim_prefix|>Hello<|fim_suffix|>World<|fim_middle|>",
        "Hello  World",
        "Hello\r\nWorld",
        " Hello ",
        "\t\t\t",
        "\n\n\n",
        "A",
        "AB",
        "ABC",
        "hello world",
        "HELLO WORLD",
        "Hello World!",
        "Hello, World?",
        "Hello; World.",
        "'Hello' \"World\"",
        "Hello (World) [Test] {Data}",
        "Hello+World=Test",
        "Hello/World\\Test",
        "www.example.com",
        "user@example.com",
        "192.168.1.1",
        "2024-01-15",
        "$1,234.56",
        "Hello_World",
        "Hello-World",
        "test_function_name",
        "TestClassName",
        "0xDEADBEEF",
        "\u00a0",
        " \u00850",
        "\u200b",
        "\ufeff",
        "Привет мир",
        "Γειά σου Κόσμε",
        "שלום עולם",
        "हैलो वर्ल्ड",
        "한국어",
        "Hello世界",
        "Hello🌍World",
        "👋👨‍👩‍👧‍👦👨‍💻",
        "🏳️‍🌈🏴‍☠️",
        "H̷̛̪e̷̛̪l̷̛̪l̷̛̪ơ̷̪",
        "print('Hello World')",
        "function test() { return 42; }",
        "<html><body>Hello</body></html>",
        '{"key": "value"}',
        "# Heading",
        "**bold**",
        "`code`",
        "[link](http://example.com)",
        "word " * 20,
        "A" * 1000,
        "<|endoftext|> hello <|fim_prefix|>",
        "<|endoftext|> hello <|fim_prefix|> there <|fim_middle|>",
        "[THINK]T1[/THINK]",
        '[TOOL_CALLS]{"name":"F1","arguments":{}}',
        '[TOOL_RESULTS]{"content":"R1","call_id":"123"}',
        "But ird and ปี   ird   ด",
        "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
        "E=mc²",
        "H₂O",
        "∞ ± × ÷",
        "$100 €200 £300 ¥400",
        "© ® ™",
        "👩🏽‍💻👨🏻‍🚀🧑🏿‍🔬",
        "1️⃣2️⃣3️⃣#️⃣*️⃣",
        "❤️🧡💛💚💙💜🖤🤍🤎",
        "🏳️‍⚧️",
        "🇺🇳🇺🇸🇯🇵",
        "🫠🫡🪿",
        "e\u0301 != é",
        "\u00f1 n\u0303",
        "क्‍ष",
        "\u200fabc\u200edef",
        "\u2066left-to-right isolate\u2069",
        "\u2028line-separator\u2029paragraph",
    ]

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
            "max_cases": 100,
            "model_candidates": [
                "unsloth/gemma-3-4b-it",
                "google/gemma-3-4b-it",
                "google/gemma-3-1b-it",
            ],
        },
        {
            "family_id": "alibaba.qwen3_5",
            "backend": "hf",
            "revision": "main",
            "max_cases": 100,
            "model_candidates": ["Qwen/Qwen3.5-0.6B", "Qwen/Qwen3-0.6B"],
        },
        {
            "family_id": "meta.llama3",
            "backend": "hf",
            "revision": "main",
            "max_cases": 100,
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
            "max_cases": 100,
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
            "max_cases": 100,
            "model_candidates": [
                "ibm-granite/granite-4.0-h-1b",
                "ibm-granite/granite-4.0-1b",
            ],
        },
        {
            "family_id": "huggingface.smollm3",
            "backend": "hf",
            "revision": "main",
            "max_cases": 100,
            "model_candidates": [
                "HuggingFaceTB/SmolLM3-3B",
                "HuggingFaceTB/SmolLM3-3B-Base",
            ],
        },
        {
            "family_id": "mistral.gpt2_pretekken",
            "backend": "hf",
            "revision": "main",
            "max_cases": 100,
            "model_candidates": [
                "mistralai/Mistral-Small-24B-Instruct-2501",
                "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                "mistralai/Mistral-Nemo-Instruct-2407",
            ],
        },
        {
            "family_id": "deepseek.v3_0324",
            "backend": "hf",
            "revision": "main",
            "max_cases": 100,
            "model_candidates": [
                "deepseek-ai/DeepSeek-V3-0324",
                "deepseek-ai/DeepSeek-V3",
                "deepseek-ai/DeepSeek-R1-0528",
            ],
        },
        {
            "family_id": "microsoft.phi4",
            "backend": "hf",
            "revision": "main",
            "max_cases": 100,
            "model_candidates": ["microsoft/phi-4", "microsoft/Phi-4-mini-instruct"],
        },
        {
            "family_id": "mistral.tekken",
            "backend": "mistral-common",
            "revision": "main",
            "max_cases": 100,
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
            adapter = HFAdapter(family["model_candidates"], revision=family["revision"])
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
        "r50k_base": 150,  # GPT-2
        "cl100k_base": 100,  # GPT-4, ChatGPT
        "o200k_base": 100,  # GPT-4o
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
