#!/usr/bin/env python3
"""
Generate ground_truth_tokens.json for tokenizer golden tests.
This creates a subset of test cases for each tiktoken encoding.
"""

import json
import base64
import sys
from pathlib import Path

# Add parent directory to path to import tiktoken
try:
    import tiktoken
except ImportError:
    print("Error: tiktoken not installed. Run: pip install tiktoken")
    print("Or use the system Python with tiktoken installed.")
    sys.exit(1)


def get_encoding(name):
    """Get a tiktoken encoding by name."""
    try:
        return tiktoken.get_encoding(name)
    except Exception as e:
        print(f"Error loading encoding {name}: {e}")
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
        # Zalgo and special Unicode
        "H̷̛̪e̷̛̪l̷̛̪l̷̛̪ơ̷̪",  # Zalgo text (combining marks)
        "ℌ𝔢𝔩𝔩𝔬",  # Mathematical fraktur
        "Ｈｅｌｌｏ",  # Fullwidth ASCII
        "🄷🄴🄻🄻🄾",  # Enclosed alphanumerics
        "ⓗⓔⓛⓛⓞ",  # Circled letters
        "🅷🅴🅻🅻🅾",  # Negative circled
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


if __name__ == "__main__":
    main()
