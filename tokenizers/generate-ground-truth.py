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
    ]

    cases = {}

    for i, text in enumerate(test_strings[:max_cases]):
        try:
            # Handle null bytes for JSON
            safe_text = text.replace("\x00", "\ufffd")

            # Tokenize
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
            case_id = f"case_{i:03d}"
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
