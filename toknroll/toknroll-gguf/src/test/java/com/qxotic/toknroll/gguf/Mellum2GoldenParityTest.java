package com.qxotic.toknroll.gguf;

/** JetBrains Mellum 2: the {@code mellum2} registration (single digits, then GPT-2 words). */
final class Mellum2GoldenParityTest extends GoldenParityTest {
    Mellum2GoldenParityTest() {
        super(
                new Family(
                        "family-mellum2-12b-a2.5b",
                        "https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q8_0/resolve/main/Mellum2-12B-A2.5B-Instruct-Q8_0.gguf",
                        "mellum2",
                        "JetBrains/Mellum2-12B-A2.5B-Instruct",
                        "mellum2_golden_tokens.json"));
    }
}
