package com.qxotic.toknroll.gguf;

/** Poolside Laguna XS 2.1: the {@code laguna} registration (CRLF-aware newline stage). */
final class LagunaGoldenParityTest extends GoldenParityTest {
    LagunaGoldenParityTest() {
        super(
                new Family(
                        "family-laguna-xs-2.1",
                        "https://huggingface.co/poolside/Laguna-XS-2.1-GGUF/resolve/main/Laguna-XS-2.1-Q4_K_M.gguf",
                        "laguna",
                        "poolside/Laguna-XS-2.1",
                        "laguna_golden_tokens.json"));
    }
}
