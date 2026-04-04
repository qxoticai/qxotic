package com.qxotic.toknroll.testkit;

import java.util.List;

/** Shared declarative model-family test configuration. */
public final class FamilyTestSpecs {

    private FamilyTestSpecs() {}

    public static final int DEFAULT_SAMPLED_CASES = 12;

    public static final List<String> SMOKE_TEXTS =
            List.of(
                    "Hello world",
                    "Tokenizer family validation",
                    "Whitespace\n\tand unicode 😀",
                    "cafe and symbols <>[]{}");

    public static final List<FamilySpec> FAMILIES =
            List.of(
                    new FamilySpec("google.gemma3", DEFAULT_SAMPLED_CASES, 1, 0),
                    new FamilySpec("google.gemma4", DEFAULT_SAMPLED_CASES, 1, 0),
                    new FamilySpec("alibaba.qwen3_5", DEFAULT_SAMPLED_CASES, 1, 0),
                    new FamilySpec("meta.llama3", DEFAULT_SAMPLED_CASES, 1, 0),
                    new FamilySpec("moonshot.kimi2_5", DEFAULT_SAMPLED_CASES, 1, 0),
                    new FamilySpec("ibm.granite4_0", DEFAULT_SAMPLED_CASES, 1, 0),
                    new FamilySpec("huggingface.smollm3", DEFAULT_SAMPLED_CASES, 1, 0),
                    new FamilySpec("mistral.gpt2_pretekken", DEFAULT_SAMPLED_CASES, 1, 0),
                    new FamilySpec("deepseek.v3_0324", DEFAULT_SAMPLED_CASES, 1, 0),
                    new FamilySpec("microsoft.phi4", DEFAULT_SAMPLED_CASES, 1, 0),
                    new FamilySpec("mistral.tekken", DEFAULT_SAMPLED_CASES, 1, 0));

    public record FamilySpec(
            String familyId, int sampleCases, double minExactRatio, int maxEncodeErrors) {}
}
