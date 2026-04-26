package com.qxotic.toknroll.testkit;

import java.util.List;

/** Shared declarative model-family test configuration. */
public final class FamilyTestSpecs {

    private FamilyTestSpecs() {}

    public static final List<String> SMOKE_TEXTS =
            List.of(
                    "Hello world",
                    "Tokenizer family validation",
                    "Whitespace\n\tand unicode 😀",
                    "cafe and symbols <>[]{}");

    public static final List<FamilySpec> FAMILIES =
            List.of(
                    new FamilySpec("google.gemma4", 1, 0),
                    new FamilySpec("nvidia.nemotron3_nano4b", 1, 0),
                    new FamilySpec("meta.llama3", 1, 0),
                    new FamilySpec("moonshot.kimi2_5", 1, 0),
                    new FamilySpec("huggingface.smollm3", 1, 0),
                    new FamilySpec("alibaba.qwen3_5", 1, 0),
                    new FamilySpec("ibm.granite4_0", 1, 0),
                    new FamilySpec("microsoft.phi4", 1, 0),
                    new FamilySpec("mistral.gpt2_pretekken", 1, 0),
                    new FamilySpec("openai.gpt-oss", 1, 0),
                    new FamilySpec("deepseek.v3_2", 1, 0),
                    new FamilySpec("deepseek.v4_pro", 1, 0),
                    new FamilySpec("zai.glm5_1", 1, 0),
                    new FamilySpec("minimax.m2_7", 1, 0),
                    new FamilySpec("xiaomi.mimo_v2_flash", 1, 0));

    public static final class FamilySpec {
        private final String familyId;
        private final double minExactRatio;
        private final int maxEncodeErrors;

        public FamilySpec(String familyId, double minExactRatio, int maxEncodeErrors) {
            this.familyId = familyId;
            this.minExactRatio = minExactRatio;
            this.maxEncodeErrors = maxEncodeErrors;
        }

        public String familyId() {
            return familyId;
        }

        public double minExactRatio() {
            return minExactRatio;
        }

        public int maxEncodeErrors() {
            return maxEncodeErrors;
        }
    }
}
