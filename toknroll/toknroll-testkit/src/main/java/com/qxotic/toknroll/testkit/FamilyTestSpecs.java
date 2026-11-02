package com.qxotic.toknroll.testkit;

import java.util.List;

/** The families of generate_ground_truth.py; each must be in the golden, and is tested. */
public final class FamilyTestSpecs {

    private FamilyTestSpecs() {}

    public static final List<String> FAMILIES =
            List.of(
                    "google.gemma3",
                    "google.gemma4",
                    "nvidia.nemotron3_nano4b",
                    "meta.llama3",
                    "moonshot.kimi2_5",
                    "huggingface.smollm3",
                    "alibaba.qwen3_5",
                    "ibm.granite4_0",
                    "mistral.gpt2_pretekken",
                    "mistral.v0_3",
                    "mistral.tekken",
                    "openai.gpt-oss",
                    "deepseek.v3_2",
                    "deepseek.v4_pro",
                    "zai.glm5_1",
                    "minimax.m2_7",
                    "xiaomi.mimo_v2_flash",
                    "liquid.lfm3",
                    "moonshot.kimi3",
                    "deepseek.v4_1",
                    "alibaba.qwen3_6",
                    "poolside.laguna_xs2_1",
                    "poolside.laguna_s2_1",
                    "inclusion.ling3_0",
                    "inclusion.ling3_1");
}
