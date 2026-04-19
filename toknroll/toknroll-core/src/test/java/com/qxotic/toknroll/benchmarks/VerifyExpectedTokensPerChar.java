package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.*;
import com.qxotic.toknroll.gguf.ModelFamilyTokenizers;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.util.*;

public class VerifyExpectedTokensPerChar {
    public static void main(String[] args) {
        System.out.println("=== TikToken Encodings ===");
        for (String enc :
                List.of("r50k_base", "p50k_base", "p50k_edit", "cl100k_base", "o200k_base")) {
            try {
                Tokenizer t = TiktokenFixtures.createJtokkitTokenizer(enc);
                System.out.printf("%-30s expected=%.4f%n", enc, t.expectedTokensPerChar());
            } catch (Exception e) {
                System.out.println(enc + ": ERROR");
            }
        }

        System.out.println("\n=== Model Families ===");
        String[] families = {
            "google.gemma3",
            "google.gemma4",
            "alibaba.qwen3_5",
            "meta.llama3",
            "moonshot.kimi2_5",
            "ibm.granite4_0",
            "huggingface.smollm3",
            "mistral.gpt2_pretekken",
            "deepseek.v3_2",
            "microsoft.phi4",
            "mistral.tekken",
            "openai.gpt-oss"
        };
        for (String family : families) {
            try {
                Optional<Tokenizer> opt = ModelFamilyTokenizers.create(family);
                if (opt.isPresent()) {
                    System.out.printf(
                            "%-30s expected=%.4f%n", family, opt.get().expectedTokensPerChar());
                }
            } catch (Exception e) {
                System.out.println(family + ": ERROR");
            }
        }
    }
}
