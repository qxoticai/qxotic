package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

/**
 * Self-speculative decoding (Gemma 4's MTP draft head) through the Spring surface: the {@code
 * speculationDepth} knob rides the builder, output is byte-identical to plain greedy decode, and
 * the acceptance counters land on {@link JinferChatModel.JinferUsage}. Model-gated via {@link
 * TestModels}. Run: {@code mvn test -Dsurefire.excludedGroups= -Dgroups=integration -pl
 * jinfer-spring-ai}
 */
@Tag("integration")
class SpeculationIT {

    static final Path MODEL =
            TestModels.require("hf.co/unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf");
    static final Path MTP =
            TestModels.require("hf.co/unsloth/gemma-4-E2B-it-GGUF/mtp-gemma-4-E2B-it.gguf");

    private static JinferChatModel model(Integer depth) {
        return JinferChatModel.builder()
                .modelPath(MODEL)
                .companionPath("speculation", MTP)
                .contextLength(4096)
                .options(JinferChatOptions.builder().maxTokens(64).temperature(0.0).build())
                .speculationDepth(depth)
                .build();
    }

    private static String ask(JinferChatModel model) {
        // predictable content is where a draft head earns its keep (the bench law: lists and
        // code accept, prose does not)
        ChatResponse r =
                model.call(
                        new Prompt(new UserMessage("Count from 1 to 20, digits and spaces only.")));
        return r.getResult().getOutput().getText();
    }

    @Test
    void depthFourMatchesPlainGreedyByteForByte() {
        try (JinferChatModel plain = model(0);
                JinferChatModel speculating = model(4)) {
            assertEquals(ask(plain), ask(speculating));
        }
    }

    @Test
    void acceptanceCountersRideTheUsage() {
        try (JinferChatModel speculating = model(4)) {
            ask(speculating);
            ChatResponse r =
                    speculating.call(
                            new Prompt(new UserMessage("Count from 21 to 40, digits and spaces.")));
            var usage = r.getMetadata().getUsage();
            var nativeUsage = (JinferChatModel.JinferUsage) usage.getNativeUsage();
            assertNotNull(nativeUsage.speculatedDrafted(), "speculation ran, so it must account");
            assertNotNull(nativeUsage.speculatedAccepted());
            assertNotNull(nativeUsage.speculatedForwards());
            assertTrue(nativeUsage.speculatedDrafted() > 0, "nothing drafted");
            assertTrue(
                    nativeUsage.speculatedAccepted() <= nativeUsage.speculatedDrafted(),
                    "accepted more than drafted");
            // every accepted draft is a target forward saved: forwards < completion tokens
            assertTrue(
                    nativeUsage.speculatedForwards() < usage.getCompletionTokens(),
                    "no forward saved: forwards="
                            + nativeUsage.speculatedForwards()
                            + " tokens="
                            + usage.getCompletionTokens());
        }
    }

    @Test
    void depthZeroReportsNoSpeculation() {
        try (JinferChatModel plain = model(0)) {
            ask(plain);
            ChatResponse r = plain.call(new Prompt(new UserMessage("One word: ok?")));
            var nativeUsage =
                    (JinferChatModel.JinferUsage) r.getMetadata().getUsage().getNativeUsage();
            assertNull(nativeUsage.speculatedDrafted(), "depth 0 must not run the draft head");
        }
    }
}
