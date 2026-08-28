package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.output.TokenUsage;
import dev.langchain4j.service.common.AbstractAiServiceIT;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * The langchain4j AiServices compliance kit (chat + tools + structured-output interplay through the
 * high-level API) against JinferChatModel on LFM2.5-8B. The kit drives the same provider the
 * low-level TCK certifies, but through {@code AiServices.create(...)} - the shape applications
 * actually use.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.JinferAiServiceTckIT#modelAvailable")
class JinferAiServiceTckIT extends AbstractAiServiceIT {

    /**
     * The AiServices kits run on LFM2.5-2.6B, not the 8B-A1B the low-level TCK uses. The
     * kit's @Tool methods are compiled without -parameters and carry no descriptions, so they reach
     * the model as add(arg0, arg1); deciding that such a declaration applies is a capability, and
     * this checkpoint has it while the 8B-A1B does not (26 vs 14 of the tools battery, measured -
     * the 8B-A1B is a sparse MoE with ~1B ACTIVE parameters, so total size is not the axis here).
     * gemma-4-E2B and gemma-4-26B score the same 26; the 2.6B is simply the fastest and smallest of
     * the three. See ToolSpecProbe for the bare-vs-described evidence.
     */
    static final String REF = "hf.co/LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf";

    static boolean modelAvailable() {
        return TestModels.find(REF).isPresent();
    }

    private static JinferChatModel model;

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    static JinferChatModel shared() {
        if (model == null) {
            model =
                    JinferChatModel.builder()
                            .modelPath(TestModels.require(REF))
                            .contextLength(8192)
                            .maxOutputTokens(512)
                            // pinned greedy, same reason as the low-level TCK: the kit tests the
                            // CONTRACT, not sampling quality, and must not flake
                            .temperature(0.0)
                            // thinking ON, as the kit's models run it; the one
                            // exception (SmolLM3) is not this checkpoint
                            .thinking(true)
                            .seed(7L)
                            .build();
        }
        return model;
    }

    @Override
    protected List<ChatModel> models() {
        return List.of(TckShield.chat(shared()));
    }

    @Override
    protected Class<? extends TokenUsage> tokenUsageType(ChatModel chatModel) {
        return JinferTokenUsage.class;
    }
}
