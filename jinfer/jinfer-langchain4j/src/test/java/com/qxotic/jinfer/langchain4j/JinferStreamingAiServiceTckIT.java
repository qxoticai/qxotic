package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.output.TokenUsage;
import dev.langchain4j.service.common.AbstractStreamingAiServiceIT;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * The langchain4j streaming AiServices kit ({@code TokenStream}: partial-response ordering,
 * completion metadata, tool execution mid-stream) against JinferStreamingChatModel on LFM2.5-8B -
 * the streaming twin of {@link JinferAiServiceTckIT}, sharing one engine via {@code
 * JinferChatModel.streaming()}.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.JinferStreamingAiServiceTckIT#modelAvailable")
class JinferStreamingAiServiceTckIT extends AbstractStreamingAiServiceIT {

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
    protected List<StreamingChatModel> models() {
        return List.of(TckShield.streaming(shared().streaming()));
    }

    @Override
    protected Class<? extends TokenUsage> tokenUsageType(StreamingChatModel streamingChatModel) {
        return JinferTokenUsage.class;
    }
}
