package com.qxotic.jinfer.models.gptoss;

import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.testkit.AbstractToolWireTest;
import com.qxotic.jinfer.testkit.ModelFixture;
import com.qxotic.toknroll.Tokenizer;
import java.nio.file.Path;
import java.util.Map;

/**
 * Harmony's generated call wire: {@code <|channel|>commentary to=functions.N <|constrain|>json
 * <|message|>{args}<|call|>}. Reasoning is the analysis channel, content the final channel; each
 * message ends with {@code <|end|><|start|>assistant} framing, and a reply carries at most ONE call
 * ({@code <|call|>} is a stop token).
 */
class GptOssToolWireTest extends AbstractToolWireTest {

    @Override
    protected Path modelPath() {
        return ModelFixture.GPTOSS_20B_Q8.path();
    }

    @Override
    protected ChatTemplate template(Tokenizer tokenizer) {
        return new GptOssTurnTemplate(tokenizer, "2026-07-25");
    }

    @Override
    protected void call(TokenRuns runs, String name, Map<String, Object> args) {
        runs.id(SpecialTokens.require(tokenizer, "<|channel|>"))
                .text("commentary to=functions." + name + " ")
                .id(SpecialTokens.require(tokenizer, "<|constrain|>"))
                .text("json")
                .id(SpecialTokens.require(tokenizer, "<|message|>"))
                .text(ToolCallSyntax.jinjaJson(args))
                .id(SpecialTokens.require(tokenizer, "<|call|>"));
    }

    @Override
    protected void think(TokenRuns runs, String text) {
        runs.id(SpecialTokens.require(tokenizer, "<|channel|>"))
                .text("analysis")
                .id(SpecialTokens.require(tokenizer, "<|message|>"))
                .text(text)
                .id(SpecialTokens.require(tokenizer, "<|end|>"))
                .id(SpecialTokens.require(tokenizer, "<|start|>"))
                .text("assistant");
    }

    @Override
    protected void content(TokenRuns runs, String text) {
        runs.id(SpecialTokens.require(tokenizer, "<|channel|>"))
                .text("final")
                .id(SpecialTokens.require(tokenizer, "<|message|>"))
                .text(text)
                .id(SpecialTokens.require(tokenizer, "<|end|>"))
                .id(SpecialTokens.require(tokenizer, "<|start|>"))
                .text("assistant");
    }

    @Override
    protected void malformedCall(TokenRuns runs) {
        runs.id(SpecialTokens.require(tokenizer, "<|channel|>"))
                .text("commentary to=functions.broken ")
                .id(SpecialTokens.require(tokenizer, "<|constrain|>"))
                .text("json")
                .id(SpecialTokens.require(tokenizer, "<|message|>"))
                .text("this is not json")
                .id(SpecialTokens.require(tokenizer, "<|call|>"));
    }

    /** Harmony's reasoning wire is the analysis channel, not think markers. */
    @Override
    protected boolean hasThinkWire() {
        return true;
    }

    /** {@code <|call|>} is a stop token: one call per generation. */
    @Override
    protected boolean supportsMultipleCalls() {
        return false;
    }
}
