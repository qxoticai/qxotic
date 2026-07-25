package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.testkit.AbstractToolWireTest;
import com.qxotic.toknroll.Tokenizer;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/** LFM2.5's generated call wire: the pythonic list inside {@code <|tool_call_start|>} spans. */
class Lfm2ToolWireTest extends AbstractToolWireTest {

    @Override
    protected Path modelPath() {
        return Path.of("/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
    }

    @Override
    protected ChatTemplate template(Tokenizer tokenizer) {
        return new Lfm2ChatTemplate(tokenizer);
    }

    @Override
    protected void call(TokenRuns runs, String name, Map<String, Object> args) {
        runs.id(SpecialTokens.require(tokenizer, "<|tool_call_start|>"))
                .text("[")
                .text(ToolCallSyntax.renderPythonic(List.of(new Part.ToolCall("", name, args))))
                .text("]")
                .id(SpecialTokens.require(tokenizer, "<|tool_call_end|>"));
    }

    @Override
    protected void malformedCall(TokenRuns runs) {
        runs.id(SpecialTokens.require(tokenizer, "<|tool_call_start|>"))
                .text("[not a( valid ]call")
                .id(SpecialTokens.require(tokenizer, "<|tool_call_end|>"));
    }
}
