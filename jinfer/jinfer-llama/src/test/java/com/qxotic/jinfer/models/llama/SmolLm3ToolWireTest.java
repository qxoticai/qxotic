package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.testkit.AbstractToolWireTest;
import com.qxotic.toknroll.Tokenizer;
import java.nio.file.Path;
import java.util.Map;

/**
 * SmolLM3's generated call wire: {@code <tool_call>\n{"name": ..., "arguments":
 * ...}\n</tool_call>}.
 */
class SmolLm3ToolWireTest extends AbstractToolWireTest {

    @Override
    protected Path modelPath() {
        return Path.of("/home/mukel/Desktop/playground/models/ggml-org/SmolLM3-Q4_K_M.gguf");
    }

    @Override
    protected ChatTemplate template(Tokenizer tokenizer) {
        return new SmolLm3ChatTemplate(tokenizer);
    }

    @Override
    protected void call(TokenRuns runs, String name, Map<String, Object> args) {
        runs.id(SpecialTokens.require(tokenizer, "<tool_call>"))
                .text("\n{\"name\": \"" + name + "\", \"arguments\": ")
                .text(ToolCallSyntax.jinjaJson(args))
                .text("}\n")
                .id(SpecialTokens.require(tokenizer, "</tool_call>"));
    }

    @Override
    protected void malformedCall(TokenRuns runs) {
        runs.id(SpecialTokens.require(tokenizer, "<tool_call>"))
                .text("\n{{{{broken")
                .id(SpecialTokens.require(tokenizer, "</tool_call>"));
    }
}
