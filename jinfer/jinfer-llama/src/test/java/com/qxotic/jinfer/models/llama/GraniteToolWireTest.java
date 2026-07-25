package com.qxotic.jinfer.models.llama;

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
 * Granite's generated call wire: {@code <tool_call>\n{"name": "N", "arguments": A}\n</tool_call>}.
 */
class GraniteToolWireTest extends AbstractToolWireTest {

    @Override
    protected Path modelPath() {
        return ModelFixture.GRANITE_41_3B_Q8.path();
    }

    @Override
    protected ChatTemplate template(Tokenizer tokenizer) {
        return new GraniteTurnTemplate(tokenizer);
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
                .text("\nnot json at all")
                .id(SpecialTokens.require(tokenizer, "</tool_call>"));
    }
}
