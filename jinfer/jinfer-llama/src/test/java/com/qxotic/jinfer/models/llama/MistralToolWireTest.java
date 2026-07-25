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
 * Ministral's generated call wire: {@code [TOOL_CALLS]name[ARGS]{json}} - no close marker, a call
 * ends at the next call or the {@code </s>} the reply always finishes with.
 */
class MistralToolWireTest extends AbstractToolWireTest {

    @Override
    protected Path modelPath() {
        return Path.of(
                "/home/mukel/Desktop/playground/models/unsloth/"
                        + "Ministral-3-3B-Instruct-2512-Q8_0.gguf");
    }

    @Override
    protected ChatTemplate template(Tokenizer tokenizer) {
        return new MistralChatTemplate(tokenizer);
    }

    @Override
    protected void call(TokenRuns runs, String name, Map<String, Object> args) {
        runs.id(SpecialTokens.require(tokenizer, "[TOOL_CALLS]"))
                .text(name)
                .id(SpecialTokens.require(tokenizer, "[ARGS]"))
                .text(ToolCallSyntax.jinjaJson(args));
    }

    @Override
    protected void endReply(TokenRuns runs) {
        runs.id(SpecialTokens.require(tokenizer, "</s>")); // the span-closing reply stop
    }

    @Override
    protected void malformedCall(TokenRuns runs) {
        runs.id(SpecialTokens.require(tokenizer, "[TOOL_CALLS]"))
                .text("name_without_args_marker {json}");
    }
}
