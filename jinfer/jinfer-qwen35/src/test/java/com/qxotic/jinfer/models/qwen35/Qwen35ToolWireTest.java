package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.testkit.AbstractToolWireTest;
import com.qxotic.toknroll.Tokenizer;
import java.nio.file.Path;
import java.util.Map;

/**
 * Qwen 3.5's generated call wire: the XML function form inside {@code <tool_call>} spans - {@code
 * \n<function=N>\n<parameter=K>\nV\n</parameter>\n</function>\n}.
 */
class Qwen35ToolWireTest extends AbstractToolWireTest {

    @Override
    protected Path modelPath() {
        return Path.of("/home/mukel/Desktop/playground/models/unsloth/Qwen3.5-4B-Q8_0.gguf");
    }

    @Override
    protected ChatTemplate template(Tokenizer tokenizer) {
        return new Qwen35TurnTemplate(tokenizer);
    }

    @Override
    protected void call(TokenRuns runs, String name, Map<String, Object> args) {
        StringBuilder body = new StringBuilder("\n<function=" + name + ">\n");
        for (var e : args.entrySet()) {
            body.append("<parameter=")
                    .append(e.getKey())
                    .append(">\n")
                    .append(ToolCallSyntax.jinjaValue(e.getValue()))
                    .append("\n</parameter>\n");
        }
        body.append("</function>\n");
        runs.id(SpecialTokens.require(tokenizer, "<tool_call>"))
                .text(body.toString())
                .id(SpecialTokens.require(tokenizer, "</tool_call>"));
    }

    @Override
    protected void malformedCall(TokenRuns runs) {
        runs.id(SpecialTokens.require(tokenizer, "<tool_call>"))
                .text("\nno function element here\n")
                .id(SpecialTokens.require(tokenizer, "</tool_call>"));
    }

    /** The XML wire types values by JSON-validity: strings stay raw, structures parse. */
    @Override
    protected Object expectedArg(Object value) {
        return value; // jinjaValue-printed then typedValue-parsed: identity for our fixtures
    }
}
