package com.qxotic.jinfer.models.nemotronh;

import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.testkit.AbstractToolWireTest;
import com.qxotic.toknroll.Tokenizer;
import java.nio.file.Path;
import java.util.Map;

/** Nemotron's generated call wire: the same XML function form as Qwen 3.5 inside its spans. */
class NemotronToolWireTest extends AbstractToolWireTest {

    @Override
    protected Path modelPath() {
        return Path.of(
                "/home/mukel/Desktop/playground/models/bartowski/"
                        + "nvidia_Nemotron-Cascade-2-30B-A3B-Q8_0.gguf");
    }

    @Override
    protected ChatTemplate template(Tokenizer tokenizer) {
        return new NemotronHTurnTemplate(tokenizer);
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
                .text("\nno function element\n")
                .id(SpecialTokens.require(tokenizer, "</tool_call>"));
    }
}
