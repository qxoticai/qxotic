package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.testkit.AbstractToolWireTest;
import com.qxotic.toknroll.Tokenizer;
import java.nio.file.Path;
import java.util.Map;

/**
 * MiniCPM5's generated call wire: {@code <function name="N"><param name="K">V</param></function>}
 * with CDATA for values containing {@code <}, {@code &} or newlines. The wire is untyped, so every
 * argument round-trips as a STRING.
 */
class MiniCpm5ToolWireTest extends AbstractToolWireTest {

    @Override
    protected Path modelPath() {
        return Path.of("/home/mukel/Desktop/playground/models/openbmb/MiniCPM5-1B-Q8_0.gguf");
    }

    @Override
    protected ChatTemplate template(Tokenizer tokenizer) {
        return new MiniCpm5ChatTemplate(tokenizer);
    }

    @Override
    protected void call(TokenRuns runs, String name, Map<String, Object> args) {
        runs.id(SpecialTokens.require(tokenizer, "<function")).text(" name=\"" + name + "\">");
        for (var e : args.entrySet()) {
            runs.text("<param name=\"" + e.getKey() + "\">")
                    .text(MiniCpmToolSyntax.paramValue(e.getValue()))
                    .id(SpecialTokens.require(tokenizer, "</param>"));
        }
        runs.id(SpecialTokens.require(tokenizer, "</function>"));
    }

    @Override
    protected void malformedCall(TokenRuns runs) {
        runs.id(SpecialTokens.require(tokenizer, "<function"))
                .text(" garbage without a name attribute")
                .id(SpecialTokens.require(tokenizer, "</function>"));
    }

    /** The wire is untyped: values arrive as the strings the template prints. */
    @Override
    protected Object expectedArg(Object value) {
        return MiniCpmToolSyntax.paramValue(value).replace("<![CDATA[", "").replace("]]>", "");
    }
}
