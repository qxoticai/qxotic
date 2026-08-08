package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.testkit.AbstractToolWireTest;
import com.qxotic.jinfer.testkit.ModelFixture;
import com.qxotic.toknroll.Tokenizer;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * Gemma 4's generated call wire: {@code <|tool_call>call:name{k:<|"|>v<|"|>}<tool_call|>} - the
 * compact notation with the trusted quote token.
 */
class Gemma4ToolWireTest extends AbstractToolWireTest {

    @Override
    protected Path modelPath() {
        return ModelFixture.GEMMA4_E2B_Q8.path();
    }

    @Override
    protected ChatTemplate template(Tokenizer tokenizer) {
        return new Gemma4TurnTemplate(tokenizer);
    }

    @Override
    protected void call(TokenRuns runs, String name, Map<String, Object> args) {
        runs.id(SpecialTokens.require(tokenizer, "<|tool_call>"));
        Gemma4ToolSyntax.call(
                name,
                args,
                new Gemma4ToolSyntax.Sink() {
                    @Override
                    public void text(String s) {
                        runs.text(s);
                    }

                    @Override
                    public void quote() {
                        runs.id(SpecialTokens.require(tokenizer, "<|\"|>"));
                    }
                });
        runs.id(SpecialTokens.require(tokenizer, "<tool_call|>"));
    }

    @Override
    protected void malformedCall(TokenRuns runs) {
        runs.id(SpecialTokens.require(tokenizer, "<|tool_call>"))
                .text("call:")
                .id(SpecialTokens.require(tokenizer, "<tool_call|>"));
    }

    /** Gemma renders call arguments dictsorted; expected maps must sort the same way. */
    @Override
    protected Map<String, Object> expected(Map<String, Object> args) {
        var sorted = new TreeMap<String, Object>(String.CASE_INSENSITIVE_ORDER);
        sorted.putAll(super.expected(args));
        return new LinkedHashMap<>(sorted);
    }
}
