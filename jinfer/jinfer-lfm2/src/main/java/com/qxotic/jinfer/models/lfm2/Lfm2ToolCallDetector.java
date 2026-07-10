package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ToolCallDetector;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.llm.GgufTokenizer;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * LFM2.5 tool-call detector. A call is the span between the trusted {@code <|tool_call_start|>} and
 * {@code <|tool_call_end|>} token ids; its payload is either a Pythonic call list {@code [f(a=1)]}
 * or a JSON {@code {name, arguments}} object/array, both handled by {@link ToolCallSyntax}. Because
 * the boundary is matched on ids (not decoded text), conversation content can never fake a call.
 *
 * <p>The span's inner tokens are accumulated as raw bytes and decoded once at the close marker, so
 * a multibyte character split across two tokens still decodes correctly.
 */
public final class Lfm2ToolCallDetector implements ToolCallDetector {

    private final GgufTokenizer tokenizer;
    private final int startMarker;
    private final int endMarker;

    private boolean inSpan;
    private final ByteArrayOutputStream span = new ByteArrayOutputStream();
    private final List<Part.ToolCall> calls = new ArrayList<>();

    public Lfm2ToolCallDetector(GgufTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.startMarker = tokenizer.requiredSpecial("<|tool_call_start|>");
        this.endMarker = tokenizer.requiredSpecial("<|tool_call_end|>");
    }

    @Override
    public boolean accept(int token) {
        if (token == startMarker) {
            inSpan = true;
            span.reset();
            return true;
        }
        if (token == endMarker) {
            if (inSpan) {
                calls.addAll(ToolCallSyntax.parseBlock(span.toString(StandardCharsets.UTF_8)));
                span.reset();
                inSpan = false;
            }
            return true;
        }
        if (inSpan) {
            byte[] bytes = tokenizer.decodeTokenBytes(token);
            span.write(bytes, 0, bytes.length);
            return true;
        }
        return false;
    }

    @Override
    public List<Part.ToolCall> calls() {
        return List.copyOf(calls);
    }
}
