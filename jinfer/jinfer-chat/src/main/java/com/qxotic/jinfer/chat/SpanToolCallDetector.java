package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.llm.GgufTokenizer;
import com.qxotic.toknroll.IntSequence;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * The generic marker-span tool-call detector: a call is the span between two trusted special-token
 * ids, its payload parsed by a model-supplied function (Pythonic, JSON, XML - see {@link
 * ToolCallSyntax}). Because the boundary is matched on ids (not decoded text), conversation content
 * can never fake a call. This is the one implementation marker-delimited models need; a
 * hand-written detector remains for formats that are not a flat marker span (Harmony channels).
 *
 * <p>The span's inner tokens are accumulated as raw bytes and decoded once at the close marker, so
 * a multibyte character split across two tokens still decodes correctly. When a span parses to
 * exactly one call, the call carries the span's payload ids as {@code verbatim}; a multi-call span
 * cannot attribute ids per call and leaves verbatim unset.
 */
public final class SpanToolCallDetector implements ToolCallDetector {

    private final GgufTokenizer tokenizer;
    private final int startMarker;
    private final int endMarker;
    private final Function<String, List<Part.ToolCall>> payloadParser;

    private boolean inSpan;
    private final ByteArrayOutputStream span = new ByteArrayOutputStream();
    private IntSequence.Builder spanIds = IntSequence.newBuilder();
    private final List<Part.ToolCall> calls = new ArrayList<>();

    public SpanToolCallDetector(
            GgufTokenizer tokenizer,
            String startMarker,
            String endMarker,
            Function<String, List<Part.ToolCall>> payloadParser) {
        this.tokenizer = tokenizer;
        this.startMarker = tokenizer.requiredSpecial(startMarker);
        this.endMarker = tokenizer.requiredSpecial(endMarker);
        this.payloadParser = payloadParser;
    }

    @Override
    public boolean accept(int token) {
        if (token == startMarker) {
            inSpan = true;
            span.reset();
            spanIds = IntSequence.newBuilder();
            return true;
        }
        if (token == endMarker) {
            if (inSpan) {
                List<Part.ToolCall> parsed =
                        payloadParser.apply(span.toString(StandardCharsets.UTF_8));
                IntSequence verbatim = spanIds.build();
                for (Part.ToolCall call : parsed) {
                    calls.add(
                            parsed.size() == 1
                                    ? new Part.ToolCall(
                                            call.id(), call.name(), call.arguments(), verbatim)
                                    : call);
                }
                span.reset();
                inSpan = false;
            }
            return true;
        }
        if (inSpan) {
            byte[] bytes = tokenizer.decodeTokenBytes(token);
            span.write(bytes, 0, bytes.length);
            spanIds.add(token);
            return true;
        }
        return false;
    }

    @Override
    public List<Part.ToolCall> calls() {
        return List.copyOf(calls);
    }
}
