package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/** Detects tool calls delimited by trusted special-token ids. */
final class SpanToolCallDetector {
    private final Tokenizer tokenizer;
    private final int start;
    private final int end;
    private final Function<String, List<Content.ToolCall>> parse;
    private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    private final List<Span> spans = new ArrayList<>();
    private IntSequence.Builder payloadIds = IntSequence.newBuilder();
    private IntSequence.Builder wireIds = IntSequence.newBuilder();
    private boolean inSpan;

    record Span(String text, IntSequence wireIds, List<Content.ToolCall> calls) {}

    SpanToolCallDetector(
            Tokenizer tokenizer,
            String start,
            String end,
            Function<String, List<Content.ToolCall>> parse) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.start = SpecialTokens.require(tokenizer, start);
        this.end = SpecialTokens.require(tokenizer, end);
        this.parse = Objects.requireNonNull(parse, "parse");
    }

    public boolean accept(int token) {
        if (token == start) {
            bytes.reset();
            inSpan = true;
            payloadIds = IntSequence.newBuilder();
            wireIds = IntSequence.newBuilder();
            wireIds.add(start);
            return true;
        }
        if (token == end) {
            if (inSpan) {
                wireIds.add(end);
                complete();
            }
            return true;
        }
        if (!inSpan) return false;
        bytes.writeBytes(tokenizer.decodeBytes(new int[] {token}));
        payloadIds.add(token);
        wireIds.add(token);
        return true;
    }

    private void complete() {
        String text = bytes.toString(StandardCharsets.UTF_8);
        List<Content.ToolCall> parsed =
                Objects.requireNonNull(parse.apply(text), "call parser returned null");
        IntSequence verbatim = payloadIds.build();
        List<Content.ToolCall> calls = new ArrayList<>(parsed.size());
        for (Content.ToolCall call : parsed) {
            calls.add(
                    parsed.size() == 1
                            ? new Content.ToolCall(
                                    call.id(), call.name(), call.arguments(), verbatim)
                            : call);
        }
        spans.add(new Span(text, wireIds.build(), List.copyOf(calls)));
        bytes.reset();
        inSpan = false;
    }

    public boolean inSpan() {
        return inSpan;
    }

    public List<Span> spans() {
        return List.copyOf(spans);
    }
}
