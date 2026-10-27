package com.qxotic.jinfer.server;

import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.SpeculativeDecoding;
import java.util.List;

/** The protocol-independent result fields consumed by the OpenAI wire builders. */
record Reply(
        Generator.GenerationResult result,
        int promptTokens,
        int cachedTokens,
        String text,
        String reasoning,
        List<Content.ToolCall> toolCalls,
        String finishReason,
        SpeculativeDecoding.SpeculationResult speculation,
        Audio audio) {

    /**
     * A spoken reply's {@code audio} object: its id, when the id expires, the encoded speech (null
     * when it streamed out), and the transcript.
     */
    record Audio(String id, long expiresAt, String data, String transcript) {}

    Reply {
        toolCalls = List.copyOf(toolCalls);
    }

    /** A text reply. */
    Reply(
            Generator.GenerationResult result,
            int promptTokens,
            int cachedTokens,
            String text,
            String reasoning,
            List<Content.ToolCall> toolCalls,
            String finishReason,
            SpeculativeDecoding.SpeculationResult speculation) {
        this(
                result,
                promptTokens,
                cachedTokens,
                text,
                reasoning,
                toolCalls,
                finishReason,
                speculation,
                null);
    }

    Reply withAudio(Audio audio) {
        return new Reply(
                result,
                promptTokens,
                cachedTokens,
                text,
                reasoning,
                toolCalls,
                finishReason,
                speculation,
                audio);
    }

    int completionTokens() {
        return result.completionTokens();
    }

    double promptMillis() {
        return result.promptTime().toNanos() / 1e6;
    }

    double predictedMillis() {
        return result.decodeTime().toNanos() / 1e6;
    }
}
