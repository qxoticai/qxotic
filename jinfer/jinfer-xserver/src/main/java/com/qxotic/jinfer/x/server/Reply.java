package com.qxotic.jinfer.x.server;

import com.qxotic.jinfer.x.chat.Content;
import com.qxotic.jinfer.x.llm.Generator;
import com.qxotic.jinfer.x.llm.SpeculativeDecoding;
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
        SpeculativeDecoding.SpeculationResult speculation) {

    Reply {
        toolCalls = List.copyOf(toolCalls);
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
