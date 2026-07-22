package com.qxotic.jinfer.server;

import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.llm.Generator;
import java.util.List;

/**
 * The server-side outcome of one generation pass: the raw token-level {@link
 * Generator.GenerationResult} plus everything the wire layer needs that the token loop no longer
 * knows - the billed usage counts (BOS discount, cache restores - server policy), the decoded text
 * (stop strings applied), the reasoning channel, the structured tool calls, and the OpenAI
 * finish_reason. Built by {@link Generation}; {@link Server}/{@link OpenAiSchema} read only this.
 */
record Reply(
        Generator.GenerationResult result,
        int promptTokens,
        int cachedTokens,
        String text,
        String reasoning,
        List<Part.ToolCall> toolCalls,
        String finishReason) {

    Reply {
        toolCalls = List.copyOf(toolCalls);
    }

    /** The reply re-tagged as parsed tool calls; {@code content} is any pre-marker text. */
    Reply asToolCalls(List<Part.ToolCall> calls, String content) {
        return new Reply(result, promptTokens, cachedTokens, content, null, calls, "tool_calls");
    }

    // Token-level delegates, so the wire layer reads one object.

    com.qxotic.toknroll.IntSequence tokens() {
        return result.tokens();
    }

    int completionTokens() {
        return result.completionTokens();
    }

    double promptMillis() {
        return result.promptNanos() / 1e6;
    }

    double predictedMillis() {
        return result.predictedNanos() / 1e6;
    }
}
