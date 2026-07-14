package com.qxotic.jinfer.server;

import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.llm.Generator;
import java.util.List;

/**
 * The server-side outcome of one generation pass: the raw token-level {@link
 * Generator.GenerationResult} plus everything the wire layer needs that the token loop no longer
 * knows - the decoded text (stop strings applied), the reasoning channel, the structured tool
 * calls, and the OpenAI finish_reason. Built by {@link Generation} from the chat decoder's parts;
 * {@link Server}/{@link OpenAiSchema} read only this.
 */
record Reply(
        Generator.GenerationResult result,
        String text,
        String reasoning,
        List<Part.ToolCall> toolCalls,
        String finishReason) {

    Reply {
        toolCalls = List.copyOf(toolCalls);
    }

    /** The reply re-tagged as parsed tool calls; {@code content} is any pre-marker text. */
    Reply asToolCalls(List<Part.ToolCall> calls, String content) {
        return new Reply(result, content, null, calls, "tool_calls");
    }

    /** Restamped billing counts (see {@link Generator.GenerationResult#withUsage}). */
    Reply withUsage(int promptTokens, int cachedTokens) {
        return new Reply(
                result.withUsage(promptTokens, cachedTokens),
                text,
                reasoning,
                toolCalls,
                finishReason);
    }

    // Token-level delegates, so the wire layer reads one object.

    com.qxotic.toknroll.IntSequence tokens() {
        return result.tokens();
    }

    int promptTokens() {
        return result.promptTokens();
    }

    int completionTokens() {
        return result.completionTokens();
    }

    int cachedTokens() {
        return result.cachedTokens();
    }

    double promptMillis() {
        return result.promptMillis();
    }

    double predictedMillis() {
        return result.predictedMillis();
    }
}
