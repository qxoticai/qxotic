// OpenAI-compatible wire shapes: the JSON envelopes for chat/completions/responses (full and
// streaming chunks), usage, and llama.cpp-style timings. Pure builders from a GenerationResult
// or running Usage counters — no transport, no generation logic.
package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.Generator.GenerationResult;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class OpenAiSchema {
    private OpenAiSchema() {}

    /**
     * Mutable per-request token counters, updated by the generation pipeline and read by the
     * streaming sinks to attach running usage to delta chunks.
     */
    static final class Usage {
        int promptTokens;
        int completionTokens;
        int cachedTokens;
    }

    static Map<String, Object> chunkUsage(Usage usage) {
        return Map.of(
                "prompt_tokens",
                usage.promptTokens,
                "completion_tokens",
                usage.completionTokens,
                "total_tokens",
                usage.promptTokens + usage.completionTokens,
                "prompt_tokens_details",
                Map.of("cached_tokens", usage.cachedTokens));
    }

    static Map<String, Object> usage(GenerationResult result) {
        return Map.of(
                "prompt_tokens", result.promptTokens(),
                "completion_tokens", result.completionTokens(),
                "total_tokens", result.promptTokens() + result.completionTokens(),
                "prompt_tokens_details", Map.of("cached_tokens", result.cachedTokens()));
    }

    /** llama.cpp-compatible timings extension: per-phase durations and rates. */
    static Map<String, Object> timings(GenerationResult result) {
        Map<String, Object> timings = new LinkedHashMap<>();
        timings.put("prompt_n", result.promptTokens());
        timings.put("prompt_ms", Math.round(result.promptMillis() * 100.0) / 100.0);
        timings.put(
                "prompt_per_second",
                result.promptMillis() > 0
                        ? Math.round(result.promptTokens() / result.promptMillis() * 100_000.0)
                                / 100.0
                        : 0.0);
        timings.put("predicted_n", result.completionTokens());
        timings.put("predicted_ms", Math.round(result.predictedMillis() * 100.0) / 100.0);
        timings.put(
                "predicted_per_second",
                result.predictedMillis() > 0
                        ? Math.round(
                                        result.completionTokens()
                                                / result.predictedMillis()
                                                * 100_000.0)
                                / 100.0
                        : 0.0);
        timings.put("cached_n", result.cachedTokens());
        return timings;
    }

    // ---- chat completions ----

    static Map<String, Object> chatCompletionResponse(
            String id, String modelId, GenerationResult result) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        message.put("content", result.toolCalls().isEmpty() ? result.text() : null);
        if (result.reasoning() != null) message.put("reasoning_content", result.reasoning());
        if (!result.toolCalls().isEmpty()) message.put("tool_calls", result.toolCalls());
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("message", message);
        choice.put("finish_reason", result.finishReason());
        return Map.of(
                "id",
                id,
                "object",
                "chat.completion",
                "created",
                System.currentTimeMillis() / 1000,
                "model",
                modelId,
                "choices",
                List.of(choice),
                "usage",
                usage(result));
    }

    static Map<String, Object> chatCompletionChunk(
            String id, String modelId, Map<String, Object> delta, String finishReason) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("delta", delta);
        choice.put("finish_reason", finishReason);
        Map<String, Object> chunk = new LinkedHashMap<>();
        chunk.put("id", id);
        chunk.put("object", "chat.completion.chunk");
        chunk.put("created", System.currentTimeMillis() / 1000);
        chunk.put("model", modelId);
        chunk.put("choices", List.of(choice));
        return chunk;
    }

    // ---- text completions ----

    static Map<String, Object> completionResponse(
            String id, String modelId, GenerationResult result) {
        return Map.of(
                "id",
                id,
                "object",
                "text_completion",
                "created",
                System.currentTimeMillis() / 1000,
                "model",
                modelId,
                "choices",
                List.of(
                        Map.of(
                                "text",
                                result.text(),
                                "index",
                                0,
                                "finish_reason",
                                result.finishReason())),
                "usage",
                usage(result));
    }

    static Map<String, Object> completionChunk(
            String id, String modelId, String text, String finishReason) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("text", text);
        choice.put("index", 0);
        choice.put("finish_reason", finishReason);
        Map<String, Object> chunk = new LinkedHashMap<>();
        chunk.put("id", id);
        chunk.put("object", "text_completion");
        chunk.put("created", System.currentTimeMillis() / 1000);
        chunk.put("model", modelId);
        chunk.put("choices", List.of(choice));
        return chunk;
    }

    // ---- responses API ----

    /** A {@code response.output_text.delta} event payload (Responses API streaming). */
    static Map<String, Object> responseTextDelta(String itemId, String text) {
        Map<String, Object> delta = new LinkedHashMap<>();
        delta.put("type", "response.output_text.delta");
        delta.put("item_id", itemId);
        delta.put("output_index", 0);
        delta.put("content_index", 0);
        delta.put("delta", text);
        return delta;
    }

    static Map<String, Object> responseResponse(
            String id, String modelId, GenerationResult result) {
        List<Map<String, Object>> output =
                result.toolCalls().isEmpty()
                        ? List.of(responseMessageItem("msg_" + id, "completed", result.text()))
                        : responseToolCallItems(result.toolCalls());
        return responseEnvelope(id, modelId, "completed", output, responseUsage(result));
    }

    static Map<String, Object> responseEnvelope(
            String id,
            String modelId,
            String status,
            List<Map<String, Object>> output,
            Map<String, Object> usage) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("id", id);
        response.put("object", "response");
        response.put("created_at", System.currentTimeMillis() / 1000);
        response.put("status", status);
        response.put("model", modelId);
        response.put("output", output);
        response.put("parallel_tool_calls", false);
        response.put("tool_choice", "auto");
        response.put("usage", usage);
        return response;
    }

    static Map<String, Object> responseMessageItem(String id, String status, String text) {
        return Map.of(
                "id",
                id,
                "type",
                "message",
                "status",
                status,
                "role",
                "assistant",
                "content",
                List.of(Map.of("type", "output_text", "text", text, "annotations", List.of())));
    }

    private static Map<String, Object> responseUsage(GenerationResult result) {
        return Map.of(
                "input_tokens", result.promptTokens(),
                "output_tokens", result.completionTokens(),
                "total_tokens", result.promptTokens() + result.completionTokens());
    }

    private static List<Map<String, Object>> responseToolCallItems(
            List<Map<String, Object>> toolCalls) {
        List<Map<String, Object>> output = new ArrayList<>();
        for (Map<String, Object> toolCall : toolCalls) {
            Map<String, Object> function =
                    Values.asObject(toolCall.get("function"), "tool_call.function");
            output.add(
                    Map.of(
                            "id", Values.stringValue(toolCall.get("id"), ""),
                            "type", "function_call",
                            "status", "completed",
                            "call_id", Values.stringValue(toolCall.get("id"), ""),
                            "name", Values.stringValue(function.get("name"), ""),
                            "arguments", Values.stringValue(function.get("arguments"), "{}")));
        }
        return output;
    }
}
