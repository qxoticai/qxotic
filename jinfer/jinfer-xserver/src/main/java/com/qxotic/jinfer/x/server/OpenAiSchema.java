// OpenAI-compatible wire shapes: the JSON envelopes for chat/completions/responses (full and
// streaming chunks), usage, and llama.cpp-style timings. Pure builders from a Reply -
// no transport, no generation logic.
package com.qxotic.jinfer.x.server;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class OpenAiSchema {
    private OpenAiSchema() {}

    static Map<String, Object> usage(Reply result) {
        return Map.of(
                "prompt_tokens", result.promptTokens(),
                "completion_tokens", result.completionTokens(),
                "total_tokens", result.promptTokens() + result.completionTokens(),
                "prompt_tokens_details", Map.of("cached_tokens", result.cachedTokens()));
    }

    /** llama.cpp-compatible timings extension: per-phase durations and rates. */
    static Map<String, Object> timings(Reply result) {
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

    static Map<String, Object> chatCompletionResponse(String id, String modelId, Reply result) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        message.put("content", result.toolCalls().isEmpty() ? result.text() : null);
        if (result.reasoning() != null) message.put("reasoning_content", result.reasoning());
        if (!result.toolCalls().isEmpty())
            message.put("tool_calls", ToolCalls.toWire(result.toolCalls()));
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
            String id,
            String modelId,
            long created,
            Map<String, Object> delta,
            String finishReason) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("delta", delta);
        choice.put("finish_reason", finishReason);
        Map<String, Object> chunk = new LinkedHashMap<>();
        chunk.put("id", id);
        chunk.put("object", "chat.completion.chunk");
        chunk.put("created", created);
        chunk.put("model", modelId);
        chunk.put("choices", List.of(choice));
        return chunk;
    }

    // ---- text completions ----

    static Map<String, Object> completionResponse(String id, String modelId, Reply result) {
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
            String id, String modelId, long created, String text, String finishReason) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("text", text);
        choice.put("index", 0);
        choice.put("finish_reason", finishReason);
        Map<String, Object> chunk = new LinkedHashMap<>();
        chunk.put("id", id);
        chunk.put("object", "text_completion");
        chunk.put("created", created);
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

    /**
     * What this turn actually produced, as Responses-API output items: one message, or the function
     * calls. The streaming handler emits these as {@code response.output_item.done} and the
     * envelope below carries the same list - one answer, so the item stream and the final response
     * cannot disagree. They used to: a tool-call reply streamed a COMPLETED message item holding
     * empty text while {@code response.completed} carried function_call items, so a client
     * following the item events saw an empty answer and never learned a tool had been called.
     */
    static List<Map<String, Object>> responseOutputItems(String id, Reply result) {
        return result.toolCalls().isEmpty()
                ? List.of(responseMessageItem("msg_" + id, "completed", result.text()))
                : responseToolCallItems(ToolCalls.toWire(result.toolCalls()));
    }

    static Map<String, Object> responseResponse(String id, String modelId, Reply result) {
        return responseResponse(id, modelId, result, responseOutputItems(id, result));
    }

    /**
     * As above over items the caller ALREADY built. The streaming handler must pass the very list
     * it emitted: a call the model did not name gets an id minted from the clock, so building the
     * items twice hands the same call two different {@code call_id}s - one in {@code
     * response.output_item.done}, another in {@code response.completed} - and a client correlating
     * them sees two calls where there was one.
     */
    static Map<String, Object> responseResponse(
            String id, String modelId, Reply result, List<Map<String, Object>> output) {
        return responseResponse(
                id, modelId, System.currentTimeMillis() / 1000, result, output);
    }

    static Map<String, Object> responseResponse(
            String id,
            String modelId,
            long created,
            Reply result,
            List<Map<String, Object>> output) {
        return responseEnvelope(id, modelId, created, "completed", output, responseUsage(result));
    }

    static Map<String, Object> responseEnvelope(
            String id,
            String modelId,
            long created,
            String status,
            List<Map<String, Object>> output,
            Map<String, Object> usage) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("id", id);
        response.put("object", "response");
        response.put("created_at", created);
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
                "in_progress".equals(status)
                        ? List.of()
                        : List.of(outputText(text)));
    }

    static Map<String, Object> outputText(String text) {
        return Map.of("type", "output_text", "text", text, "annotations", List.of());
    }

    private static Map<String, Object> responseUsage(Reply result) {
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
