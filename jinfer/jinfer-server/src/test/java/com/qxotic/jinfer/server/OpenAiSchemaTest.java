package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.llm.Generator;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;

class OpenAiSchemaTest {

    private static Reply reply(String text, List<Content.ToolCall> calls) {
        var result =
                new Generator.GenerationResult(
                        new int[] {1, 2},
                        OptionalInt.empty(),
                        Generator.FinishReason.STOP,
                        Duration.ZERO,
                        Duration.ZERO);
        return new Reply(result, 3, 0, text, null, calls, "tool_calls", null);
    }

    /** OpenAI's own order, extensions last, the same on every response. */
    @Test
    void responseKeysKeepOneOrder() {
        Reply reply = reply("hi", List.of());
        assertEquals(
                List.of("id", "object", "created", "model", "choices", "usage", "timings"),
                List.copyOf(OpenAiSchema.chatCompletionResponse("id", "m", reply).keySet()));
        assertEquals(
                List.of("id", "object", "created", "model", "choices", "usage", "timings"),
                List.copyOf(OpenAiSchema.completionResponse("id", "m", reply).keySet()));
        assertEquals(
                List.of(
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "prompt_tokens_details"),
                List.copyOf(OpenAiSchema.usage(reply).keySet()));
        String json = JsonCodec.stringify(OpenAiSchema.chatCompletionResponse("id", "m", reply));
        assertTrue(json.startsWith("{\"id\":\"id\",\"object\":\"chat.completion\","), json);
    }

    @Test
    void responsesReportTruncationWithoutClaimingTheOutputIsComplete() {
        var generation =
                new Generator.GenerationResult(
                        new int[] {1},
                        OptionalInt.empty(),
                        Generator.FinishReason.LENGTH,
                        Duration.ZERO,
                        Duration.ZERO);
        var partial = new Reply(generation, 3, 0, "{", null, List.of(), "length", null);
        Map<String, Object> response = OpenAiSchema.responseResponse("id", "m", partial);
        assertEquals("incomplete", response.get("status"));
        assertEquals(Map.of("reason", "max_output_tokens"), response.get("incomplete_details"));
        assertEquals(
                "incomplete",
                OpenAiSchema.responseOutputItems("id", partial).getFirst().get("status"));
        assertEquals(
                "{",
                Values.asObject(
                                Values.asArray(
                                                OpenAiSchema.responseOutputItems("id", partial)
                                                        .getFirst()
                                                        .get("content"),
                                                "content")
                                        .getFirst(),
                                "text")
                        .get("text"));

        // A user stop can coincide with the budget boundary; the resolved finish reason wins.
        var stopped = new Reply(generation, 3, 0, "ok", null, List.of(), "stop", null);
        assertEquals("completed", OpenAiSchema.responseResponse("id", "m", stopped).get("status"));
    }

    /** A deadline cuts the reply like the budget does; a client must be able to tell. */
    @Test
    void responsesReportADeadlineAsIncomplete() {
        for (var cut :
                Map.of(
                                Generator.FinishReason.TIMEOUT,
                                "timeout",
                                Generator.FinishReason.ABORT,
                                "cancelled")
                        .entrySet()) {
            var generation =
                    new Generator.GenerationResult(
                            new int[] {1, 2, 3},
                            OptionalInt.empty(),
                            cut.getKey(),
                            Duration.ZERO,
                            Duration.ZERO);
            var partial = new Reply(generation, 3, 0, "part", null, List.of(), "other", null);
            Map<String, Object> response = OpenAiSchema.responseResponse("id", "m", partial);
            assertEquals("incomplete", response.get("status"), cut.getValue());
            assertEquals(Map.of("reason", cut.getValue()), response.get("incomplete_details"));
            assertEquals(
                    "incomplete",
                    OpenAiSchema.responseOutputItems("id", partial).getFirst().get("status"));
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void textAlongsideToolCallsIsContentInEveryShape() {
        // "Let me check that.<tool_call>..." streamed those words as content deltas; the
        // non-streaming body and the Responses items must carry them too
        List<Content.ToolCall> calls =
                List.of(new Content.ToolCall("call_0", "get_weather", Map.of("city", "Zurich")));
        Reply withText = reply("Let me check that.", calls);
        Map<String, Object> message =
                (Map<String, Object>)
                        ((Map<String, Object>)
                                        ((List<Object>)
                                                        OpenAiSchema.chatCompletionResponse(
                                                                        "id", "m", withText)
                                                                .get("choices"))
                                                .get(0))
                                .get("message");
        assertEquals("Let me check that.", message.get("content"));
        List<Map<String, Object>> items = OpenAiSchema.responseOutputItems("id", withText);
        assertEquals(2, items.size());
        assertEquals("message", items.get(0).get("type"));
        assertEquals("function_call", items.get(1).get("type"));

        Reply callOnly = reply("", calls);
        Map<String, Object> bare =
                (Map<String, Object>)
                        ((Map<String, Object>)
                                        ((List<Object>)
                                                        OpenAiSchema.chatCompletionResponse(
                                                                        "id", "m", callOnly)
                                                                .get("choices"))
                                                .get(0))
                                .get("message");
        assertNull(bare.get("content"), "a call-only reply keeps the null content");
        assertEquals(1, OpenAiSchema.responseOutputItems("id", callOnly).size());
    }
}
