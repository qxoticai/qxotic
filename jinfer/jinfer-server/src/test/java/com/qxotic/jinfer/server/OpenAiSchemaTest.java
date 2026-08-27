package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

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
