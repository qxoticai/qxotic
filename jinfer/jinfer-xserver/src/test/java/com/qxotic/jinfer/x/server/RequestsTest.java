package com.qxotic.jinfer.x.server;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class RequestsTest {

    @Test
    void responsesInputPreservesMediaAndNormalizesTextParts() {
        Map<String, Object> image =
                Map.of(
                        "type",
                        "input_image",
                        "image_url",
                        "data:image/png;base64,AA==");
        Map<String, Object> request =
                Map.of(
                        "input",
                        List.of(
                                Map.of(
                                        "role",
                                        "user",
                                        "content",
                                        List.of(
                                                image,
                                                Map.of(
                                                        "type",
                                                        "input_text",
                                                        "text",
                                                        "describe")))));

        Map<?, ?> message = (Map<?, ?>) Requests.responseInputMessages(request).getFirst();
        List<?> content = (List<?>) message.get("content");

        assertEquals(image, content.getFirst());
        assertEquals(Map.of("type", "text", "text", "describe"), content.get(1));
    }

    @Test
    void responsesToolRoundTripKeepsCallIdentity() {
        Map<String, Object> request =
                new HashMap<>(
                        Map.of(
                                "input",
                                List.of(
                                        Map.of(
                                                "type",
                                                "function_call",
                                                "call_id",
                                                "call_42",
                                                "name",
                                                "weather",
                                                "arguments",
                                                "{\"city\":\"Zurich\"}"),
                                        Map.of(
                                                "type",
                                                "function_call_output",
                                                "call_id",
                                                "call_42",
                                                "output",
                                                List.of(
                                                        Map.of(
                                                                "type",
                                                                "input_text",
                                                                "text",
                                                                "Sunny")))),
                                "tools",
                                List.of(Map.of("type", "function", "name", "weather")),
                                "tool_choice",
                                Map.of("type", "function", "name", "weather")));

        Requests.normalizeResponse(request);
        List<Object> messages = Requests.responseInputMessages(request);

        Map<?, ?> assistant = (Map<?, ?>) messages.getFirst();
        Map<?, ?> call = (Map<?, ?>) ((List<?>) assistant.get("tool_calls")).getFirst();
        Map<?, ?> tool = (Map<?, ?>) messages.get(1);
        assertEquals("call_42", call.get("id"));
        assertEquals("call_42", tool.get("tool_call_id"));
        assertEquals(
                List.of(Map.of("type", "text", "text", "Sunny")), tool.get("content"));
        assertEquals(
                Map.of(
                        "type",
                        "function",
                        "function",
                        Map.of("name", "weather")),
                request.get("tool_choice"));
    }

    @Test
    void responsesJsonSchemaUsesTheSharedGrammarShape() {
        Map<String, Object> schema =
                Map.of(
                        "type",
                        "object",
                        "properties",
                        Map.of("answer", Map.of("type", "string")));
        Map<String, Object> request =
                new HashMap<>(
                        Map.of(
                                "input",
                                "Return JSON",
                                "text",
                                Map.of(
                                        "format",
                                        Map.of(
                                                "type",
                                                "json_schema",
                                                "name",
                                                "answer",
                                                "schema",
                                                schema,
                                                "strict",
                                                true))));

        Requests.normalizeResponse(request);

        assertEquals(
                Map.of(
                        "type",
                        "json_schema",
                        "json_schema",
                        Map.of("name", "answer", "schema", schema, "strict", true)),
                request.get("response_format"));
    }
}
