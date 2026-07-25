package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import java.net.URI;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.content.Media;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.DefaultToolDefinition;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.util.MimeType;

/** Model-free mapping seam: round-trips, canonical tool JSON, fallback maps, media rejection. */
class JinferMappingsTest {

    private static ToolCallback weatherTool() {
        ToolDefinition def =
                DefaultToolDefinition.builder()
                        .name("get_weather")
                        .description("Get current weather for a city")
                        .inputSchema(
                                "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}")
                        .build();
        return new ToolCallback() {
            @Override
            public ToolDefinition getToolDefinition() {
                return def;
            }

            @Override
            public String call(String toolInput) {
                return "";
            }
        };
    }

    @Test
    void toolCallbackCanonicalJson() {
        Tool tool = JinferMappings.toTools(List.of(weatherTool())).get(0);
        assertEquals("get_weather", tool.name());
        // Jinja tojson canonical form: ", " and ": " separators, insertion order - the exact
        // contract the native templates weld into the prompt (see Lfm2ToolOracle).
        assertEquals(
                "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\","
                        + " \"description\": \"Get current weather for a city\", \"parameters\":"
                        + " {\"type\": \"object\", \"properties\": {\"city\": {\"type\":"
                        + " \"string\"}}, \"required\": [\"city\"]}}}",
                tool.rawJson());
    }

    @Test
    void messagesRoundTrip() {
        List<com.qxotic.jinfer.chat.Message> out =
                JinferMappings.toMessages(
                        List.of(
                                new SystemMessage("be terse"),
                                new UserMessage("hi"),
                                AssistantMessage.builder()
                                        .content("let me check")
                                        .toolCalls(
                                                List.of(
                                                        new AssistantMessage.ToolCall(
                                                                "abc",
                                                                "function",
                                                                "get_weather",
                                                                "{\"city\":\"Paris\"}")))
                                        .build(),
                                ToolResponseMessage.builder()
                                        .responses(
                                                List.of(
                                                        new ToolResponseMessage.ToolResponse(
                                                                "abc", "get_weather", "18C sunny")))
                                        .build()));
        assertEquals(Role.SYSTEM, out.get(0).role());
        assertEquals("be terse", out.get(0).text());
        assertEquals(Role.USER, out.get(1).role());
        assertEquals("hi", out.get(1).text());
        assertEquals(Role.ASSISTANT, out.get(2).role());
        Part.ToolCall call = (Part.ToolCall) out.get(2).content().get(1);
        assertEquals("abc", call.id());
        assertEquals("get_weather", call.name());
        assertEquals(Map.of("city", "Paris"), call.arguments());
        assertEquals(Role.TOOL, out.get(3).role());
        Part.ToolResult result = (Part.ToolResult) out.get(3).content().get(0);
        assertEquals("abc", result.callId());
        assertEquals("18C sunny", result.text());
    }

    @Test
    void replyToAssistantMessage() {
        com.qxotic.jinfer.chat.Message reply =
                new com.qxotic.jinfer.chat.Message(
                        Role.ASSISTANT,
                        List.of(
                                new Part.Reasoning(List.of(new Part.Text("hmm", null)), null),
                                new Part.Text("the weather is nice", null),
                                new Part.ToolCall(
                                        "", "get_weather", Map.of("city", "Paris"), null)));
        AssistantMessage ai = JinferMappings.toAssistantMessage(reply);
        assertEquals("the weather is nice", ai.getText());
        assertTrue(ai.hasToolCalls());
        // pythonic syntaxes carry no ids: positional ones are minted (Ollama-style)
        assertEquals("call_0", ai.getToolCalls().get(0).id());
        assertEquals("function", ai.getToolCalls().get(0).type());
        assertEquals("{\"city\":\"Paris\"}", ai.getToolCalls().get(0).arguments());
    }

    @Test
    void fallbackMaps() {
        List<Object> maps =
                JinferMappings.toMessageMaps(
                        List.of(
                                new SystemMessage("sys"),
                                AssistantMessage.builder()
                                        .content("")
                                        .toolCalls(
                                                List.of(
                                                        new AssistantMessage.ToolCall(
                                                                "x", "function", "f", "{}")))
                                        .build(),
                                ToolResponseMessage.builder()
                                        .responses(
                                                List.of(
                                                        new ToolResponseMessage.ToolResponse(
                                                                "x", "f", "done")))
                                        .build()));
        assertEquals(Map.of("role", "system", "content", "sys"), maps.get(0));
        @SuppressWarnings("unchecked")
        Map<String, Object> assistant = (Map<String, Object>) maps.get(1);
        assertEquals("assistant", assistant.get("role"));
        assertTrue(assistant.containsKey("tool_calls"));
        // one OpenAI tool message per response
        assertEquals(
                Map.of("role", "tool", "content", "done", "tool_call_id", "x", "name", "f"),
                maps.get(2));
    }

    @Test
    void remoteMediaRejected() {
        UserMessage u =
                UserMessage.builder()
                        .text("what is this?")
                        .media(
                                new Media(
                                        MimeType.valueOf("image/png"),
                                        URI.create("https://example.com/x.png")))
                        .build();
        assertThrows(
                UnsupportedOperationException.class, () -> JinferMappings.toMessages(List.of(u)));
    }

    @Test
    void imageBytesDecodeToBlob() throws Exception {
        var img = new java.awt.image.BufferedImage(8, 8, java.awt.image.BufferedImage.TYPE_INT_RGB);
        var png = new java.io.ByteArrayOutputStream();
        javax.imageio.ImageIO.write(img, "png", png);
        UserMessage u =
                UserMessage.builder()
                        .text("look")
                        .media(
                                Media.builder()
                                        .mimeType(MimeType.valueOf("image/png"))
                                        .data(png.toByteArray())
                                        .build())
                        .build();
        List<com.qxotic.jinfer.chat.Message> out = JinferMappings.toMessages(List.of(u));
        Part.Blob blob = (Part.Blob) out.get(0).content().get(1);
        assertTrue(blob.media() instanceof com.qxotic.jinfer.Media.Image);
    }

    @Test
    void audioBytesDecodeToBlob() {
        UserMessage u =
                UserMessage.builder()
                        .text("listen")
                        .media(
                                Media.builder()
                                        .mimeType(MimeType.valueOf("audio/wav"))
                                        .data(silenceWav())
                                        .build())
                        .build();
        List<com.qxotic.jinfer.chat.Message> out = JinferMappings.toMessages(List.of(u));
        Part.Blob blob = (Part.Blob) out.get(0).content().get(1);
        assertTrue(blob.media() instanceof com.qxotic.jinfer.Media.Audio);
    }

    @Test
    void unsupportedMediaTypeRejected() {
        UserMessage u =
                UserMessage.builder()
                        .text("read")
                        .media(
                                Media.builder()
                                        .mimeType(MimeType.valueOf("application/pdf"))
                                        .data(new byte[] {1, 2, 3})
                                        .build())
                        .build();
        assertThrows(
                UnsupportedOperationException.class, () -> JinferMappings.toMessages(List.of(u)));
    }

    @Test
    void remoteAudioRejected() {
        UserMessage u =
                UserMessage.builder()
                        .text("listen")
                        .media(
                                new Media(
                                        MimeType.valueOf("audio/wav"),
                                        URI.create("https://example.com/x.wav")))
                        .build();
        assertThrows(
                UnsupportedOperationException.class, () -> JinferMappings.toMessages(List.of(u)));
    }

    @Test
    void multipleToolResponsesShareOneMessage() {
        List<com.qxotic.jinfer.chat.Message> out =
                JinferMappings.toMessages(
                        List.of(
                                ToolResponseMessage.builder()
                                        .responses(
                                                List.of(
                                                        new ToolResponseMessage.ToolResponse(
                                                                "a", "f", "1"),
                                                        new ToolResponseMessage.ToolResponse(
                                                                "b", "g", "2")))
                                        .build()));
        assertEquals(1, out.size());
        assertEquals(Role.TOOL, out.get(0).role());
        assertEquals("a", ((Part.ToolResult) out.get(0).content().get(0)).callId());
        assertEquals("b", ((Part.ToolResult) out.get(0).content().get(1)).callId());
    }

    @Test
    void assistantWithOnlyToolCallsHasNoTextPart() {
        List<com.qxotic.jinfer.chat.Message> out =
                JinferMappings.toMessages(
                        List.of(
                                AssistantMessage.builder()
                                        .content("")
                                        .toolCalls(
                                                List.of(
                                                        new AssistantMessage.ToolCall(
                                                                "i", "function", "f", null)))
                                        .build()));
        assertEquals(1, out.get(0).content().size());
        Part.ToolCall call = (Part.ToolCall) out.get(0).content().get(0);
        assertEquals(Map.of(), call.arguments()); // null arguments map to empty
    }

    @Test
    void toolWithoutOptionalFieldsOmitsThem() {
        // inputSchema is mandatory in Spring AI; description is not
        ToolDefinition def = DefaultToolDefinition.builder().name("bare").inputSchema("{}").build();
        ToolCallback cb =
                new ToolCallback() {
                    @Override
                    public ToolDefinition getToolDefinition() {
                        return def;
                    }

                    @Override
                    public String call(String toolInput) {
                        return "";
                    }
                };
        Tool tool = JinferMappings.toTools(List.of(cb)).get(0);
        // Spring AI defaults a missing description to the tool name
        assertEquals(
                "{\"type\": \"function\", \"function\": {\"name\": \"bare\", \"description\":"
                        + " \"bare\", \"parameters\": {}}}",
                tool.rawJson());
    }

    @Test
    void positionalIdsIncrementAcrossCalls() {
        com.qxotic.jinfer.chat.Message reply =
                new com.qxotic.jinfer.chat.Message(
                        Role.ASSISTANT,
                        List.of(
                                new Part.ToolCall("", "f", Map.of(), null),
                                new Part.ToolCall("", "g", Map.of(), null)));
        AssistantMessage ai = JinferMappings.toAssistantMessage(reply);
        assertEquals("call_0", ai.getToolCalls().get(0).id());
        assertEquals("call_1", ai.getToolCalls().get(1).id());
    }

    @Test
    void fallbackUserMapsCarryTextOnly() {
        List<Object> maps =
                JinferMappings.toMessageMaps(
                        List.of(
                                UserMessage.builder()
                                        .text("describe")
                                        .media(
                                                Media.builder()
                                                        .mimeType(MimeType.valueOf("image/png"))
                                                        .data(new byte[] {1})
                                                        .build())
                                        .build()));
        assertEquals(Map.of("role", "user", "content", "describe"), maps.get(0));
    }

    /** A short WAV of silence (0 Hz sine) for the blob-mapping tests. */
    private static byte[] silenceWav() {
        return Gemma4MediaIT.toneWav(0, 0.05, 16000);
    }

    @Test
    void thinkingStoredInMetadataAndReplayedAsReasoningPart() {
        // reply side: reasoning lands in AssistantMessage metadata (Ollama/OpenAI convention)
        com.qxotic.jinfer.chat.Message reply =
                new com.qxotic.jinfer.chat.Message(
                        Role.ASSISTANT,
                        List.of(
                                new Part.Reasoning(
                                        List.of(new Part.Text("hmm, let me think", null)), null),
                                new Part.Text("42", null)));
        AssistantMessage ai = JinferMappings.toAssistantMessage(reply);
        assertEquals("42", ai.getText());
        assertEquals("hmm, let me think", ai.getMetadata().get("thinking"));

        // history side: the stored thinking renders back as a Reasoning part on the next turn
        List<com.qxotic.jinfer.chat.Message> out = JinferMappings.toMessages(List.of(ai));
        Part.Reasoning reasoning = (Part.Reasoning) out.get(0).content().get(0);
        assertEquals("hmm, let me think", ((Part.Text) reasoning.content().get(0)).text());
        assertEquals("42", ((Part.Text) out.get(0).content().get(1)).text());
    }

    @Test
    void noThinkingKeyWhenReplyHasNoReasoning() {
        com.qxotic.jinfer.chat.Message reply =
                new com.qxotic.jinfer.chat.Message(
                        Role.ASSISTANT, List.of(new Part.Text("just text", null)));
        AssistantMessage ai = JinferMappings.toAssistantMessage(reply);
        assertTrue(!ai.getMetadata().containsKey("thinking"));
    }
}
