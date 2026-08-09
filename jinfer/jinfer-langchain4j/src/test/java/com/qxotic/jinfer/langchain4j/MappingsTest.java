package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.media.VideoSampler;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.output.FinishReason;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.UncheckedIOException;
import java.net.URI;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;

/** The mapping seam, model-free: typed messages, fallback maps, tools, replies, finish reasons. */
class MappingsTest {

    @Test
    void conversationRoundTrip() {
        List<ChatMessage> in =
                List.of(
                        SystemMessage.from("be brief"),
                        UserMessage.from("hi"),
                        AiMessage.from("hello"),
                        ToolExecutionResultMessage.from("call1", "get_weather", "18C"));
        List<Message> out = Mappings.toMessages(in, VideoSampler.UNIFORM);
        assertEquals(Role.SYSTEM, out.get(0).role());
        assertEquals("be brief", out.get(0).text());
        assertEquals(Role.USER, out.get(1).role());
        assertEquals(Role.ASSISTANT, out.get(2).role());
        assertEquals(Role.TOOL, out.get(3).role());
        Part.ToolResult r = assertInstanceOf(Part.ToolResult.class, out.get(3).content().get(0));
        assertEquals("call1", r.callId());
        assertEquals("18C", r.text());
    }

    @Test
    void assistantToolCallsBothDirections() {
        AiMessage withCall =
                AiMessage.builder()
                        .toolExecutionRequests(
                                List.of(
                                        ToolExecutionRequest.builder()
                                                .id("c1")
                                                .name("get_weather")
                                                .arguments("{\"city\": \"Paris\"}")
                                                .build()))
                        .build();
        Message m = Mappings.toMessages(List.of(withCall), VideoSampler.UNIFORM).get(0);
        Part.ToolCall call = assertInstanceOf(Part.ToolCall.class, m.content().get(0));
        assertEquals("get_weather", call.name());
        assertEquals(Map.of("city", "Paris"), call.arguments());

        AiMessage back = Mappings.toAiMessage(m);
        assertTrue(back.hasToolExecutionRequests());
        assertEquals("get_weather", back.toolExecutionRequests().get(0).name());
        assertEquals("{\"city\":\"Paris\"}", back.toolExecutionRequests().get(0).arguments());
    }

    @Test
    void toolSpecCanonicalJson() {
        ToolSpecification spec =
                ToolSpecification.builder()
                        .name("get_weather")
                        .description("Get current weather for a city")
                        .parameters(
                                JsonObjectSchema.builder()
                                        .addStringProperty("city")
                                        .required("city")
                                        .build())
                        .build();
        Tool tool = Mappings.toTools(List.of(spec)).get(0);
        assertEquals("get_weather", tool.name());
        // Jinja tojson canonical form: ", " and ": " separators, insertion order - the exact
        // contract the native templates weld into the prompt (see Lfm2ToolOracle).
        assertEquals(
                "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\","
                        + " \"description\": \"Get current weather for a city\", \"parameters\":"
                        + " {\"type\": \"object\", \"properties\": {\"city\": {\"type\":"
                        + " \"string\"}}, \"required\": [\"city\"]}}}",
                tool.rawJson());
        // the cached-prompt hit test hangs on this: a REBUILT but identical spec renders equal,
        // so a request re-stating a view's welded tools compares equal by value, not identity
        ToolSpecification rebuilt =
                ToolSpecification.builder()
                        .name("get_weather")
                        .description("Get current weather for a city")
                        .parameters(
                                JsonObjectSchema.builder()
                                        .addStringProperty("city")
                                        .required("city")
                                        .build())
                        .build();
        assertEquals(Mappings.toTools(List.of(spec)), Mappings.toTools(List.of(rebuilt)));
    }

    @Test
    void fallbackMapsCarryToolCalls() {
        AiMessage withCall =
                AiMessage.builder()
                        .toolExecutionRequests(
                                List.of(
                                        ToolExecutionRequest.builder()
                                                .id("c1")
                                                .name("f")
                                                .arguments("{}")
                                                .build()))
                        .build();
        var maps = Mappings.toMessageMaps(List.of(withCall));
        Map<?, ?> m = (Map<?, ?>) maps.get(0);
        assertEquals("assistant", m.get("role"));
        List<?> calls = (List<?>) m.get("tool_calls");
        Map<?, ?> fn = (Map<?, ?>) ((Map<?, ?>) calls.get(0)).get("function");
        assertEquals("f", fn.get("name"));
    }

    @Test
    void reasoningSplitsIntoThinking() {
        Message reply =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Part.Reasoning(List.of(new Part.Text("pondering", null)), null),
                                new Part.Text("answer", null)));
        AiMessage ai = Mappings.toAiMessage(reply);
        assertEquals("pondering", ai.thinking());
        assertEquals("answer", ai.text());
    }

    @Test
    void finishReasons() {
        assertEquals(FinishReason.STOP, Mappings.toFinishReason("stop", false));
        assertEquals(FinishReason.TOOL_EXECUTION, Mappings.toFinishReason("stop", true));
        assertEquals(FinishReason.LENGTH, Mappings.toFinishReason("length", false));
        assertEquals(FinishReason.OTHER, Mappings.toFinishReason("abort", false));
    }

    @Test
    void imageContentDecodesToMediaBlob() throws Exception {
        var img = new BufferedImage(4, 3, BufferedImage.TYPE_INT_RGB);
        var png = new ByteArrayOutputStream();
        ImageIO.write(img, "png", png);
        String base64 = Base64.getEncoder().encodeToString(png.toByteArray());

        Message m =
                Mappings.toMessages(
                                List.of(UserMessage.from(ImageContent.from(base64, "image/png"))),
                                VideoSampler.UNIFORM)
                        .get(0);
        Part.Blob blob = assertInstanceOf(Part.Blob.class, m.content().get(0));
        var decoded = assertInstanceOf(Media.Image.class, blob.media());
        assertEquals(4, decoded.width());
        assertEquals(3, decoded.height());
    }

    @Test
    void badMediaRejectedLoudly() {
        assertThrows(
                UncheckedIOException.class,
                () ->
                        Mappings.toMessages(
                                List.of(UserMessage.from(ImageContent.from("aGk=", "image/png"))),
                                VideoSampler.UNIFORM));
        assertThrows(
                UnsupportedFeatureException.class,
                () ->
                        Mappings.toMessages(
                                List.of(
                                        UserMessage.from(
                                                ImageContent.from(
                                                        URI.create("https://example.com/a.png")))),
                                VideoSampler.UNIFORM));
    }
}
