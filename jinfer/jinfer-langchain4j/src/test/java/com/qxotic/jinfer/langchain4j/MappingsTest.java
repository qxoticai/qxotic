package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.codecs.VideoSampler;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.media.Media;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.CustomMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.request.json.JsonAnyOfSchema;
import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonNullSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonReferenceSchema;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import dev.langchain4j.model.output.FinishReason;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.UncheckedIOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;

/** The mapping seam, model-free: typed messages, fallback maps, tools, replies, finish reasons. */
class MappingsTest {

    @Test
    void customMessageIsRefusedLoudly() {
        // the Ollama custom-role passthrough (a "context" role for guardian models) has no
        // jinfer equivalent: no family template has a slot for caller-invented roles, so a
        // CustomMessage must refuse loudly instead of being silently dropped from the prompt
        UnsupportedFeatureException e =
                assertThrows(
                        UnsupportedFeatureException.class,
                        () ->
                                Mappings.toMessages(
                                        List.of(
                                                UserMessage.from("hi"),
                                                CustomMessage.from(
                                                        Map.of("role", "context", "content", "x"))),
                                        VideoSampler.UNIFORM));
        assertTrue(e.getMessage().contains("CUSTOM"), e.getMessage());
    }

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
        Content.ToolResult r =
                assertInstanceOf(Content.ToolResult.class, out.get(3).content().get(0));
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
        Content.ToolCall call = assertInstanceOf(Content.ToolCall.class, m.content().get(0));
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
        // the OpenAI-shaped function definition the templates weld into the prompt
        assertEquals(
                Map.of(
                        "type",
                        "function",
                        "function",
                        Map.of(
                                "name", "get_weather",
                                "description", "Get current weather for a city",
                                "parameters",
                                        Map.of(
                                                "type", "object",
                                                "properties",
                                                        Map.of("city", Map.of("type", "string")),
                                                "required", List.of("city")))),
                tool.definition());
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
    void reasoningSplitsIntoThinking() {
        Message reply =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Content.Reasoning(
                                        List.of(new Content.Text("pondering", null)), null),
                                new Content.Text("answer", null)));
        AiMessage ai = Mappings.toAiMessage(reply);
        assertEquals("pondering", ai.thinking());
        assertEquals("answer", ai.text());
    }

    @Test
    void theSplitPreservesWhitespaceVerbatim() {
        // the GPULlama3 splitter laws at jinfer's structural boundary: NOTHING is trimmed or
        // re-spaced crossing the reasoning/content seam - downstream code (AiServices parsers,
        // user output) sees exactly the model's bytes
        Message reply =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Content.Reasoning(
                                        List.of(new Content.Text("  r\n\n r2  ", null)), null),
                                new Content.Text("line1\nline2\n\n  indented", null)));
        AiMessage ai = Mappings.toAiMessage(reply);
        assertEquals("  r\n\n r2  ", ai.thinking(), "thinking keeps its exact whitespace");
        assertEquals(
                "line1\nline2\n\n  indented",
                ai.text(),
                "text after a thinking block is not trimmed");

        // no reasoning: the text passes through byte-identical, padding included
        AiMessage plain =
                Mappings.toAiMessage(
                        new Message(Role.ASSISTANT, List.of(new Content.Text("  padded  ", null))));
        assertEquals("  padded  ", plain.text());
        assertNull(plain.thinking(), "no reasoning lane, no thinking field");

        // reasoning only: text stays absent (never an empty string inventing content)
        AiMessage thoughtsOnly =
                Mappings.toAiMessage(
                        new Message(
                                Role.ASSISTANT,
                                List.of(
                                        new Content.Reasoning(
                                                List.of(new Content.Text("hmm", null)), null))));
        assertNull(thoughtsOnly.text());
        assertEquals("hmm", thoughtsOnly.thinking());
    }

    @Test
    void finishReasons() {
        assertEquals(
                FinishReason.STOP, Mappings.toFinishReason(Generator.FinishReason.STOP, false));
        assertEquals(
                FinishReason.TOOL_EXECUTION,
                Mappings.toFinishReason(Generator.FinishReason.STOP, true));
        assertEquals(
                FinishReason.LENGTH, Mappings.toFinishReason(Generator.FinishReason.LENGTH, false));
        assertEquals(
                FinishReason.OTHER, Mappings.toFinishReason(Generator.FinishReason.ABORT, false));
        // a deadline is neither the model's end nor the token budget: OTHER, never LENGTH
        assertEquals(
                FinishReason.OTHER, Mappings.toFinishReason(Generator.FinishReason.TIMEOUT, false));
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
        Content.Media blob = assertInstanceOf(Content.Media.class, m.content().get(0));
        var decoded = assertInstanceOf(Media.Image.class, blob.value());
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

    // ---- the schema line: what the grammar cannot tell the model ----

    @Test
    void describeSchemaSketchesEveryShapeInOneLine() {
        JsonObjectSchema item =
                JsonObjectSchema.builder()
                        .addStringProperty("sku")
                        .addIntegerProperty("qty")
                        .build();
        JsonObjectSchema root =
                JsonObjectSchema.builder()
                        .addStringProperty("name", "full name")
                        .addIntegerProperty("year")
                        .addEnumProperty("genre", List.of("comedy", "drama"))
                        .addProperty(
                                "address",
                                JsonObjectSchema.builder().addStringProperty("city").build())
                        .addProperty("items", JsonArraySchema.builder().items(item).build())
                        .addProperty(
                                "nickname",
                                JsonAnyOfSchema.builder()
                                        .anyOf(new JsonStringSchema(), new JsonNullSchema())
                                        .build())
                        .addProperty("ref", JsonReferenceSchema.builder().reference("Item").build())
                        .definitions(Map.of("Item", item))
                        .required("name", "year")
                        .build();
        assertEquals(
                "Reply with JSON of this shape: {\"name\": string (full name), \"year\": integer,"
                        + " \"genre\": \"comedy\"|\"drama\", \"address\": {\"city\": string},"
                        + " \"items\": [{\"sku\": string, \"qty\": integer}], \"nickname\":"
                        + " string|null, \"ref\": {\"sku\": string, \"qty\": integer}} Leave out a"
                        + " field the text does not give.",
                Mappings.describeSchema(Mappings.toSchemaMap(root)));
    }

    @Test
    void theSchemaLineLandsOnTheLastUserMessageOnly() {
        List<Message> messages =
                new ArrayList<>(
                        List.of(
                                Message.system("Be terse."),
                                Message.user("first"),
                                Message.assistant("ok"),
                                Message.user("Extract the person from: Ada, 1815.")));
        JinferChatModel.describeSchemaOnTheLastUserMessage(
                messages,
                Mappings.toSchemaMap(
                        JsonObjectSchema.builder()
                                .addStringProperty("name")
                                .addIntegerProperty("year")
                                .build()));
        assertEquals("first", messages.get(1).text(), "earlier user turns are untouched");
        assertEquals(
                "Extract the person from: Ada, 1815.\n\nReply with JSON of this shape: {\"name\":"
                        + " string, \"year\": integer} Leave out a field the text does not give.",
                messages.get(3).text());
        assertEquals(Role.USER, messages.get(3).role());
    }

    /** A model may send a null argument; the adapter forwards JSON null instead of failing. */
    @Test
    void aNullToolArgumentIsForwardedAsJsonNull() {
        Map<String, Object> args = new java.util.LinkedHashMap<>();
        args.put("origin", "Zurich");
        args.put("flexible", null);
        Message reply =
                new Message(
                        Role.ASSISTANT,
                        List.of(new Content.ToolCall("c1", "book_flight", args, null)));
        AiMessage ai = Mappings.toAiMessage(reply);
        assertEquals(
                "{\"origin\":\"Zurich\",\"flexible\":null}",
                ai.toolExecutionRequests().get(0).arguments());
    }

    /** With tools offered, the line waits for a tool result: round one is the tool decision. */
    @Test
    void theSchemaLineWaitsForTheToolRound() {
        List<Message> roundOne = List.of(Message.user("Report the weather in Lisbon."));
        assertFalse(JinferChatModel.afterAToolRound(roundOne));
        List<Message> roundTwo =
                List.of(
                        Message.user("Report the weather in Lisbon."),
                        new Message(
                                Role.ASSISTANT,
                                List.of(
                                        new Content.ToolCall(
                                                "c1",
                                                "temperature",
                                                Map.of("city", "Lisbon"),
                                                null))),
                        new Message(Role.TOOL, List.of(new Content.ToolResult("c1", "21.5"))));
        assertTrue(JinferChatModel.afterAToolRound(roundTwo));
    }

    /** A stringified array or object is unwrapped only where the tool declares that shape. */
    @Test
    void stringifiedArgumentsUnwrapWhereTheToolDeclaresAnArrayOrObject() {
        Tool tag =
                new Tool(
                        "tag_ticket",
                        Map.of(
                                "type",
                                "function",
                                "function",
                                Map.of(
                                        "name",
                                        "tag_ticket",
                                        "parameters",
                                        Map.of(
                                                "type",
                                                "object",
                                                "properties",
                                                Map.of(
                                                        "priorities", Map.of("type", "array"),
                                                        "amounts", Map.of("type", "object"),
                                                        "note", Map.of("type", "string"))))));
        AiMessage sent =
                AiMessage.builder()
                        .toolExecutionRequests(
                                List.of(
                                        ToolExecutionRequest.builder()
                                                .id("c1")
                                                .name("tag_ticket")
                                                .arguments(
                                                        "{\"priorities\":\"[\\\"high\\\"]\",\"amounts\":\"{\\\"a\\\":1}\",\"note\":\"[keep]\"}")
                                                .build()))
                        .build();
        AiMessage fixed = Mappings.unwrapStringifiedArguments(sent, List.of(tag));
        assertEquals(
                "{\"priorities\":[\"high\"],\"amounts\":{\"a\":1},\"note\":\"[keep]\"}",
                fixed.toolExecutionRequests().get(0).arguments());
        assertEquals("c1", fixed.toolExecutionRequests().get(0).id());
        // a well-formed call, or an unknown tool, passes through untouched (same instance)
        AiMessage fine =
                AiMessage.builder()
                        .toolExecutionRequests(
                                List.of(
                                        ToolExecutionRequest.builder()
                                                .name("tag_ticket")
                                                .arguments("{\"priorities\":[\"low\"]}")
                                                .build()))
                        .build();
        assertTrue(Mappings.unwrapStringifiedArguments(fine, List.of(tag)) == fine);
        assertTrue(Mappings.unwrapStringifiedArguments(sent, List.of()) == sent);
        // the pythonic spelling a 1B model falls into is unwrapped too
        AiMessage pythonic =
                AiMessage.builder()
                        .toolExecutionRequests(
                                List.of(
                                        ToolExecutionRequest.builder()
                                                .name("tag_ticket")
                                                .arguments(
                                                        "{\"amounts\":\"{'travel': 120.5, 'meals':"
                                                                + " 42}\"}")
                                                .build()))
                        .build();
        assertEquals(
                "{\"amounts\":{\"travel\":120.5,\"meals\":42}}",
                Mappings.unwrapStringifiedArguments(pythonic, List.of(tag))
                        .toolExecutionRequests()
                        .get(0)
                        .arguments());
    }
}
