package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.Verbatim;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.codecs.VideoSampler;
import com.qxotic.toknroll.IntSequence;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.messages.part.MediaPart;
import org.springframework.ai.chat.messages.part.MessagePart;
import org.springframework.ai.chat.messages.part.OpaquePayload;
import org.springframework.ai.chat.messages.part.ReasoningPart;
import org.springframework.ai.chat.messages.part.StreamingParts;
import org.springframework.ai.chat.messages.part.TextPart;
import org.springframework.ai.chat.messages.part.ToolCallPart;
import org.springframework.ai.chat.messages.part.ToolResultPart;
import org.springframework.ai.chat.messages.part.UnknownPart;
import org.springframework.ai.content.Media;
import org.springframework.ai.session.SessionEvent;
import org.springframework.ai.session.support.SessionEventCodec;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.DefaultToolDefinition;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.util.MimeType;

/**
 * Model-free mapping seam: round-trips, tool definition geometry, media rejection. Jinfer renders
 * its own fallback maps from the conversation (RenderMaps), so the fallback-map assertions of the
 * old suite have no counterpart here.
 */
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
    void toolCallbackDefinitionGeometry() {
        Tool tool = JinferMappings.toTools(List.of(weatherTool())).get(0);
        assertEquals("get_weather", tool.name());
        // the engine's templates key on this exact map geometry (see RenderMaps)
        assertEquals(
                Map.of(
                        "type",
                        "function",
                        "function",
                        Map.of(
                                "name",
                                "get_weather",
                                "description",
                                "Get current weather for a city",
                                "parameters",
                                Map.of(
                                        "type",
                                        "object",
                                        "properties",
                                        Map.of("city", Map.of("type", "string")),
                                        "required",
                                        List.of("city")))),
                tool.definition());
    }

    @Test
    void messagesRoundTrip() {
        List<Message> out =
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
                                        .build()),
                        VideoSampler.UNIFORM,
                        new Replies());
        assertEquals(Role.SYSTEM, out.get(0).role());
        assertEquals("be terse", out.get(0).text());
        assertEquals(Role.USER, out.get(1).role());
        assertEquals("hi", out.get(1).text());
        assertEquals(Role.ASSISTANT, out.get(2).role());
        Content.ToolCall call = (Content.ToolCall) out.get(2).content().get(1);
        assertEquals("abc", call.id());
        assertEquals("get_weather", call.name());
        assertEquals(Map.of("city", "Paris"), call.arguments());
        assertEquals(Role.TOOL, out.get(3).role());
        Content.ToolResult result = (Content.ToolResult) out.get(3).content().get(0);
        assertEquals("abc", result.callId());
        assertEquals("18C sunny", result.text());
    }

    @Test
    void replyToAssistantMessage() {
        Message reply =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Content.Reasoning("hmm"),
                                new Content.Text("the weather is nice"),
                                new Content.ToolCall("", "get_weather", Map.of("city", "Paris"))));
        AssistantMessage ai = JinferMappings.toAssistantMessage(reply);
        assertEquals("the weather is nice", ai.getText());
        assertTrue(ai.hasToolCalls());
        // pythonic syntaxes carry no ids: positional ones are minted (Ollama-style)
        assertEquals("call_0", ai.getToolCalls().get(0).id());
        assertEquals("function", ai.getToolCalls().get(0).type());
        assertEquals("{\"city\":\"Paris\"}", ai.getToolCalls().get(0).arguments());
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
                UnsupportedOperationException.class,
                () -> JinferMappings.toMessages(List.of(u), VideoSampler.UNIFORM, new Replies()));
    }

    @Test
    void imageBytesDecodeToBlob() throws Exception {
        var img = new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB);
        var png = new ByteArrayOutputStream();
        ImageIO.write(img, "png", png);
        UserMessage u =
                UserMessage.builder()
                        .text("look")
                        .media(
                                Media.builder()
                                        .mimeType(MimeType.valueOf("image/png"))
                                        .data(png.toByteArray())
                                        .build())
                        .build();
        List<Message> out =
                JinferMappings.toMessages(List.of(u), VideoSampler.UNIFORM, new Replies());
        assertTrue(out.get(0).content().get(1) instanceof Content.Media);
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
        List<Message> out =
                JinferMappings.toMessages(List.of(u), VideoSampler.UNIFORM, new Replies());
        assertTrue(out.get(0).content().get(1) instanceof Content.Media);
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
        UnsupportedOperationException e =
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                JinferMappings.toMessages(
                                        List.of(u), VideoSampler.UNIFORM, new Replies()));
        assertEquals(
                "media type application/pdf is not supported: jinfer takes image, audio and video",
                e.getMessage());
    }

    /** A decode failure names the input, inline or a file, and keeps the codec's reason. */
    @Test
    void undecodableMediaNamesTheInputAndTheReason(@TempDir Path dir) throws Exception {
        Media inlineImage =
                Media.builder()
                        .mimeType(MimeType.valueOf("image/png"))
                        .data(new byte[] {1, 2})
                        .build();
        UncheckedIOException inline =
                assertThrows(
                        UncheckedIOException.class,
                        () -> JinferMappings.media(inlineImage, VideoSampler.UNIFORM));
        assertTrue(
                inline.getMessage()
                        .startsWith("image inline image/png, 2 bytes could not be decoded: "),
                inline.getMessage());
        assertTrue(inline.getMessage().endsWith(inline.getCause().getMessage()));

        Path file = Files.write(dir.resolve("cat.wav"), new byte[] {'h', 'i'});
        Media fileAudio = new Media(MimeType.valueOf("audio/wav"), file.toUri());
        UncheckedIOException fromFile =
                assertThrows(
                        UncheckedIOException.class,
                        () -> JinferMappings.media(fileAudio, VideoSampler.UNIFORM));
        assertTrue(
                fromFile.getMessage().startsWith("audio " + file + " could not be decoded: "),
                fromFile.getMessage());
        assertTrue(fromFile.getMessage().endsWith(fromFile.getCause().getMessage()));
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
                UnsupportedOperationException.class,
                () -> JinferMappings.toMessages(List.of(u), VideoSampler.UNIFORM, new Replies()));
    }

    @Test
    void userPartsKeepTheirOrder() throws Exception {
        var img = new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB);
        var png = new ByteArrayOutputStream();
        ImageIO.write(img, "png", png);
        Media image =
                Media.builder()
                        .mimeType(MimeType.valueOf("image/png"))
                        .data(png.toByteArray())
                        .build();
        UserMessage u =
                UserMessage.builder()
                        .part(TextPart.of("Before:"))
                        .part(MediaPart.of(image))
                        .part(TextPart.of("After."))
                        .build();
        List<Content> parts =
                JinferMappings.toMessages(List.of(u), VideoSampler.UNIFORM, new Replies())
                        .get(0)
                        .content();
        assertEquals(3, parts.size());
        assertEquals("Before:", ((Content.Text) parts.get(0)).text());
        assertTrue(parts.get(1) instanceof Content.Media);
        assertEquals("After.", ((Content.Text) parts.get(2)).text());

        // the legacy builder orders text, then media; an empty text adds no part
        UserMessage legacy = UserMessage.builder().text("").media(image).build();
        List<Content> legacyParts =
                JinferMappings.toMessages(List.of(legacy), VideoSampler.UNIFORM, new Replies())
                        .get(0)
                        .content();
        assertEquals(1, legacyParts.size());
        assertTrue(legacyParts.get(0) instanceof Content.Media);
    }

    @Test
    void toolResultsAreReadByParts() {
        ToolResponseMessage m =
                ToolResponseMessage.builder()
                        .result(
                                new ToolResultPart(
                                        "abc",
                                        "get_weather",
                                        List.of(TextPart.of("18C"), TextPart.of(" sunny")),
                                        false,
                                        null,
                                        Map.of()))
                        .build();
        Content.ToolResult result =
                (Content.ToolResult)
                        JinferMappings.toMessages(List.of(m), VideoSampler.UNIFORM, new Replies())
                                .get(0)
                                .content()
                                .get(0);
        assertEquals("abc", result.callId());
        assertEquals("18C sunny", result.text());
    }

    @Test
    void aMediaToolResultIsRefusedInWords() {
        Media image =
                Media.builder()
                        .mimeType(MimeType.valueOf("image/png"))
                        .data(new byte[] {1, 2, 3})
                        .build();
        ToolResponseMessage m =
                ToolResponseMessage.builder()
                        .result(
                                new ToolResultPart(
                                        "abc",
                                        "screenshot",
                                        List.of(MediaPart.of(image)),
                                        false,
                                        null,
                                        Map.of()))
                        .build();
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                JinferMappings.toMessages(
                                        List.of(m), VideoSampler.UNIFORM, new Replies()));
        assertTrue(e.getMessage().contains("screenshot"), e.getMessage());
        assertTrue(e.getMessage().contains("image/png"), e.getMessage());
    }

    /** Another provider's block in a request may be its whole content: refused, never dropped. */
    @Test
    void aForeignPartInARequestIsRefusedByName() {
        UnknownPart document = new UnknownPart("anthropic", "document", "{}", null, Map.of());
        UserMessage user =
                UserMessage.builder().part(TextPart.of("Summarize:")).part(document).build();
        ToolResponseMessage tool =
                ToolResponseMessage.builder()
                        .result(
                                new ToolResultPart(
                                        "abc", "search", List.of(document), false, null, Map.of()))
                        .build();
        for (var m : List.of(user, tool)) {
            IllegalArgumentException e =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    JinferMappings.toMessages(
                                            List.of(m), VideoSampler.UNIFORM, new Replies()));
            assertTrue(
                    e.getMessage().endsWith("cannot carry this part: anthropic document"),
                    e.getMessage());
        }
    }

    @Test
    void assistantHistoryIsReadByParts() {
        AssistantMessage.ToolCall call =
                new AssistantMessage.ToolCall(
                        "c1", "function", "get_weather", "{\"city\":\"Paris\"}");
        AssistantMessage ai =
                AssistantMessage.builder()
                        // a streamed argument delta is not a call, a foreign payload's text is
                        // still the model's reasoning, an unknown block has nothing to render
                        .part(StreamingParts.partial(ToolCallPart.of(call), 0))
                        .part(
                                new ReasoningPart(
                                        "hmm",
                                        null,
                                        new OpaquePayload("anthropic", "sig", "x"),
                                        Map.of()))
                        .part(TextPart.of("The weather:"))
                        .part(
                                new UnknownPart(
                                        "anthropic", "redacted_thinking", "{}", null, Map.of()))
                        .part(ToolCallPart.of(call))
                        .build();
        List<Content> parts =
                JinferMappings.toMessages(List.of(ai), VideoSampler.UNIFORM, new Replies())
                        .get(0)
                        .content();
        assertEquals(3, parts.size());
        assertTrue(parts.get(0) instanceof Content.Reasoning);
        assertEquals("The weather:", ((Content.Text) parts.get(1)).text());
        assertEquals("c1", ((Content.ToolCall) parts.get(2)).id());
    }

    @Test
    void multipleToolResponsesShareOneMessage() {
        List<Message> out =
                JinferMappings.toMessages(
                        List.of(
                                ToolResponseMessage.builder()
                                        .responses(
                                                List.of(
                                                        new ToolResponseMessage.ToolResponse(
                                                                "a", "f", "1"),
                                                        new ToolResponseMessage.ToolResponse(
                                                                "b", "g", "2")))
                                        .build()),
                        VideoSampler.UNIFORM,
                        new Replies());
        assertEquals(1, out.size());
        assertEquals(Role.TOOL, out.get(0).role());
        assertEquals("a", ((Content.ToolResult) out.get(0).content().get(0)).callId());
        assertEquals("b", ((Content.ToolResult) out.get(0).content().get(1)).callId());
    }

    @Test
    void assistantWithOnlyToolCallsHasNoTextPart() {
        List<Message> out =
                JinferMappings.toMessages(
                        List.of(
                                AssistantMessage.builder()
                                        .content("")
                                        .toolCalls(
                                                List.of(
                                                        new AssistantMessage.ToolCall(
                                                                "i", "function", "f", null)))
                                        .build()),
                        VideoSampler.UNIFORM,
                        new Replies());
        assertEquals(1, out.get(0).content().size());
        Content.ToolCall call = (Content.ToolCall) out.get(0).content().get(0);
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
                Map.of(
                        "type",
                        "function",
                        "function",
                        Map.of("name", "bare", "description", "bare", "parameters", Map.of())),
                tool.definition());
    }

    @Test
    void positionalIdsIncrementAcrossCalls() {
        Message reply =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Content.ToolCall("", "f", Map.of()),
                                new Content.ToolCall("", "g", Map.of())));
        AssistantMessage ai = JinferMappings.toAssistantMessage(reply);
        assertEquals("call_0", ai.getToolCalls().get(0).id());
        assertEquals("call_1", ai.getToolCalls().get(1).id());
    }

    /** A short WAV of silence (0 Hz sine) for the blob-mapping tests. */
    private static byte[] silenceWav() {
        return toneWav(0, 0.05, 16000);
    }

    static byte[] toneWav(double hz, double seconds, int rate) {
        int n = (int) (seconds * rate);
        byte[] pcm = new byte[n * 2];
        for (int i = 0; i < n; i++) {
            short s = (short) (Math.sin(2 * Math.PI * hz * i / rate) * 12000);
            pcm[i * 2] = (byte) s;
            pcm[i * 2 + 1] = (byte) (s >> 8);
        }
        var out = new ByteArrayOutputStream();
        try {
            var data = new DataOutputStream(out);
            data.writeBytes("RIFF");
            data.writeInt(Integer.reverseBytes(36 + pcm.length));
            data.writeBytes("WAVEfmt ");
            data.writeInt(Integer.reverseBytes(16));
            data.writeShort(Short.reverseBytes((short) 1)); // PCM
            data.writeShort(Short.reverseBytes((short) 1)); // mono
            data.writeInt(Integer.reverseBytes(rate));
            data.writeInt(Integer.reverseBytes(rate * 2));
            data.writeShort(Short.reverseBytes((short) 2));
            data.writeShort(Short.reverseBytes((short) 16));
            data.writeBytes("data");
            data.writeInt(Integer.reverseBytes(pcm.length));
            data.write(pcm);
        } catch (IOException impossible) {
            throw new AssertionError(impossible);
        }
        return out.toByteArray();
    }

    @Test
    void reasoningIsAPartAndReplaysAsAReasoningSpan() {
        // reply side: the reasoning is a ReasoningPart before the text, in the reply's order
        Message reply =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Content.Reasoning("hmm, let me think"),
                                new Content.Text("42")));
        AssistantMessage ai = JinferMappings.toAssistantMessage(reply);
        assertEquals("42", ai.getText());
        assertEquals(
                List.<MessagePart>of(ReasoningPart.of("hmm, let me think"), TextPart.of("42")),
                ai.getParts());

        // history side: the part renders back as a Reasoning span on the next turn (the
        // Ollama/OpenAI convention)
        List<Message> out =
                JinferMappings.toMessages(List.of(ai), VideoSampler.UNIFORM, new Replies());
        Content.Reasoning reasoning = (Content.Reasoning) out.get(0).content().get(0);
        assertEquals("hmm, let me think", reasoning.text());
        assertEquals("42", ((Content.Text) out.get(0).content().get(1)).text());
    }

    @Test
    void speechIsAWavMediaPartAfterItsTranscript() throws IOException {
        // a spoken reply: its words as text, the clip whole as a Content.Media audio
        float[] pcm = new float[2400];
        for (int i = 0; i < pcm.length; i++) pcm[i] = (float) Math.sin(i * 0.1);
        com.qxotic.jinfer.media.Media.Audio clip =
                new com.qxotic.jinfer.media.Media.Audio(pcm, 24000, 1);
        Verbatim record =
                new Verbatim(
                        new ContentKey("test-model"),
                        IntSequence.of(5, Verbatim.FRAME, 6),
                        IntSequence.of(new int[8]));
        Message reply =
                new Message(
                        Role.ASSISTANT,
                        List.of(new Content.Text("Paris."), new Content.Media(clip)),
                        record);
        AssistantMessage ai = JinferMappings.toAssistantMessage(reply);
        assertEquals("Paris.", ai.getText());
        assertEquals(2, ai.getParts().size());
        MediaPart speech = (MediaPart) ai.getParts().get(1);
        assertEquals(JinferMappings.WAV, speech.media().getMimeType());
        // the WAV names the clip's own rate and channels (a decode would resample to 16 kHz)
        ByteBuffer wav =
                ByteBuffer.wrap(speech.media().getDataAsByteArray()).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(1, wav.getShort(22), "channels");
        assertEquals(24000, wav.getInt(24), "sample rate");
        assertEquals(pcm.length * 2, wav.getInt(40), "PCM16 bytes: the WAV is the clip");
        assertEquals(16000, AudioCodec.decode(wav.array()).sampleRate(), "decodable, resampled");

        // history side: our own reply replays its record, speech and all; a foreign assistant's
        // media part is not something the template can say, so it is left out
        Replies replies = new Replies();
        replies.remember(ai, record);
        assertEquals(
                record,
                JinferMappings.toMessages(List.of(ai), VideoSampler.UNIFORM, replies)
                        .get(0)
                        .verbatim());
        AssistantMessage foreign =
                AssistantMessage.builder()
                        .part(TextPart.of("Paris."))
                        .part(MediaPart.of(JinferMappings.audio(clip)))
                        .build();
        Message replayed =
                JinferMappings.toMessages(List.of(foreign), VideoSampler.UNIFORM, new Replies())
                        .get(0);
        assertEquals(List.of(new Content.Text("Paris.")), replayed.content());
    }

    /**
     * A reply survives a session store (Spring AI's own codec, the JSON its JDBC repository
     * writes): the message carries Spring AI's fields only, and its stored echo still replays the
     * record. An edited echo, or one this model never produced, renders from its parts.
     */
    @Test
    void aStoredEchoReplaysItsRecord() {
        Verbatim record =
                new Verbatim(
                        new ContentKey("test-model"), IntSequence.of(7, 8, 9), IntSequence.empty());
        Message reply =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Content.Reasoning("checking"),
                                new Content.Text("Sunny."),
                                new Content.ToolCall("c1", "weather", Map.of("city", "Rome"))),
                        record);
        AssistantMessage delivered = JinferMappings.toAssistantMessage(reply);
        Replies replies = new Replies();
        replies.remember(delivered, record);
        SessionEventCodec codec = new SessionEventCodec();
        SessionEvent event = SessionEvent.builder().sessionId("s1").message(delivered).build();
        AssistantMessage stored =
                (AssistantMessage)
                        codec.decode(codec.encode(event)).sessionId("s1").build().getMessage();

        assertEquals(
                record,
                JinferMappings.toMessages(List.of(stored), VideoSampler.UNIFORM, replies)
                        .get(0)
                        .verbatim());
        AssistantMessage edited = stored.mutate().content("Rainy.").build();
        assertEquals(
                Verbatim.NONE,
                JinferMappings.toMessages(List.of(edited), VideoSampler.UNIFORM, replies)
                        .get(0)
                        .verbatim());
        assertEquals(
                Verbatim.NONE,
                JinferMappings.toMessages(List.of(stored), VideoSampler.UNIFORM, new Replies())
                        .get(0)
                        .verbatim(),
                "another model's table, or this one after a restart");
    }

    @Test
    void noThinkingKeyWhenReplyHasNoReasoning() {
        Message reply = new Message(Role.ASSISTANT, List.of(new Content.Text("just text")));
        AssistantMessage ai = JinferMappings.toAssistantMessage(reply);
        assertTrue(!ai.getMetadata().containsKey("thinking"));
    }
}
