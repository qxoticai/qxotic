package com.qxotic.jinfer.spring.ai;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.codecs.VideoSampler;
import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.messages.part.MediaPart;
import org.springframework.ai.chat.messages.part.MessagePart;
import org.springframework.ai.chat.messages.part.ReasoningPart;
import org.springframework.ai.chat.messages.part.StreamingParts;
import org.springframework.ai.chat.messages.part.TextPart;
import org.springframework.ai.chat.messages.part.ToolCallPart;
import org.springframework.ai.chat.messages.part.ToolResultPart;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.core.io.Resource;
import org.springframework.util.MimeType;

/**
 * The mapping seam between Spring AI's message/tool model and jinfer's. One output shape: typed
 * {@link Message}s - the engine renders its own fallback maps from the conversation, so the adapter
 * no longer speaks raw maps at all.
 */
final class JinferMappings {

    private static final Json.ParseOptions JSON =
            Json.ParseOptions.defaults().decimalsAsBigDecimal(false);

    private JinferMappings() {}

    // ---- Spring AI -> jinfer ----

    static List<Message> toMessages(
            List<org.springframework.ai.chat.messages.Message> messages,
            VideoSampler videoSampler) {
        List<Message> out = new ArrayList<>(messages.size());
        for (org.springframework.ai.chat.messages.Message m : messages) {
            switch (m) {
                case SystemMessage s -> out.add(new Message(Role.SYSTEM, s.getText()));
                case UserMessage u -> out.add(new Message(Role.USER, userParts(u, videoSampler)));
                case AssistantMessage ai -> out.add(assistant(ai));
                case ToolResponseMessage r -> {
                    List<Content> results = new ArrayList<>(r.getResults().size());
                    for (ToolResultPart t : r.getResults()) {
                        results.add(new Content.ToolResult(t.id(), toolResultText(t)));
                    }
                    out.add(new Message(Role.TOOL, results));
                }
                default ->
                        throw new IllegalArgumentException(
                                "message type " + m.getMessageType() + " is not supported");
            }
        }
        return out;
    }

    /**
     * User content in the user's order, text and media interleaved as written: text stays text,
     * image/audio/video decode to media parts.
     */
    private static List<Content> userParts(UserMessage u, VideoSampler videoSampler) {
        List<Content> parts = new ArrayList<>();
        for (MessagePart part : u.getParts()) {
            switch (part) {
                case TextPart t when t.text() != null && !t.text().isEmpty() ->
                        parts.add(new Content.Text(t.text(), null));
                case MediaPart m -> parts.add(media(m.media(), videoSampler));
                default -> {} // empty, or another provider's block
            }
        }
        return parts;
    }

    /**
     * A tool result's content as the text jinfer renders: its text parts joined. A tool result is
     * text here, so media in one is refused rather than dropped in silence.
     */
    private static String toolResultText(ToolResultPart result) {
        StringBuilder text = new StringBuilder();
        for (MessagePart part : result.content()) {
            switch (part) {
                case TextPart t -> text.append(t.text());
                case MediaPart m ->
                        throw new IllegalArgumentException(
                                "tool "
                                        + result.name()
                                        + " returned "
                                        + m.media().getMimeType()
                                        + " media: a tool result is text here");
                default -> {} // another provider's block
            }
        }
        return text.toString();
    }

    /** One image, audio or video, decoded by its MIME type; any other type is refused. */
    static Content.Media media(org.springframework.ai.content.Media media, VideoSampler sampler) {
        String kind = media.getMimeType().getType();
        return switch (kind) {
            case "image" -> {
                byte[] src = bytes(media);
                yield blob(kind, ContentKey.sha256(src), () -> ImageCodec.decode(src));
            }
            case "audio" -> {
                byte[] src = bytes(media);
                yield blob(kind, ContentKey.sha256(src), () -> AudioCodec.decode(src));
            }
            case "video" -> {
                Path src = localPath(media);
                yield blob(kind, sha256(src), () -> sampler.sample(src));
            }
            default ->
                    throw new UnsupportedOperationException(
                            "media type " + media.getMimeType() + " is not supported");
        };
    }

    private interface MediaDecode {
        Media decode() throws IOException;
    }

    private static Content.Media blob(String kind, ContentKey contentKey, MediaDecode decode) {
        try {
            return new Content.Media(decode.decode(), contentKey);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to decode " + kind, e);
        }
    }

    /**
     * Streaming digest of a local file - videos should not be pulled onto the heap to be hashed.
     */
    private static ContentKey sha256(Path file) {
        try (var in = Files.newInputStream(file)) {
            var md = MessageDigest.getInstance("SHA-256");
            byte[] buf = new byte[1 << 16];
            for (int n; (n = in.read(buf)) > 0; ) md.update(buf, 0, n);
            return new ContentKey("sha256:" + HexFormat.of().formatHex(md.digest()));
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read " + file, e);
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /** Inline/bytes or a LOCAL file - the library never fetches over the network. */
    private static byte[] bytes(org.springframework.ai.content.Media media) {
        Object data = rejectRemote(media.getData());
        if (data instanceof byte[] inline) return inline;
        Path file = localFile(data);
        if (file != null) {
            try {
                return Files.readAllBytes(file);
            } catch (IOException e) {
                throw new UncheckedIOException("cannot read media file " + file, e);
            }
        }
        return media.getDataAsByteArray();
    }

    /** A file:// URI (the Media(MimeType, URI) form, stored as a String) or a file Resource. */
    private static Path localFile(Object data) {
        if (data instanceof URI u && "file".equals(u.getScheme())) return Path.of(u);
        if (data instanceof String s && s.startsWith("file:")) return Path.of(URI.create(s));
        if (data instanceof Resource r) {
            try {
                return r.getFile().toPath();
            } catch (IOException e) {
                return null;
            }
        }
        return null;
    }

    private static Path localPath(org.springframework.ai.content.Media media) {
        Object data = rejectRemote(media.getData());
        Path file = localFile(data);
        if (file != null) return file;
        // inline bytes deferred - the fix is byte[] overloads on VideoCodec when a caller
        // actually holds bytes, NOT per-adapter temp-file handling. Spring AI 2.x reads a
        // Resource into bytes eagerly, so the only form that reaches here as a file is the
        // Media(MimeType, URI) constructor with a file:// URI
        if (data instanceof byte[])
            throw new UnsupportedOperationException(
                    "inline video bytes are not supported: write them to a file and pass"
                            + " Media(MimeType, URI) with a file:// URI");
        throw new UnsupportedOperationException(
                "video needs Media(MimeType, URI) with a file:// URI, got " + data);
    }

    private static Object rejectRemote(Object data) {
        String url = null;
        if (data instanceof String s) url = s; // the Media(MimeType, URI) ctor stores a String
        if (data instanceof URI u) url = u.toString();
        if (data instanceof Resource r) {
            try {
                url = r.getURI().toString();
            } catch (IOException ignored) {
                // not a URI-backed resource (bytes, stream) - fine
            }
        }
        if (url != null && url.startsWith("http")) {
            throw new UnsupportedOperationException(
                    "remote media URLs are not supported ("
                            + url
                            + "): the library never fetches over the network; pass bytes or a"
                            + " local file");
        }
        return data;
    }

    /**
     * Metadata key carrying the parsed reply {@link Message} whole. The round-trip law: an
     * UNMODIFIED assistant echo re-encodes to the exact tokens the model generated (verbatim ids),
     * which is what makes retained-session extension hits deterministic; an edited echo fails the
     * equality check and re-renders faithfully instead.
     */
    static final String REPLY_KEY = "jinfer.reply";

    /** A spoken reply's media type: the container that carries its own rate and channels. */
    static final MimeType WAV = MimeType.valueOf("audio/wav");

    private static Message assistant(AssistantMessage ai) {
        // an UNMODIFIED echo of a reply this provider produced restores the parsed Message with
        // its verbatim ids (byte-exact re-encode: what makes retained-session extension hits
        // deterministic); an edited echo (or one from another provider) re-renders faithfully
        if (ai.getMetadata().get(REPLY_KEY) instanceof Message reply && unchanged(ai, reply)) {
            return reply;
        }
        // the message's parts in their order: the reasoning the model produced earlier is
        // replayed too, the Ollama/OpenAI convention
        List<Content> parts = new ArrayList<>();
        for (MessagePart part : ai.getParts()) {
            switch (part) {
                case ReasoningPart r when r.text() != null && !r.text().isEmpty() ->
                        parts.add(
                                new Content.Reasoning(
                                        List.of(new Content.Text(r.text(), null)), null));
                case TextPart t when t.text() != null && !t.text().isEmpty() ->
                        parts.add(new Content.Text(t.text(), null));
                case ToolCallPart c when !StreamingParts.isPartial(c) -> {
                    AssistantMessage.ToolCall call = c.toolCall();
                    parts.add(
                            new Content.ToolCall(
                                    call.id() == null ? "" : call.id(),
                                    call.name(),
                                    args(call.arguments()),
                                    null));
                }
                default -> {} // empty, a streamed argument delta, media, another provider's block
            }
        }
        return new Message(Role.ASSISTANT, parts);
    }

    /**
     * Whether an echo says what the reply said: same text, reasoning and calls. A streamed echo's
     * parts were rebuilt from the deltas, and a reasoning span's seed newline never streamed, so
     * the parts themselves are not compared.
     */
    private static boolean unchanged(AssistantMessage echo, Message reply) {
        AssistantMessage parsed = toAssistantMessage(reply);
        return parsed.getText().equals(echo.getText())
                && reasoning(parsed).strip().equals(reasoning(echo).strip())
                && parsed.getToolCalls().equals(echo.getToolCalls());
    }

    /** A message's reasoning as one text. */
    static String reasoning(AssistantMessage message) {
        return message.getReasoning().stream()
                .map(ReasoningPart::text)
                .collect(Collectors.joining());
    }

    /** A call's arguments JSON as the map jinfer's tools speak; empty/blank = no arguments. */
    private static Map<String, Object> args(String argumentsJson) {
        if (argumentsJson == null || argumentsJson.isBlank()) return Map.of();
        return Json.parseMap(argumentsJson, JSON);
    }

    /** A JSON text as the engine's value-model map (Json.NULL becomes Java null). */
    @SuppressWarnings("unchecked")
    static Map<String, Object> jsonMap(String jsonText) {
        return (Map<String, Object>) nulls(Json.parse(jsonText, JSON));
    }

    static List<Tool> toTools(List<ToolCallback> callbacks) {
        List<Tool> out = new ArrayList<>(callbacks.size());
        for (ToolCallback cb : callbacks) {
            ToolDefinition def = cb.getToolDefinition();
            var function = new LinkedHashMap<String, Object>();
            function.put("name", def.name());
            if (def.description() != null) function.put("description", def.description());
            if (def.inputSchema() != null) {
                function.put("parameters", jsonMap(def.inputSchema()));
            }
            var definition = new LinkedHashMap<String, Object>();
            definition.put("type", "function");
            definition.put("function", function);
            out.add(new Tool(def.name(), definition));
        }
        return out;
    }

    /** Json.NULL -> Java null, in place (the parser's containers are mutable). */
    private static Object nulls(Object value) {
        if (value == Json.NULL) return null;
        if (value instanceof Map<?, ?> map) {
            @SuppressWarnings("unchecked")
            Map<Object, Object> mutable = (Map<Object, Object>) map;
            mutable.replaceAll((k, v) -> nulls(v));
            return map;
        }
        if (value instanceof List<?> list) {
            for (int i = 0; i < list.size(); i++) {
                @SuppressWarnings("unchecked")
                List<Object> mutable = (List<Object>) list;
                mutable.set(i, nulls(mutable.get(i)));
            }
            return list;
        }
        return value;
    }

    // ---- jinfer reply -> Spring AI ----

    /**
     * The reply as a message of parts in the reply's own order: reasoning, text, tool calls. The
     * parsed reply rides along whole, so an unmodified echo restores verbatim ids instead of
     * re-tokenizing.
     */
    static AssistantMessage toAssistantMessage(Message reply) {
        List<MessagePart> parts = new ArrayList<>();
        collect(reply.content(), parts);
        if (parts.isEmpty()) parts.add(TextPart.of("")); // an empty reply has an empty text
        Map<String, Object> properties = new HashMap<>();
        properties.put(REPLY_KEY, reply);
        return AssistantMessage.builder().parts(parts).properties(properties).build();
    }

    /**
     * The reply's content as parts: a reasoning span whole, text, calls with canonical JSON args.
     */
    /** A clip as Spring media: WAV, so the bytes name their own sample rate and channels. */
    static org.springframework.ai.content.Media audio(Media.Audio clip) {
        return org.springframework.ai.content.Media.builder()
                .mimeType(WAV)
                .data(AudioCodec.wav(clip))
                .build();
    }

    private static void collect(List<Content> content, List<MessagePart> parts) {
        for (Content part : content) {
            switch (part) {
                case Content.Text t when !t.text().isEmpty() -> parts.add(TextPart.of(t.text()));
                case Content.Reasoning r -> {
                    // Blank-only reasoning (a prompt-opened span's scaffold newlines, e.g. the
                    // closed empty pair when thinking is off) is no reasoning at all
                    String text =
                            r.content().stream()
                                    .filter(Content.Text.class::isInstance)
                                    .map(c -> ((Content.Text) c).text())
                                    .collect(Collectors.joining());
                    if (!text.isBlank()) parts.add(ReasoningPart.of(text));
                    // a call claimed inside the span is a call
                    collect(
                            r.content().stream().filter(c -> !(c instanceof Content.Text)).toList(),
                            parts);
                }
                case Content.Speech s -> parts.add(MediaPart.of(audio(s.audio())));
                case Content.ToolCall c -> {
                    // pythonic syntaxes carry no call ids: mint stable positional ones (what
                    // Ollama's server does); ids never render back into the prompt (the
                    // template's call syntax has no id slot), so echoes stay byte-identical
                    long calls = parts.stream().filter(ToolCallPart.class::isInstance).count();
                    parts.add(
                            ToolCallPart.of(
                                    new AssistantMessage.ToolCall(
                                            c.id().isEmpty() ? "call_" + calls : c.id(),
                                            "function",
                                            c.name(),
                                            Json.stringify(c.arguments()))));
                }
                default -> {} // ToolResult/Media never appear in a generated reply
            }
        }
    }
}
