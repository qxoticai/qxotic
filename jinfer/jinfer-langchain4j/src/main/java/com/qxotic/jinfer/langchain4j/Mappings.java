package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.OpenAiMaps;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyParts;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.media.AudioCodec;
import com.qxotic.jinfer.media.ImageCodec;
import com.qxotic.jinfer.media.VideoSampler;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.message.VideoContent;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.request.json.JsonAnyOfSchema;
import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonBooleanSchema;
import dev.langchain4j.model.chat.request.json.JsonEnumSchema;
import dev.langchain4j.model.chat.request.json.JsonIntegerSchema;
import dev.langchain4j.model.chat.request.json.JsonNullSchema;
import dev.langchain4j.model.chat.request.json.JsonNumberSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonRawSchema;
import dev.langchain4j.model.chat.request.json.JsonReferenceSchema;
import dev.langchain4j.model.chat.request.json.JsonSchemaElement;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * The mapping seam between langchain4j's message/tool model and jinfer's. Two output shapes per
 * conversation: typed {@link Message}s for the native codec, and OpenAI-shaped maps for the Jinja
 * whole-render fallback (templates read arbitrary fields, so the fallback speaks raw maps).
 */
final class Mappings {

    private Mappings() {}

    // ---- langchain4j -> jinfer (typed, for the native codec) ----

    static List<Message> toMessages(List<ChatMessage> messages, VideoSampler videoSampler) {
        List<Message> out = new ArrayList<>(messages.size());
        for (ChatMessage m : messages) {
            switch (m) {
                case SystemMessage s -> out.add(new Message(Role.SYSTEM, s.text()));
                case UserMessage u -> out.add(new Message(Role.USER, userParts(u, videoSampler)));
                case AiMessage ai -> out.add(assistant(ai));
                case ToolExecutionResultMessage r ->
                        out.add(
                                new Message(
                                        Role.TOOL, List.of(new Part.ToolResult(r.id(), r.text()))));
                default ->
                        throw new UnsupportedFeatureException(
                                "message type " + m.type() + " is not supported");
            }
        }
        return out;
    }

    /** User content, media included: text stays text, image/audio/video decode to media parts. */
    private static List<Part> userParts(UserMessage u, VideoSampler videoSampler) {
        List<Part> parts = new ArrayList<>();
        for (Content c : u.contents()) {
            switch (c) {
                case TextContent t -> parts.add(new Part.Text(t.text(), null));
                case ImageContent i -> {
                    byte[] src = bytes(i.image().base64Data(), i.image().url());
                    parts.add(blob("image", sha256(src), () -> ImageCodec.decode(src)));
                }
                case AudioContent a -> {
                    byte[] src = bytes(a.audio().base64Data(), a.audio().url());
                    parts.add(blob("audio", sha256(src), () -> AudioCodec.decode(src)));
                }
                case VideoContent v -> {
                    // no base64 door here, unlike image/audio: the frame sampler reads a FILE
                    // (ffmpeg seam), so the refusal must not advise the base64 the user just sent
                    if (v.video().base64Data() != null)
                        throw new UnsupportedFeatureException(
                                "inline base64 video is not supported: write it to a file and"
                                        + " pass its file:// URI");
                    Path src = localPath(v.video().url());
                    parts.add(blob("video", sha256(src), () -> videoSampler.sample(src)));
                }
                default ->
                        throw new UnsupportedFeatureException(
                                c.getClass().getSimpleName() + " is not supported");
            }
        }
        return parts;
    }

    private interface MediaDecode {
        Media decode() throws IOException;
    }

    private static Part.Blob blob(String kind, byte[] contentKey, MediaDecode decode) {
        try {
            return new Part.Blob(decode.decode(), contentKey);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to decode " + kind, e);
        }
    }

    /**
     * The SOURCE digest that keys media caching deterministically (encoder rows drift an ulp while
     * the JIT warms; the original bytes never do) - same law as the server wire. Video frames
     * derive per-frame keys from this digest in the template.
     */
    private static byte[] sha256(byte[] source) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(source);
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /**
     * Streaming digest of a local file - videos should not be pulled onto the heap to be hashed.
     */
    private static byte[] sha256(Path file) {
        try (var in = Files.newInputStream(file)) {
            var md = MessageDigest.getInstance("SHA-256");
            byte[] buf = new byte[1 << 16];
            for (int n; (n = in.read(buf)) > 0; ) md.update(buf, 0, n);
            return md.digest();
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read " + file, e);
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /** Inline base64 first; else a LOCAL file URI. The library never fetches over the network. */
    private static byte[] bytes(String base64, URI url) {
        if (base64 != null && !base64.isBlank()) {
            return Base64.getDecoder().decode(base64);
        }
        try {
            return Files.readAllBytes(localPath(url));
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read " + url, e);
        }
    }

    private static Path localPath(URI url) {
        if (url == null)
            throw new UnsupportedFeatureException("media needs base64 data or a file:// URI");
        if (!"file".equals(url.getScheme()))
            throw new UnsupportedFeatureException(
                    "remote image/media URLs are not supported ("
                            + url.getScheme()
                            + "): the library never fetches over the network; pass bytes or a"
                            + " file:// URI");
        return Path.of(url);
    }

    private static String userText(UserMessage u) {
        StringBuilder sb = new StringBuilder();
        for (Content c : u.contents()) {
            if (c instanceof TextContent t) sb.append(t.text());
        }
        return sb.toString();
    }

    private static Message assistant(AiMessage ai) {
        // an UNMODIFIED echo of a reply this provider produced restores the parsed Message with
        // its verbatim ids; an edited echo (or one from another provider) re-renders faithfully
        if (ai.attributes().get(REPLY_ATTRIBUTE) instanceof Message reply
                && toAiMessage(reply).equals(ai)) {
            return reply;
        }
        List<Part> parts = new ArrayList<>();
        if (ai.thinking() != null && !ai.thinking().isEmpty()) {
            parts.add(new Part.Reasoning(List.of(new Part.Text(ai.thinking(), null)), null));
        }
        if (ai.text() != null && !ai.text().isEmpty()) {
            parts.add(new Part.Text(ai.text(), null));
        }
        if (ai.hasToolExecutionRequests()) {
            for (ToolExecutionRequest r : ai.toolExecutionRequests()) {
                parts.add(
                        new Part.ToolCall(
                                r.id() == null ? "" : r.id(),
                                r.name(),
                                OpenAiMaps.args(r.arguments()),
                                null));
            }
        }
        return new Message(Role.ASSISTANT, parts);
    }

    static List<Tool> toTools(List<ToolSpecification> specs) {
        List<Tool> out = new ArrayList<>(specs.size());
        for (ToolSpecification spec : specs) {
            out.add(new Tool(spec.name(), ToolCallSyntax.jinjaJson(toolMap(spec))));
        }
        return out;
    }

    // ---- langchain4j -> OpenAI-shaped maps (for the Jinja whole-render fallback) ----

    static List<Object> toMessageMaps(List<ChatMessage> messages) {
        List<Object> out = new ArrayList<>(messages.size());
        for (ChatMessage m : messages) {
            var map = new LinkedHashMap<String, Object>();
            switch (m) {
                case SystemMessage s -> {
                    map.put("role", "system");
                    map.put("content", s.text());
                }
                case UserMessage u -> {
                    map.put("role", "user");
                    map.put("content", userText(u));
                }
                case AiMessage ai -> {
                    map.put("role", "assistant");
                    map.put("content", ai.text() == null ? "" : ai.text());
                    if (ai.hasToolExecutionRequests()) {
                        List<Object> calls = new ArrayList<>();
                        for (ToolExecutionRequest r : ai.toolExecutionRequests()) {
                            calls.add(OpenAiMaps.toolCall(r.id(), r.name(), r.arguments()));
                        }
                        map.put("tool_calls", calls);
                    }
                }
                case ToolExecutionResultMessage r ->
                        map.putAll(OpenAiMaps.toolResponse(r.text(), r.id(), r.toolName()));
                default ->
                        throw new UnsupportedFeatureException(
                                "message type " + m.type() + " is not supported");
            }
            out.add(map);
        }
        return out;
    }

    static List<Object> toToolMaps(List<ToolSpecification> specs) {
        List<Object> out = new ArrayList<>(specs.size());
        for (ToolSpecification spec : specs) out.add(toolMap(spec));
        return out;
    }

    private static Map<String, Object> toolMap(ToolSpecification spec) {
        return OpenAiMaps.tool(
                spec.name(),
                spec.description(),
                spec.parameters() != null ? toSchemaMap(spec.parameters()) : null);
    }

    // ---- JSON Schema: langchain4j's typed tree -> the plain map jinfer's grammar and templates
    // consume. Deliberately written against the PUBLIC dev.langchain4j.model.chat.request.json
    // types: langchain4j ships an internal JsonSchemaElementUtils.toMap that does this, but an
    // internal class can change in a patch release, and this is the one conversion both the tool
    // declarations and the structured-output grammar rest on. MappingsTest pins it against that
    // internal implementation as an oracle, so a semantic drift upstream fails a test rather than
    // a user's prompt. ----

    /** One schema element as a plain JSON-Schema map (recursive; ordering is insertion order). */
    static Map<String, Object> toSchemaMap(JsonSchemaElement element) {
        Map<String, Object> map = new LinkedHashMap<>();
        switch (element) {
            case JsonObjectSchema o -> {
                map.put("type", "object");
                putIfPresent(map, "description", o.description());
                Map<String, Object> properties = new LinkedHashMap<>();
                if (o.properties() != null) {
                    o.properties()
                            .forEach((name, child) -> properties.put(name, toSchemaMap(child)));
                }
                map.put("properties", properties);
                // ALWAYS present, empty list included: an object with no required properties still
                // renders "required": [] in the declaration models were trained on. Note
                // additionalProperties is deliberately NOT emitted - langchain4j's own conversion
                // drops it here, and these bytes go into the prompt
                map.put("required", o.required() == null ? List.of() : List.copyOf(o.required()));
                if (o.definitions() != null && !o.definitions().isEmpty()) {
                    Map<String, Object> defs = new LinkedHashMap<>();
                    o.definitions().forEach((name, child) -> defs.put(name, toSchemaMap(child)));
                    map.put("$defs", defs);
                }
            }
            case JsonArraySchema a -> {
                map.put("type", "array");
                putIfPresent(map, "description", a.description());
                map.put("items", toSchemaMap(a.items()));
            }
            case JsonEnumSchema e -> {
                map.put("type", "string");
                putIfPresent(map, "description", e.description());
                map.put("enum", List.copyOf(e.enumValues()));
            }
            case JsonAnyOfSchema anyOf -> {
                putIfPresent(map, "description", anyOf.description());
                List<Object> alternatives = new ArrayList<>();
                for (JsonSchemaElement child : anyOf.anyOf()) alternatives.add(toSchemaMap(child));
                map.put("anyOf", alternatives);
            }
            // the reference holds the DEFINITION NAME; the pointer prefix is added here (same rule
            // as langchain4j's conversion, so a name already spelled as a pointer double-prefixes)
            case JsonReferenceSchema r -> putIfPresent(map, "$ref", "#/$defs/" + r.reference());
            case JsonRawSchema raw -> {
                // already JSON text: parse it rather than re-describe it
                Object parsed = JsonCodec.parse(raw.schema());
                if (!(parsed instanceof Map)) {
                    throw new UnsupportedFeatureException(
                            "a raw JSON schema must be a JSON object, got: " + raw.schema());
                }
                @SuppressWarnings("unchecked")
                Map<String, Object> object = (Map<String, Object>) parsed;
                return object;
            }
            case JsonStringSchema s -> scalar(map, "string", s.description());
            case JsonIntegerSchema i -> scalar(map, "integer", i.description());
            case JsonNumberSchema n -> scalar(map, "number", n.description());
            case JsonBooleanSchema b -> scalar(map, "boolean", b.description());
            case JsonNullSchema ignored -> map.put("type", "null");
            default ->
                    throw new UnsupportedFeatureException(
                            "unsupported JSON schema element: " + element.getClass().getName());
        }
        return map;
    }

    private static void scalar(Map<String, Object> map, String type, String description) {
        map.put("type", type);
        putIfPresent(map, "description", description);
    }

    private static void putIfPresent(Map<String, Object> map, String key, Object value) {
        if (value != null) map.put(key, value);
    }

    // ---- jinfer reply -> langchain4j ----

    /**
     * The attribute carrying the parsed reply {@link Message} - verbatim token ids included -
     * across the langchain4j wire. An unmodified echo restores it ({@link #assistant}), letting the
     * native codec's verbatim splice re-encode the turn to the EXACT generated tokens (the
     * round-trip law; what makes {@code cachedSessions} hits deterministic instead of tokenization
     * luck).
     */
    static final String REPLY_ATTRIBUTE = "jinfer.reply";

    static AiMessage toAiMessage(Message reply) {
        ReplyParts parts = ReplyParts.of(reply);
        AiMessage.Builder b = AiMessage.builder();
        if (!parts.text().isEmpty()) b.text(parts.text());
        if (!parts.thinking().isEmpty()) b.thinking(parts.thinking());
        if (!parts.toolCalls().isEmpty()) {
            List<ToolExecutionRequest> calls = new ArrayList<>(parts.toolCalls().size());
            for (ReplyParts.ToolCall c : parts.toolCalls()) {
                calls.add(
                        ToolExecutionRequest.builder()
                                .id(c.id())
                                .name(c.name())
                                .arguments(c.argumentsJson())
                                .build());
            }
            b.toolExecutionRequests(calls);
        }
        // the parsed reply rides along whole: an unmodified echo restores verbatim ids instead of
        // re-tokenizing (what makes cachedSessions extension hits byte-exact)
        b.attributes(Map.of(REPLY_ATTRIBUTE, reply));
        return b.build();
    }

    /**
     * The same reply with replaced text - a stop-sequence cut. Attributes ride along: the reply
     * attribute is what lets an unmodified echo re-encode to the exact generated tokens.
     */
    static AiMessage withText(AiMessage ai, String text) {
        AiMessage.Builder b = AiMessage.builder();
        if (!text.isEmpty()) b.text(text);
        if (ai.thinking() != null && !ai.thinking().isEmpty()) b.thinking(ai.thinking());
        if (ai.hasToolExecutionRequests()) b.toolExecutionRequests(ai.toolExecutionRequests());
        b.attributes(ai.attributes());
        return b.build();
    }

    static ChatResponse response(
            String modelName, AiMessage ai, int promptTokens, ChatEngine.Completion done) {
        Generator.GenerationResult result = done.result();
        return ChatResponse.builder()
                .id(UUID.randomUUID().toString()) // generation identity for listeners
                .aiMessage(ai)
                .modelName(modelName)
                // cache read + phase timings ride the usage: cache behavior and generation
                // speed are diagnosable per response, not guessed from latency
                .tokenUsage(new JinferTokenUsage(promptTokens, done))
                .finishReason(
                        done.stopped() // a stop-sequence cut IS a stop, not an abort
                                ? FinishReason.STOP
                                : toFinishReason(
                                        result.finishReason(), ai.hasToolExecutionRequests()))
                .build();
    }

    static FinishReason toFinishReason(String jinferReason, boolean hasToolCalls) {
        if (hasToolCalls) return FinishReason.TOOL_EXECUTION;
        return switch (jinferReason) {
            case "stop" -> FinishReason.STOP;
            case "length" -> FinishReason.LENGTH;
            default -> FinishReason.OTHER;
        };
    }
}
