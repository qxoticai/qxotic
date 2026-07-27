package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.media.AudioCodec;
import com.qxotic.jinfer.media.ImageCodec;
import com.qxotic.jinfer.media.VideoCodec;
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
import dev.langchain4j.internal.JsonSchemaElementUtils;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
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

    static List<Message> toMessages(List<ChatMessage> messages) {
        List<Message> out = new ArrayList<>(messages.size());
        for (ChatMessage m : messages) {
            switch (m) {
                case SystemMessage s -> out.add(new Message(Role.SYSTEM, s.text()));
                case UserMessage u -> out.add(new Message(Role.USER, userParts(u)));
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
    private static List<Part> userParts(UserMessage u) {
        List<Part> parts = new ArrayList<>();
        for (Content c : u.contents()) {
            switch (c) {
                case TextContent t -> parts.add(new Part.Text(t.text(), null));
                case ImageContent i ->
                        parts.add(
                                blob(
                                        "image",
                                        () ->
                                                ImageCodec.decode(
                                                        bytes(
                                                                i.image().base64Data(),
                                                                i.image().url()))));
                case AudioContent a ->
                        parts.add(
                                blob(
                                        "audio",
                                        () ->
                                                AudioCodec.decode(
                                                        bytes(
                                                                a.audio().base64Data(),
                                                                a.audio().url()))));
                case VideoContent v ->
                        parts.add(blob("video", () -> VideoCodec.load(localPath(v.video().url()))));
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

    private static Part.Blob blob(String kind, MediaDecode decode) {
        try {
            return new Part.Blob(decode.decode());
        } catch (IOException e) {
            throw new UncheckedIOException("failed to decode " + kind, e);
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
                                argsMap(r.arguments()),
                                null));
            }
        }
        return new Message(Role.ASSISTANT, parts);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> argsMap(String argumentsJson) {
        if (argumentsJson == null || argumentsJson.isBlank()) return Map.of();
        Object parsed = JsonCodec.parse(argumentsJson);
        return parsed instanceof Map ? (Map<String, Object>) parsed : Map.of();
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
                            var call = new LinkedHashMap<String, Object>();
                            call.put("id", r.id() == null ? "" : r.id());
                            call.put("type", "function");
                            var fn = new LinkedHashMap<String, Object>();
                            fn.put("name", r.name());
                            fn.put("arguments", r.arguments());
                            call.put("function", fn);
                            calls.add(call);
                        }
                        map.put("tool_calls", calls);
                    }
                }
                case ToolExecutionResultMessage r -> {
                    map.put("role", "tool");
                    map.put("content", r.text());
                    map.put("tool_call_id", r.id());
                    map.put("name", r.toolName());
                }
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
        var fn = new LinkedHashMap<String, Object>();
        fn.put("name", spec.name());
        if (spec.description() != null) fn.put("description", spec.description());
        if (spec.parameters() != null) {
            fn.put("parameters", JsonSchemaElementUtils.toMap(spec.parameters()));
        }
        var tool = new LinkedHashMap<String, Object>();
        tool.put("type", "function");
        tool.put("function", fn);
        return tool;
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
        StringBuilder text = new StringBuilder();
        StringBuilder thinking = new StringBuilder();
        List<ToolExecutionRequest> calls = new ArrayList<>();
        collect(reply.content(), text, thinking, calls, false);
        AiMessage.Builder b = AiMessage.builder();
        if (!text.isEmpty()) b.text(text.toString());
        if (!thinking.isEmpty()) b.thinking(thinking.toString());
        if (!calls.isEmpty()) b.toolExecutionRequests(calls);
        b.attributes(Map.of(REPLY_ATTRIBUTE, reply));
        return b.build();
    }

    private static void collect(
            List<Part> parts,
            StringBuilder text,
            StringBuilder thinking,
            List<ToolExecutionRequest> calls,
            boolean inReasoning) {
        for (Part part : parts) {
            switch (part) {
                case Part.Text t -> (inReasoning ? thinking : text).append(t.text());
                case Part.Reasoning r -> collect(r.content(), text, thinking, calls, true);
                case Part.ToolCall c ->
                        // pythonic syntaxes carry no call ids: mint stable positional ones (what
                        // Ollama's server does); ids never render back into the prompt (the
                        // template's call syntax has no id slot), so echoes stay byte-identical
                        calls.add(
                                ToolExecutionRequest.builder()
                                        .id(c.id().isEmpty() ? "call_" + calls.size() : c.id())
                                        .name(c.name())
                                        .arguments(JsonCodec.stringify(c.arguments()))
                                        .build());
                default -> {} // ToolResult/Blob never appear in a generated reply
            }
        }
    }

    /**
     * {@code ai} with its text replaced (empty = none); thinking, tool calls AND attributes
     * preserved - dropping attributes would lose the {@link #REPLY_ATTRIBUTE} verbatim round-trip
     * on stop-sequence-cut replies (the Spring twin keeps its metadata the same way).
     */
    static AiMessage withText(AiMessage ai, String text) {
        AiMessage.Builder b = AiMessage.builder();
        if (!text.isEmpty()) b.text(text);
        if (ai.thinking() != null && !ai.thinking().isEmpty()) b.thinking(ai.thinking());
        if (ai.hasToolExecutionRequests()) b.toolExecutionRequests(ai.toolExecutionRequests());
        b.attributes(ai.attributes());
        return b.build();
    }

    /** The shared ChatResponse assembly (blocking and streaming build the identical shape). */
    static ChatResponse response(
            String modelName,
            AiMessage ai,
            int promptTokens,
            Generator.GenerationResult result,
            boolean stoppedBySequence) {
        return ChatResponse.builder()
                .id(UUID.randomUUID().toString()) // generation identity for listeners
                .aiMessage(ai)
                .modelName(modelName)
                .tokenUsage(new TokenUsage(promptTokens, result.completionTokens()))
                .finishReason(
                        stoppedBySequence // a stop-sequence cut IS a stop, not an abort
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
