package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.OpenAiMaps;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyParts;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.media.AudioCodec;
import com.qxotic.jinfer.media.ImageCodec;
import com.qxotic.jinfer.media.VideoCodec;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.core.io.Resource;

/**
 * The mapping seam between Spring AI's message/tool model and jinfer's. Two output shapes per
 * conversation: typed {@link Message}s for the native codec, and OpenAI-shaped maps for the Jinja
 * whole-render fallback (templates read arbitrary fields, so the fallback speaks raw maps).
 */
final class JinferMappings {

    private JinferMappings() {}

    // ---- Spring AI -> jinfer (typed, for the native codec) ----

    static List<Message> toMessages(List<org.springframework.ai.chat.messages.Message> messages) {
        List<Message> out = new ArrayList<>(messages.size());
        for (org.springframework.ai.chat.messages.Message m : messages) {
            switch (m) {
                case SystemMessage s -> out.add(new Message(Role.SYSTEM, s.getText()));
                case UserMessage u -> out.add(new Message(Role.USER, userParts(u)));
                case AssistantMessage ai -> out.add(assistant(ai));
                case ToolResponseMessage r -> {
                    List<Part> results = new ArrayList<>(r.getResponses().size());
                    for (ToolResponseMessage.ToolResponse t : r.getResponses()) {
                        results.add(new Part.ToolResult(t.id(), t.responseData()));
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

    /** User content, media included: text stays text, image/audio/video decode to media parts. */
    private static List<Part> userParts(UserMessage u) {
        List<Part> parts = new ArrayList<>();
        if (u.getText() != null && !u.getText().isEmpty()) {
            parts.add(new Part.Text(u.getText(), null));
        }
        for (org.springframework.ai.content.Media media : u.getMedia()) {
            String kind = media.getMimeType().getType();
            switch (kind) {
                case "image" -> parts.add(blob(kind, () -> ImageCodec.decode(bytes(media))));
                case "audio" -> parts.add(blob(kind, () -> AudioCodec.decode(bytes(media))));
                case "video" ->
                        parts.add(blob(kind, () -> VideoCodec.ffmpeg().uniform(localPath(media))));
                default ->
                        throw new UnsupportedOperationException(
                                "media type " + media.getMimeType() + " is not supported");
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

    /** Inline/bytes or a LOCAL file - the library never fetches over the network. */
    private static byte[] bytes(org.springframework.ai.content.Media media) {
        rejectRemote(media.getData());
        return media.getDataAsByteArray();
    }

    private static java.nio.file.Path localPath(org.springframework.ai.content.Media media) {
        Object data = rejectRemote(media.getData());
        if (data instanceof Resource r) {
            try {
                return r.getFile().toPath();
            } catch (IOException e) {
                throw new UnsupportedOperationException(
                        "video needs a local file Resource, got " + r, e);
            }
        }
        if (data instanceof URI u && "file".equals(u.getScheme())) {
            return java.nio.file.Path.of(u);
        }
        if (data instanceof String s && s.startsWith("file:")) {
            return java.nio.file.Path.of(URI.create(s));
        }
        throw new UnsupportedOperationException(
                "video needs a local file Resource or file:// URI, got " + data);
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

    /** Metadata key carrying a reply's reasoning (Ollama convention); replayed on history. */
    static final String THINKING_KEY = "thinking";

    /**
     * Metadata key carrying the parsed reply {@link Message} whole. The round-trip law: an
     * UNMODIFIED assistant echo re-encodes to the exact tokens the model generated (verbatim ids),
     * which is what makes {@code cachedSessions} extension hits deterministic; an edited echo fails
     * the equality check and re-renders faithfully instead.
     */
    static final String REPLY_KEY = "jinfer.reply";

    private static Message assistant(AssistantMessage ai) {
        // an UNMODIFIED echo of a reply this provider produced restores the parsed Message with
        // its verbatim ids (byte-exact re-encode: what makes cachedSessions extension hits
        // deterministic); an edited echo (or one from another provider) re-renders faithfully
        if (ai.getMetadata().get(REPLY_KEY) instanceof Message reply
                && toAssistantMessage(reply).equals(ai)) {
            return reply;
        }
        List<Part> parts = new ArrayList<>();
        // replay the reasoning the model produced earlier (stored by toAssistantMessage)
        if (ai.getMetadata().get(THINKING_KEY) instanceof String thinking && !thinking.isEmpty()) {
            parts.add(new Part.Reasoning(List.of(new Part.Text(thinking, null)), null));
        }
        if (ai.getText() != null && !ai.getText().isEmpty()) {
            parts.add(new Part.Text(ai.getText(), null));
        }
        for (AssistantMessage.ToolCall c : ai.getToolCalls()) {
            parts.add(
                    new Part.ToolCall(
                            c.id() == null ? "" : c.id(),
                            c.name(),
                            OpenAiMaps.args(c.arguments()),
                            null));
        }
        return new Message(Role.ASSISTANT, parts);
    }

    static List<Tool> toTools(List<ToolCallback> callbacks) {
        List<Tool> out = new ArrayList<>(callbacks.size());
        for (ToolCallback cb : callbacks) {
            ToolDefinition def = cb.getToolDefinition();
            out.add(new Tool(def.name(), ToolCallSyntax.jinjaJson(toolMap(def))));
        }
        return out;
    }

    // ---- Spring AI -> OpenAI-shaped maps (for the Jinja whole-render fallback) ----

    static List<Object> toMessageMaps(List<org.springframework.ai.chat.messages.Message> messages) {
        List<Object> out = new ArrayList<>(messages.size());
        for (org.springframework.ai.chat.messages.Message m : messages) {
            var map = new LinkedHashMap<String, Object>();
            switch (m) {
                case SystemMessage s -> {
                    map.put("role", "system");
                    map.put("content", s.getText());
                }
                case UserMessage u -> {
                    map.put("role", "user");
                    map.put("content", u.getText() == null ? "" : u.getText());
                }
                case AssistantMessage ai -> {
                    map.put("role", "assistant");
                    map.put("content", ai.getText() == null ? "" : ai.getText());
                    if (ai.hasToolCalls()) {
                        List<Object> calls = new ArrayList<>();
                        for (AssistantMessage.ToolCall c : ai.getToolCalls()) {
                            calls.add(OpenAiMaps.toolCall(c.id(), c.name(), c.arguments()));
                        }
                        map.put("tool_calls", calls);
                    }
                }
                case ToolResponseMessage r -> {
                    // one OpenAI tool message per response
                    for (ToolResponseMessage.ToolResponse t : r.getResponses()) {
                        out.add(OpenAiMaps.toolResponse(t.responseData(), t.id(), t.name()));
                    }
                    continue;
                }
                default ->
                        throw new IllegalArgumentException(
                                "message type " + m.getMessageType() + " is not supported");
            }
            out.add(map);
        }
        return out;
    }

    static List<Object> toToolMaps(List<ToolCallback> callbacks) {
        List<Object> out = new ArrayList<>(callbacks.size());
        for (ToolCallback cb : callbacks) out.add(toolMap(cb.getToolDefinition()));
        return out;
    }

    private static Map<String, Object> toolMap(ToolDefinition def) {
        return OpenAiMaps.tool(
                def.name(),
                def.description(),
                def.inputSchema() != null ? JsonCodec.parse(def.inputSchema()) : null);
    }

    // ---- jinfer reply -> Spring AI ----

    static AssistantMessage toAssistantMessage(Message reply) {
        ReplyParts parts = ReplyParts.of(reply);
        List<AssistantMessage.ToolCall> calls = new ArrayList<>(parts.toolCalls().size());
        for (ReplyParts.ToolCall c : parts.toolCalls()) {
            calls.add(
                    new AssistantMessage.ToolCall(c.id(), "function", c.name(), c.argumentsJson()));
        }
        AssistantMessage.Builder<?> b =
                AssistantMessage.builder().content(parts.text()).toolCalls(calls);
        // reasoning survives in metadata (Spring AI's AssistantMessage has no thinking slot) and
        // is replayed into the next request's assistant turn - the Ollama/OpenAI convention.
        // Blank-only reasoning (a prompt-opened span's scaffold newlines, e.g. the closed empty
        // pair when thinking is off) is no reasoning at all. The parsed reply rides along whole,
        // so an unmodified echo restores verbatim ids instead of re-tokenizing.
        Map<String, Object> properties = new java.util.HashMap<>();
        properties.put(REPLY_KEY, reply);
        if (!parts.thinking().isBlank()) {
            properties.put(THINKING_KEY, parts.thinking());
        }
        b.properties(properties);
        return b.build();
    }
}
