package com.qxotic.jinfer.spring.ai;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.x.boundary.Media;
import com.qxotic.jinfer.x.boundary.media.AudioCodec;
import com.qxotic.jinfer.x.boundary.media.ImageCodec;
import com.qxotic.jinfer.x.boundary.media.VideoSampler;
import com.qxotic.jinfer.x.chat.Content;
import com.qxotic.jinfer.x.chat.Message;
import com.qxotic.jinfer.x.chat.Role;
import com.qxotic.jinfer.x.chat.Tool;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
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
                    List<Content> results = new ArrayList<>(r.getResponses().size());
                    for (ToolResponseMessage.ToolResponse t : r.getResponses()) {
                        results.add(new Content.ToolResult(t.id(), t.responseData()));
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
    private static List<Content> userParts(UserMessage u, VideoSampler videoSampler) {
        List<Content> parts = new ArrayList<>();
        if (u.getText() != null && !u.getText().isEmpty()) {
            parts.add(new Content.Text(u.getText(), null));
        }
        for (org.springframework.ai.content.Media media : u.getMedia()) {
            String kind = media.getMimeType().getType();
            switch (kind) {
                case "image" -> {
                    byte[] src = bytes(media);
                    parts.add(blob(kind, sha256(src), () -> ImageCodec.decode(src)));
                }
                case "audio" -> {
                    byte[] src = bytes(media);
                    parts.add(blob(kind, sha256(src), () -> AudioCodec.decode(src)));
                }
                case "video" -> {
                    Path src = localPath(media);
                    parts.add(blob(kind, sha256(src), () -> videoSampler.sample(src)));
                }
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

    private static Content.Media blob(String kind, byte[] contentKey, MediaDecode decode) {
        try {
            return new Content.Media(decode.decode(), contentKey);
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

    /** Inline/bytes or a LOCAL file - the library never fetches over the network. */
    private static byte[] bytes(org.springframework.ai.content.Media media) {
        rejectRemote(media.getData());
        return media.getDataAsByteArray();
    }

    private static Path localPath(org.springframework.ai.content.Media media) {
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
            return Path.of(u);
        }
        if (data instanceof String s && s.startsWith("file:")) {
            return Path.of(URI.create(s));
        }
        if (data instanceof byte[])
            throw new UnsupportedOperationException(
                    "inline video bytes are not supported: write them to a file and pass a local"
                            + " file Resource or file:// URI");
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
        List<Content> parts = new ArrayList<>();
        // replay the reasoning the model produced earlier (stored by toAssistantMessage)
        if (ai.getMetadata().get(THINKING_KEY) instanceof String thinking && !thinking.isEmpty()) {
            parts.add(new Content.Reasoning(List.of(new Content.Text(thinking, null)), null));
        }
        if (ai.getText() != null && !ai.getText().isEmpty()) {
            parts.add(new Content.Text(ai.getText(), null));
        }
        for (AssistantMessage.ToolCall c : ai.getToolCalls()) {
            parts.add(
                    new Content.ToolCall(
                            c.id() == null ? "" : c.id(), c.name(), args(c.arguments()), null));
        }
        return new Message(Role.ASSISTANT, parts);
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

    static AssistantMessage toAssistantMessage(Message reply) {
        StringBuilder text = new StringBuilder();
        StringBuilder thinking = new StringBuilder();
        List<AssistantMessage.ToolCall> calls = new ArrayList<>();
        collect(reply.content(), text, thinking, calls, false);
        AssistantMessage.Builder<?> b =
                AssistantMessage.builder().content(text.toString()).toolCalls(calls);
        // reasoning survives in metadata (Spring AI's AssistantMessage has no thinking slot) and
        // is replayed into the next request's assistant turn - the Ollama/OpenAI convention.
        // Blank-only reasoning (a prompt-opened span's scaffold newlines, e.g. the closed empty
        // pair when thinking is off) is no reasoning at all. The parsed reply rides along whole,
        // so an unmodified echo restores verbatim ids instead of re-tokenizing.
        Map<String, Object> properties = new HashMap<>();
        properties.put(REPLY_KEY, reply);
        if (!thinking.toString().isBlank()) {
            properties.put(THINKING_KEY, thinking.toString());
        }
        b.properties(properties);
        return b.build();
    }

    /** The reply flattened: content lane, reasoning lane, tool calls with canonical JSON args. */
    private static void collect(
            List<Content> parts,
            StringBuilder text,
            StringBuilder thinking,
            List<AssistantMessage.ToolCall> calls,
            boolean inReasoning) {
        for (Content part : parts) {
            switch (part) {
                case Content.Text t -> (inReasoning ? thinking : text).append(t.text());
                case Content.Reasoning r -> collect(r.content(), text, thinking, calls, true);
                case Content.ToolCall c ->
                        // pythonic syntaxes carry no call ids: mint stable positional ones (what
                        // Ollama's server does); ids never render back into the prompt (the
                        // template's call syntax has no id slot), so echoes stay byte-identical
                        calls.add(
                                new AssistantMessage.ToolCall(
                                        c.id().isEmpty() ? "call_" + calls.size() : c.id(),
                                        "function",
                                        c.name(),
                                        Json.stringify(c.arguments())));
                default -> {} // ToolResult/Media never appear in a generated reply
            }
        }
    }
}
