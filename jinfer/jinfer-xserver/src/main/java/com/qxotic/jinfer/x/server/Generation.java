package com.qxotic.jinfer.x.server;

import com.qxotic.jinfer.x.boundary.ContentKey;
import com.qxotic.jinfer.x.boundary.Media;
import com.qxotic.jinfer.x.boundary.media.AudioCodec;
import com.qxotic.jinfer.x.boundary.media.ImageCodec;
import com.qxotic.jinfer.x.boundary.media.VideoSampler;
import com.qxotic.jinfer.x.cache.PromptCache;
import com.qxotic.jinfer.x.chat.Channel;
import com.qxotic.jinfer.x.chat.ChatEngine;
import com.qxotic.jinfer.x.chat.Content;
import com.qxotic.jinfer.x.chat.MediaEncodingCache;
import com.qxotic.jinfer.x.chat.Message;
import com.qxotic.jinfer.x.chat.Role;
import com.qxotic.jinfer.x.chat.TextStops;
import com.qxotic.jinfer.x.chat.Tool;
import com.qxotic.jinfer.x.llm.Grammar;
import com.qxotic.jinfer.x.llm.Sampling;
import com.qxotic.jinfer.x.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;

/** OpenAI wire values lowered onto the framework-neutral {@link ChatEngine}. */
final class Generation {

    private final ChatEngine engine;
    private final ServerConfig config;
    private final Metrics metrics;
    private final Sampling defaults;

    Generation(ChatEngine engine, ServerConfig config, Metrics metrics) {
        this.engine = engine;
        this.config = config;
        this.metrics = metrics;
        this.defaults =
                config.defaults().sampling() != null
                        ? config.defaults().sampling()
                        : engine.loaded().samplingDefaults().resolve(null, null, null, null, null);
    }

    PromptCache.Sample cacheSample() {
        return engine.cacheSample();
    }

    MediaEncodingCache.Sample mediaCacheSample() {
        return engine.mediaCacheSample();
    }

    boolean blockCaching() {
        return engine.blockCaching();
    }

    Sampling defaults() {
        return defaults;
    }

    String forcedTool(Map<String, Object> request) {
        return ToolUse.forced(request);
    }

    Reply chat(Map<String, Object> request, List<Object> messages, Sinks sinks) {
        List<Tool> tools = ToolUse.offered(request) ? tools(request) : List.of();
        ChatEngine.Request lowered =
                new ChatEngine.Request(
                        messages(messages),
                        tools,
                        thinking(request) && forcedTool(request) == null,
                        maxTokens(request),
                        reasoningMax(request),
                        config.limits().requestTimeout(),
                        sampling(request, defaults),
                        grammar(request),
                        forced(request),
                        stops(request.get("stop")),
                        templateKwargs(request));
        return recorded(run(lowered, request, sinks));
    }

    Reply completion(Map<String, Object> request, String prompt, Sinks sinks) {
        IntSequence ids =
                config.defaults().rawPrompt()
                        ? SpecialTokens.encode(engine.loaded().tokenizer(), prompt)
                        : engine.loaded().tokenizer().encode(prompt);
        try (ChatEngine.Prepared prepared =
                engine.prepareRaw(
                        ids.toArray(),
                        sampling(request, defaults),
                        maxTokens(request),
                        config.limits().requestTimeout(),
                        grammar(request),
                        stops(request.get("stop")))) {
            WireSink sink = new WireSink(request, sinks);
            ChatEngine.Completion completion = engine.complete(prepared, sink);
            sink.finish();
            return recorded(finish(completion, prepared.stops(), sink.inline));
        }
    }

    private Reply run(ChatEngine.Request request, Map<String, Object> wire, Sinks sinks) {
        try (ChatEngine.Prepared prepared = engine.prepare(request)) {
            WireSink sink = new WireSink(wire, sinks);
            ChatEngine.Completion completion = engine.complete(prepared, sink);
            sink.finish();
            return finish(completion, prepared.stops(), sink.inline);
        }
    }

    private static final class WireSink implements ChatEngine.ReplySink {
        private final Sinks sinks;
        private final boolean inline;
        private boolean reasoning;

        WireSink(Map<String, Object> request, Sinks sinks) {
            this.sinks = sinks;
            inline = "none".equals(Values.stringValue(request.get("reasoning_format"), null));
        }

        @Override
        public void on(ChatEngine.Delta delta) {
            if (delta.channel() == Channel.REASONING) {
                if (inline) {
                    if (!reasoning && sinks.onText() != null) sinks.onText().accept("<think>");
                    reasoning = true;
                    if (sinks.onText() != null) sinks.onText().accept(delta.text());
                } else if (sinks.onReasoning() != null) {
                    sinks.onReasoning().accept(delta.text());
                }
            } else if (delta.channel() == Channel.CONTENT && sinks.onText() != null) {
                if (reasoning) {
                    sinks.onText().accept("</think>");
                    reasoning = false;
                }
                sinks.onText().accept(delta.text());
            }
        }

        void finish() {
            if (inline && reasoning && sinks.onText() != null) sinks.onText().accept("</think>");
        }
    }

    private Reply finish(
            ChatEngine.Completion completion, List<String> stops, boolean inlineReasoning) {
        Message message = completion.reply();
        List<Content.ToolCall> calls = new ArrayList<>();
        StringBuilder text = new StringBuilder();
        StringBuilder reasoning = new StringBuilder();
        if (message != null) collect(message.content(), text, reasoning, calls);
        String content =
                inlineReasoning && !reasoning.isEmpty()
                        ? "<think>" + reasoning + "</think>" + text
                        : text.toString();
        TextStops.Result visible = TextStops.apply(content, stops);
        String finish =
                !calls.isEmpty()
                        ? "tool_calls"
                        : completion.result().stopToken().isPresent()
                                        || completion.stopped()
                                        || visible.stopped()
                                ? "stop"
                                : switch (completion.result().finishReason()) {
                                    case LENGTH -> "length";
                                    default -> "stop";
                                };
        metrics.recordPromptCache(completion.tier(), completion.restoredTokens());
        return new Reply(
                completion.result(),
                completion.promptTokens(),
                completion.restoredTokens(),
                visible.text(),
                inlineReasoning || reasoning.isEmpty() ? null : reasoning.toString(),
                calls,
                finish,
                completion.speculation());
    }

    private static void collect(
            List<Content> content,
            StringBuilder text,
            StringBuilder reasoning,
            List<Content.ToolCall> calls) {
        for (Content part : content) {
            switch (part) {
                case Content.Text value -> text.append(value.text());
                case Content.Reasoning value -> {
                    reasoning.append(value.text());
                    collectCalls(value.content(), calls);
                }
                case Content.ToolCall value -> calls.add(value);
                default -> {}
            }
        }
    }

    private static void collectCalls(List<Content> content, List<Content.ToolCall> calls) {
        for (Content part : content) {
            if (part instanceof Content.ToolCall call) calls.add(call);
            else if (part instanceof Content.Reasoning nested)
                collectCalls(nested.content(), calls);
        }
    }

    private Reply recorded(Reply reply) {
        metrics.record(reply);
        return reply;
    }

    private List<Message> messages(List<Object> values) {
        List<Message> out = new ArrayList<>(values.size());
        for (Object value : values) {
            Map<String, Object> message = Values.asObject(value, "message");
            String role = Values.stringValue(message.get("role"), "user");
            if ("tool".equals(role)) {
                out.add(
                        new Message(
                                Role.TOOL,
                                List.of(
                                        new Content.ToolResult(
                                                Values.stringValue(message.get("tool_call_id"), ""),
                                                Values.messageContent(message.get("content"))))));
                continue;
            }
            List<Content> content = content(message.get("content"));
            if ("assistant".equals(role)) {
                String thought = Values.stringValue(message.get("reasoning_content"), null);
                if (thought == null) thought = Values.stringValue(message.get("reasoning"), null);
                if (thought != null && !thought.isEmpty()) {
                    content.addFirst(
                            new Content.Reasoning(List.of(new Content.Text(thought)), null));
                }
                content.addAll(toolCalls(message.get("tool_calls")));
            }
            out.add(new Message(new Role(role), content));
        }
        return out;
    }

    private List<Content> content(Object value) {
        if (!(value instanceof List<?> parts)) {
            return new ArrayList<>(List.of(new Content.Text(Values.stringValue(value, ""))));
        }
        List<Content> out = new ArrayList<>();
        for (Object raw : parts) {
            Map<String, Object> part = Values.asObject(raw, "content part");
            switch (Values.stringValue(part.get("type"), "")) {
                case "text", "input_text" ->
                        out.add(
                                new Content.Text(
                                        Values.stringValue(
                                                part.get("text") != null
                                                        ? part.get("text")
                                                        : part.get("input_text"),
                                                "")));
                case "image_url", "input_image" -> out.add(image(part));
                case "input_audio" -> out.add(audio(part));
                case "video_url" -> out.add(video(part));
                default ->
                        throw new IllegalArgumentException(
                                "unsupported content part type: " + part.get("type"));
            }
        }
        return out;
    }

    private static List<Content> toolCalls(Object value) {
        if (value == null) return List.of();
        List<Content> out = new ArrayList<>();
        for (Object raw : Values.asArray(value, "tool_calls")) {
            Map<String, Object> call = Values.asObject(raw, "tool call");
            Map<String, Object> function = Values.asObject(call.get("function"), "function");
            Object arguments = function.get("arguments");
            Map<String, Object> args =
                    arguments instanceof Map<?, ?> map
                            ? castMap(map)
                            : parseArguments(Values.stringValue(arguments, ""));
            out.add(
                    new Content.ToolCall(
                            Values.stringValue(call.get("id"), ""),
                            Values.stringValue(function.get("name"), ""),
                            args));
        }
        return out;
    }

    private static Map<String, Object> parseArguments(String json) {
        if (json.isBlank()) return Map.of();
        Object parsed = JsonCodec.parse(json);
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IllegalArgumentException("tool arguments must be a JSON object");
        }
        return castMap(map);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> castMap(Map<?, ?> map) {
        return (Map<String, Object>) map;
    }

    private static List<Tool> tools(Map<String, Object> request) {
        List<Tool> out = new ArrayList<>();
        for (Object raw : Values.asArray(request.get("tools"), "tools")) {
            Map<String, Object> definition = Values.asObject(raw, "tool");
            Map<String, Object> function = Values.asObject(definition.get("function"), "function");
            String name = Values.stringValue(function.get("name"), "");
            if (!name.isEmpty()) out.add(new Tool(name, definition));
        }
        return out;
    }

    private Content.Media image(Map<String, Object> part) {
        Object image = part.get("image_url") != null ? part.get("image_url") : part.get("image");
        byte[] bytes = dataUri(url(image), "image_url");
        try {
            return new Content.Media(ImageCodec.decode(bytes), ContentKey.sha256(bytes));
        } catch (IOException failure) {
            throw new IllegalArgumentException(
                    "image could not be decoded: " + failure.getMessage());
        }
    }

    private Content.Media audio(Map<String, Object> part) {
        Map<String, Object> audio = Values.asObject(part.get("input_audio"), "input_audio");
        byte[] bytes;
        try {
            bytes = Base64.getDecoder().decode(Values.stringValue(audio.get("data"), ""));
        } catch (IllegalArgumentException failure) {
            throw new IllegalArgumentException("input_audio data is not valid base64");
        }
        try {
            return new Content.Media(AudioCodec.decode(bytes), ContentKey.sha256(bytes));
        } catch (IOException failure) {
            throw new IllegalArgumentException(
                    "audio could not be decoded: " + failure.getMessage());
        }
    }

    private Content.Media video(Map<String, Object> part) {
        MediaSpill.Spilled spilled = null;
        try {
            spilled = MediaSpill.base64Video(url(part.get("video_url")), "video_url");
            Media.Video video = VideoSampler.UNIFORM.sample(spilled.file());
            return new Content.Media(video, spilled.key());
        } catch (IOException failure) {
            throw new IllegalArgumentException(
                    "video could not be decoded: " + failure.getMessage());
        } finally {
            if (spilled != null) {
                MediaSpill.deleteQuietly(spilled.file());
            }
        }
    }

    private static String url(Object value) {
        return value instanceof Map<?, ?> map
                ? Values.stringValue(map.get("url"), "")
                : Values.stringValue(value, "");
    }

    private static byte[] dataUri(String value, String field) {
        Validation.require(
                value.startsWith("data:"),
                "%s must be a data: URI (the server does not fetch remote URLs)",
                field);
        int comma = value.indexOf(',');
        Validation.require(
                comma > 0 && value.substring(0, comma).endsWith(";base64"),
                "%s data: URI must be base64-encoded",
                field);
        try {
            return Base64.getDecoder().decode(value.substring(comma + 1));
        } catch (IllegalArgumentException failure) {
            throw new IllegalArgumentException(field + " base64 payload is malformed");
        }
    }

    static Sampling sampling(Map<String, Object> request, Sampling defaults) {
        Long seed = defaults.seed();
        if (request.get("seed") != null) seed = Values.longValue(request.get("seed"), 0);
        return defaults.override(
                number(request.get("temperature")),
                number(request.get("top_p")),
                integer(request.get("top_k")),
                number(request.get("min_p")),
                seed);
    }

    private static Float number(Object value) {
        return value == null ? null : Values.floatValue(value, 0);
    }

    private static Integer integer(Object value) {
        return value == null ? null : Values.intValue(value, 0);
    }

    private int maxTokens(Map<String, Object> request) {
        return Values.intValue(Requests.budget(request), config.defaults().maxOutputTokens());
    }

    private static Integer reasoningMax(Map<String, Object> request) {
        return request.get("reasoning_max_tokens") == null
                ? null
                : Values.intValue(request.get("reasoning_max_tokens"), -1);
    }

    private boolean thinking(Map<String, Object> request) {
        if (request.get("chat_template_kwargs") instanceof Map<?, ?> kwargs
                && kwargs.get("enable_thinking") instanceof Boolean value) return value;
        return config.defaults().think();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> templateKwargs(Map<String, Object> request) {
        return request.get("chat_template_kwargs") instanceof Map<?, ?> map
                ? (Map<String, Object>) map
                : null;
    }

    private static ChatEngine.ForcedTool forced(Map<String, Object> request) {
        String name = ToolUse.forced(request);
        if (name == null) return ChatEngine.ForcedTool.NONE;
        return name.isEmpty() ? ChatEngine.ForcedTool.ANY : new ChatEngine.ForcedTool.Named(name);
    }

    private static String grammar(Map<String, Object> request) {
        if (request.get("grammar") instanceof String source && !source.isBlank()) return source;
        if (request.get("response_format") instanceof Map<?, ?> format) {
            if ("json_object".equals(format.get("type"))) return Grammar.jsonGbnf();
            if ("json_schema".equals(format.get("type"))) {
                Map<String, Object> wrapper =
                        Values.asObject(format.get("json_schema"), "response_format.json_schema");
                return Grammar.schemaGbnf(
                        Values.asObject(
                                wrapper.get("schema"), "response_format.json_schema.schema"));
            }
        }
        return null;
    }

    private static List<String> stops(Object value) {
        if (value == null) return List.of();
        if (value instanceof String stop) return stop.isEmpty() ? List.of() : List.of(stop);
        List<String> out = new ArrayList<>();
        for (Object item : Values.asArray(value, "stop")) {
            String stop = Values.stringValue(item, "");
            if (!stop.isEmpty()) out.add(stop);
        }
        return List.copyOf(out);
    }
}
