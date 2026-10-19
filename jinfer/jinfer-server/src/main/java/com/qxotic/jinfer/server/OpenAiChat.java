package com.qxotic.jinfer.server;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TextStops;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.codecs.VideoSampler;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.Sampling;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;

/**
 * The OpenAI chat.completions dialect, request and response, as this server speaks it - for a host
 * that speaks it in-process instead of over HTTP (libjinfer, embedders of the engine). {@link
 * #lower} validates and lowers a parsed request onto {@link ChatEngine.Request} with exactly the
 * server's rules and messages; {@link #completion} is the {@code chat.completion} envelope of a
 * finished pass. The server's own handlers route through the same code.
 */
public final class OpenAiChat {

    /**
     * A lowered request: what the engine runs, the stop strings to truncate the visible text with,
     * and whether reasoning rides inline as {@code <think>} text ({@code reasoning_format: none}).
     */
    public record Lowered(
            ChatEngine.Request request, List<String> stops, boolean inlineReasoning) {}

    /**
     * Validates and lowers one chat.completions request. {@code servedModel} is the name a {@code
     * model} field must match when present; {@code sampling} is the host's resolved defaults, the
     * engine's own when it has no opinion. Throws {@link IllegalArgumentException} (or the server's
     * {@code InvalidParam}) with the server's message when the request is not acceptable.
     */
    public static Lowered lower(
            Map<String, Object> request,
            String servedModel,
            ServerConfig config,
            Sampling sampling) {
        Validation.validateChatRequest(request);
        Validation.validateChatOptions(request);
        Validation.validateGenerationParams(request, servedModel, config);
        return lower(
                request, Values.asArray(request.get("messages"), "messages"), config, sampling);
    }

    /** The lowering alone, over messages the server already validated. */
    static Lowered lower(
            Map<String, Object> request,
            List<Object> messages,
            ServerConfig config,
            Sampling defaults) {
        List<Tool> tools = ToolUse.offered(request) ? tools(request) : List.of();
        List<Message> prompt = messages(messages);
        Map<String, Object> schema = schema(request);
        if (schema != null) describeSchema(prompt, schema);
        String forcedTool = ToolUse.forced(request);
        ChatEngine.Request lowered =
                new ChatEngine.Request(
                        prompt,
                        tools,
                        thinking(request, config.defaults().think()) && forcedTool == null,
                        maxTokens(request, config),
                        maxReasoningTokensOrDefault(request, config),
                        reasoningCutoffMessageOrDefault(request, config),
                        config.limits().requestTimeout(),
                        sampling(request, defaults),
                        grammar(request),
                        forced(request),
                        stops(request.get("stop")),
                        templateKwargs(request));
        boolean inline = "none".equals(Values.stringValue(request.get("reasoning_format"), null));
        return new Lowered(lowered, stops(request.get("stop")), inline);
    }

    /** The {@code model} the response echoes: the request's, else the served name. */
    public static String modelId(Map<String, Object> request, String servedModel) {
        return Requests.modelId(request, servedModel);
    }

    /** A fresh {@code chatcmpl-} id, the server's own scheme. */
    public static String newId() {
        return "chatcmpl-" + Long.toUnsignedString(System.nanoTime(), 36);
    }

    /**
     * The {@code chat.completion} envelope of a finished pass as JSON text, {@code usage} and
     * {@code timings} included - the bytes the server would send.
     */
    public static String completion(
            String id, String modelId, ChatEngine.Completion completion, Lowered lowered) {
        return completion(id, modelId, completion, lowered, null);
    }

    /**
     * As {@link #completion(String, String, ChatEngine.Completion, Lowered)}, with the words a
     * cancelled pass had produced: the engine drops a cancelled reply, a streaming caller has seen
     * it, and the record of the pass should say what was said. Null when there is nothing to add.
     */
    public static String completion(
            String id,
            String modelId,
            ChatEngine.Completion completion,
            Lowered lowered,
            Message produced) {
        return JsonCodec.stringify(
                OpenAiSchema.chatCompletionResponse(
                        id,
                        modelId,
                        reply(completion, produced, lowered.stops(), lowered.inlineReasoning())));
    }

    static Reply reply(
            ChatEngine.Completion completion, List<String> stops, boolean inlineReasoning) {
        return reply(completion, null, stops, inlineReasoning);
    }

    static Reply reply(
            ChatEngine.Completion completion,
            Message produced,
            List<String> stops,
            boolean inlineReasoning) {
        Message message = completion.cancelled() ? produced : completion.reply();
        List<Content.ToolCall> calls = new ArrayList<>();
        StringBuilder text = new StringBuilder();
        StringBuilder reasoning = new StringBuilder();
        if (message != null) collect(message.content(), text, reasoning, calls);
        String content =
                inlineReasoning && !reasoning.isEmpty()
                        ? "<think>" + reasoning + "</think>" + text
                        : text.toString();
        TextStops.Result visible = TextStops.apply(content, stops);
        // a pass cancelled during prefill has no generator result: nothing was produced, no time
        // was spent decoding
        Generator.GenerationResult result =
                completion.result() != null
                        ? completion.result()
                        : new Generator.GenerationResult(
                                new int[0],
                                OptionalInt.empty(),
                                Generator.FinishReason.ABORT,
                                Duration.ZERO,
                                Duration.ZERO);
        boolean stopped =
                result.stopToken().isPresent() || completion.stopped() || visible.stopped();
        String finish =
                completion.cancelled()
                        ? "cancelled"
                        : finishReason(result.finishReason(), !calls.isEmpty(), stopped);
        return new Reply(
                result,
                completion.promptTokens(),
                completion.restoredTokens(),
                visible.text(),
                inlineReasoning || reasoning.isEmpty() ? null : reasoning.toString(),
                calls,
                finish,
                completion.speculation());
    }

    /**
     * OpenAI's {@code finish_reason}. A deadline or a cancel is "other": neither the model nor the
     * token budget ended the reply, and "stop" would claim the model did - the providers say OTHER.
     */
    static String finishReason(Generator.FinishReason reason, boolean hasCalls, boolean stopped) {
        if (hasCalls) return "tool_calls";
        if (stopped) return "stop";
        return switch (reason) {
            case STOP -> "stop";
            case LENGTH -> "length";
            default -> "other";
        };
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

    // ---- messages ----

    static List<Message> messages(List<Object> values) {
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

    private static List<Content> content(Object value) {
        if (!(value instanceof List<?> parts)) {
            return new ArrayList<>(List.of(new Content.Text(Values.stringValue(value, ""))));
        }
        List<Content> out = new ArrayList<>();
        for (Object raw : parts) {
            Map<String, Object> part = Values.asObject(raw, "content part");
            String modality = MediaParts.modality(part);
            if (modality == null)
                throw new IllegalArgumentException(
                        "unsupported content part type: " + part.get("type"));
            out.add(
                    switch (modality) {
                        case "text" -> new Content.Text(MediaParts.text(part));
                        case "image" -> MediaParts.image(part);
                        case "audio" -> MediaParts.audio(part);
                        default -> MediaParts.video(part, VideoSampler.UNIFORM);
                    });
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

    static Map<String, Object> parseArguments(String json) {
        if (json.isBlank()) return Map.of();
        Object parsed;
        try {
            parsed = JsonCodec.parse(json);
        } catch (Json.ParseException e) {
            throw new IllegalArgumentException(
                    "tool_calls[].function.arguments is not valid JSON: " + e.getMessage());
        }
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

    // ---- knobs ----

    static Sampling sampling(Map<String, Object> request, Sampling defaults) {
        Long seed = defaults.seed();
        if (request.get("seed") != null) seed = Values.longValue(request.get("seed"), "seed", 0);
        return defaults.override(
                number(request, "temperature"),
                number(request, "top_p"),
                integer(request, "top_k"),
                number(request, "min_p"),
                seed);
    }

    private static Float number(Map<String, Object> request, String field) {
        Object value = request.get(field);
        return value == null ? null : Values.floatValue(value, field, 0);
    }

    private static Integer integer(Map<String, Object> request, String field) {
        Object value = request.get(field);
        return value == null ? null : Values.intValue(value, field, 0);
    }

    static int maxTokens(Map<String, Object> request, ServerConfig config) {
        return Values.intValue(
                Requests.budget(request),
                Requests.budgetField(request),
                config.defaults().maxOutputTokens());
    }

    static Integer reasoningMax(Map<String, Object> request) {
        return request.get("max_reasoning_tokens") == null
                ? null
                : Values.intValue(request.get("max_reasoning_tokens"), "max_reasoning_tokens", -1);
    }

    /**
     * What the model "decides" when the think-span cap fires (llama.cpp's {@code
     * --reasoning-cutoff-message}); absent = a bare paragraph break.
     */
    static String reasoningCutoffMessage(Map<String, Object> request) {
        return Values.stringValue(request.get("reasoning_cutoff_message"), null);
    }

    /** The request's value, else the server's configured default - as {@code think} does. */
    private static Integer maxReasoningTokensOrDefault(
            Map<String, Object> request, ServerConfig config) {
        Integer value = reasoningMax(request);
        return value != null ? value : config.defaults().maxReasoningTokens();
    }

    private static String reasoningCutoffMessageOrDefault(
            Map<String, Object> request, ServerConfig config) {
        String value = reasoningCutoffMessage(request);
        return value != null ? value : config.defaults().reasoningCutoffMessage();
    }

    /**
     * {@code chat_template_kwargs.enable_thinking} first, else the OpenAI spelling: {@code
     * reasoning_effort} "none" turns reasoning off and any level turns it on (the levels themselves
     * have no dial here, as in llama.cpp), else the server default.
     */
    static boolean thinking(Map<String, Object> request, boolean fallback) {
        if (request.get("chat_template_kwargs") instanceof Map<?, ?> kwargs
                && kwargs.get("enable_thinking") instanceof Boolean value) return value;
        if (request.get("reasoning_effort") instanceof String effort) return !effort.equals("none");
        return fallback;
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

    static String grammar(Map<String, Object> request) {
        if (request.get("grammar") instanceof String source && !source.isBlank()) return source;
        if (request.get("response_format") instanceof Map<?, ?> format) {
            if ("json_object".equals(format.get("type"))) return Grammar.jsonGbnf();
        }
        Map<String, Object> schema = schema(request);
        return schema == null ? null : Grammar.schemaGbnf(schema);
    }

    private static Map<String, Object> schema(Map<String, Object> request) {
        if (!(request.get("response_format") instanceof Map<?, ?> format)
                || !"json_schema".equals(format.get("type"))) return null;
        Map<String, Object> wrapper =
                Values.asObject(format.get("json_schema"), "response_format.json_schema");
        return Values.asObject(wrapper.get("schema"), "response_format.json_schema.schema");
    }

    /** The grammar enforces syntax; the prompt tells the model what to produce. */
    static void describeSchema(List<Message> messages, Map<String, Object> schema) {
        String instruction =
                "Your final answer must be JSON matching this schema: "
                        + JsonCodec.stringify(schema);
        for (int i = 0; i < messages.size(); i++) {
            Message message = messages.get(i);
            if (!Role.SYSTEM.equals(message.role())) continue;
            List<Content> content = new ArrayList<>(message.content());
            content.add(new Content.Text("\n\n" + instruction));
            messages.set(i, new Message(Role.SYSTEM, content));
            return;
        }
        messages.addFirst(Message.system(instruction));
    }

    static List<String> stops(Object value) {
        if (value == null) return List.of();
        if (value instanceof String stop) return stop.isEmpty() ? List.of() : List.of(stop);
        List<String> out = new ArrayList<>();
        for (Object item : Values.asArray(value, "stop")) {
            String stop = Values.stringValue(item, "");
            if (!stop.isEmpty()) out.add(stop);
        }
        return List.copyOf(out);
    }

    private OpenAiChat() {}
}
