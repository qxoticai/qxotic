package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.cache.StateCodec;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.RequestPolicy;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.Values;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.Generator.GenerationResult;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * The inference service: turns a parsed request into a {@link Reply}, with streaming sinks for the
 * live channels. Owns the model and drives generation through the tokens-only {@link Generator}:
 * sampler / grammar / think wiring, stop conditions, tool-call seeding/parsing. The reply token
 * stream is structured by the model's {@link ReplyParser} (text channels + structure); transport -
 * endpoint handlers and SSE streaming - lives in {@link Server}.
 *
 * <p>Plain chat on a model with a {@link ChatTemplate} is lowered through the hand-written codec
 * (injection-inert, oracle-validated framing); when the model also has a {@link StateCodec} and
 * caching is enabled, the conversation is served through a per-model {@link PromptCache}, so a
 * follow-up request that echoes the prior turns resumes their KV instead of re-prefilling. Requests
 * with tools the template cannot frame, and models without a template, take the whole-render
 * fallback.
 */
final class Generation {

    private final LoadedModel<?> model;
    private final LLMOptions options;
    private final String servedModel;
    private final ChatTemplate template; // memoized model framing, null when the model has none
    // raw-prompt mode encodes text special-token-aware; nothing renders a template for it
    private final com.qxotic.toknroll.Specials specials;
    // The shared runtime: template stack, sampling policy, block tree and session pool. This
    // class predates all of it and had its own of each; what remains here is the OpenAI wire.
    private final ChatEngine engine;
    private final Metrics metrics;

    Generation(LoadedModel<?> chatModel, LLMOptions options, Metrics metrics) {
        this.metrics = metrics;
        this.servedModel = options.modelPath().getFileName().toString();
        this.model = chatModel;
        this.options = options;
        this.template = chatModel.template().orElse(null);
        this.specials = SpecialTokens.encoder(model.tokenizer());
        // borrowed weights: the server loaded the model and keeps its arena
        this.engine = new ChatEngine(chatModel, servedModel, null, RuntimeFlags.SESSIONS);
    }

    /** Frees the engine's states and blocks; the weights arena stays the server's. */
    void close() {
        engine.close();
    }

    // ---- chat / completion -------------------------------------------------

    @SuppressWarnings("unchecked")
    /**
     * Every served request records here, whichever path produced it. Deliberately at the ENTRY and
     * not down in the generation pass: a path that bypasses that pass stops incrementing /metrics
     * silently, which is exactly what a migration is likely to do.
     */
    Reply chat(Map<String, Object> request, List<Object> messages, Sinks sinks) {
        return recorded(chat0(request, messages, sinks));
    }

    private Reply recorded(Reply reply) {
        metrics.record(reply);
        return reply;
    }

    private Reply chat0(Map<String, Object> request, List<Object> messages, Sinks sinks) {
        boolean tools = ToolUse.offered(request);
        List<Message> turns = toConversation(messages);
        Map<String, Object> kwargs =
                request.get("chat_template_kwargs") instanceof Map<?, ?> kw
                        ? (Map<String, Object>) kw
                        : null;
        // ONE lowering for both framings. prepare() tries the model's own codec and falls back to
        // the scrubbed whole-render itself - the decision this class used to make by hand, with a
        // second prompt build behind it. Forced calls, the think floor, the grammar and the parser
        // seed all ride the shared recipe.
        ChatEngine.Request lowered =
                new ChatEngine.Request(
                        turns,
                        tools ? buildTools(request) : List.of(),
                        requestThink(request),
                        maxTokens(request),
                        ServerFlags.SERVER_REQUEST_TIMEOUT_NANOS,
                        Values.floatValue(request.get("temperature"), options.temperature()),
                        Values.floatValue(request.get("top_p"), options.topp()),
                        Values.longValue(request.get("seed"), options.seed()),
                        grammarSpec(request),
                        ToolUse.forced(request) != null && nativeForcedOk(request),
                        false,
                        textStops(request.get("stop")),
                        kwargs);
        ChatEngine.Prepared prepared =
                engine.prepare(
                        lowered,
                        () -> messages,
                        () -> tools ? Values.asArray(request.get("tools"), "tools") : null);
        Reply reply =
                complete(
                        prepared,
                        request,
                        sinks,
                        consumedPromptTokens(model.tokenizer(), prepared.encoded().prompt()));
        // Bare-call recovery (llama.cpp #21242): LFM2.5 sometimes emits pythonic calls WITHOUT its
        // markers, so the structural parser finds nothing; the string scan is a no-op otherwise.
        return tools ? ToolUse.parse(model, reply, request) : reply;
    }

    /**
     * Drive one prepared request through the engine and shape the reply the OpenAI schema wants.
     * Both entry points land here: chat once the template stack has lowered it, completions once a
     * raw prompt has lowered itself.
     */
    private Reply complete(
            ChatEngine.Prepared prepared, Map<String, Object> request, Sinks sinks, int billed) {
        // stops belong to the engine: it holds back a could-still-be-a-stop suffix and ends the
        // pass, so the router only accumulates and projects inline reasoning
        FragmentRouter router =
                new FragmentRouter(
                        List.of(), sinks.onText(), sinks.onReasoning(), inlineReasoning(request));
        ChatEngine.Completion done =
                engine.complete(
                        prepared,
                        new ChatEngine.ReplySink() {
                            @Override
                            public void content(String delta) {
                                router.fragment(delta, false);
                            }

                            @Override
                            public void thinking(String delta) {
                                router.fragment(delta, true);
                            }

                            @Override
                            public boolean cancelled() {
                                return false; // the worker owns the request; SSE has its watchdog
                            }
                        });
        router.flush();
        OpenAiSchema.Usage usage = sinks.usage();
        if (usage != null) {
            usage.promptTokens = billed;
            usage.cachedTokens = done.restoredTokens();
            usage.completionTokens = done.result().completionTokens();
        }
        metrics.recordPromptCache(done.tier() == ChatEngine.Tier.SESSION, done.restoredTokens());
        return router.reply(
                done.result(), done.reply(), billed, done.restoredTokens(), prepared.stops());
    }

    /** Forced requests stay native only when the template's forced-call recipe covers them. */
    private boolean nativeForcedOk(Map<String, Object> request) {
        String forced = ToolUse.forced(request);
        if (forced == null) return true;
        // null template = no native codec, so no call seed to force with. The old caller
        // short-circuited on `template != null &&` before reaching here; this method must own it.
        if (template == null || template.callSeed().length == 0) return false;
        return forced.isEmpty() || ToolUse.names(request).contains(forced);
    }

    /**
     * Maps OpenAI messages to {@link Message}s, or null when a message can't be represented on the
     * codec path (tool role / tool_calls / non-text content) - the caller falls back.
     */
    private List<Message> toConversation(List<Object> messages) {
        List<Message> out = new ArrayList<>(messages.size());
        for (Object raw : messages) {
            if (!(raw instanceof Map<?, ?> m)) return null;
            String role = Values.stringValue(m.get("role"), "user");
            if (m.get("function_call") != null) return null; // legacy shape: whole-render
            Object raw2 = m.get("content");
            if (raw2 instanceof List<?> parts) { // multimodal content array: only
                for (Object part : parts) { // pure-text flattens faithfully
                    if (!(part instanceof Map<?, ?> pm) || !"text".equals(pm.get("type")))
                        return null;
                }
            }
            String content = Values.messageContent(raw2);
            List<Part> callParts = toolCallParts(m.get("tool_calls"));
            if (callParts == null) return null; // malformed tool_calls: whole-render
            if (!callParts.isEmpty()) {
                List<Part> all = new ArrayList<>();
                if (!content.isEmpty()) all.add(new Part.Text(content));
                all.addAll(callParts);
                out.add(new Message(new Role(role), all));
            } else {
                out.add(new Message(new Role(role), content));
            }
        }
        return out;
    }

    /**
     * An assistant message's {@code tool_calls} array to {@link Part.ToolCall} parts (empty when
     * absent), or null when the shape is unusable (so the caller falls back to whole-render). Each
     * call's {@code arguments} JSON string is parsed to an ordered map; a non-object leaves an
     * empty argument map.
     */
    @SuppressWarnings("unchecked")
    private static List<Part> toolCallParts(Object toolCalls) {
        if (toolCalls == null) return List.of();
        if (!(toolCalls instanceof List<?> calls)) return null;
        List<Part> parts = new ArrayList<>();
        for (Object c : calls) {
            if (!(c instanceof Map<?, ?> call)) return null;
            Object fn = call.get("function");
            if (!(fn instanceof Map<?, ?> f)) return null;
            String name = Values.stringValue(f.get("name"), null);
            if (name == null) return null;
            Object args = f.get("arguments");
            Map<String, Object> argMap = new LinkedHashMap<>();
            if (args instanceof String s && !s.isBlank()) {
                try {
                    if (JsonCodec.parse(s) instanceof Map<?, ?> parsed)
                        argMap.putAll((Map<String, Object>) parsed);
                } catch (RuntimeException notJson) {
                    /* leave empty */
                }
            } else if (args instanceof Map<?, ?> parsed) {
                argMap.putAll((Map<String, Object>) parsed);
            }
            parts.add(new Part.ToolCall(Values.stringValue(call.get("id"), ""), name, argMap));
        }
        return parts;
    }

    /**
     * The offered tools as {@link Tool}s, each carrying its request JSON canonicalized to the form
     * Jinja {@code tojson} produces (so a template that embeds it stays byte-exact).
     */
    private static List<Tool> buildTools(Map<String, Object> request) {
        List<Tool> out = new ArrayList<>();
        for (Object raw : Values.asArray(request.get("tools"), "tools")) {
            if (!(raw instanceof Map<?, ?> t)) continue;
            String name =
                    t.get("function") instanceof Map<?, ?> fn
                            ? Values.stringValue(fn.get("name"), "")
                            : "";
            if (!name.isEmpty()) out.add(new Tool(name, ToolCallSyntax.jinjaJson(t)));
        }
        return out;
    }

    /**
     * The codec chat path: lower the conversation to the model's own framing and, when the prompt
     * cache is available, resume the longest cached prefix into the state (skipping that prefill)
     * and ingest only the delta, caching it for the next turn. Assistant history is re-tokenized
     * from text - best-effort, but the framing is byte-exact with the model's Jinja template
     * (validated by the per-model oracle), so a stable client echo reuses the whole prefix.
     */
    Reply completion(Map<String, Object> request, String prompt, Sinks sinks) {
        return recorded(completion0(request, prompt, sinks));
    }

    private Reply completion0(Map<String, Object> request, String prompt, Sinks sinks) {
        Tokenizer tokenizer = model.tokenizer();
        IntSequence promptTokens =
                options.rawPrompt() ? specials.encode(tokenizer, prompt) : tokenizer.encode(prompt);
        return generate(request, promptTokens, sinks);
    }

    /** One pass through the tokens-only {@link Generator}: a fresh state prefills the prompt. */
    /**
     * A raw prompt is a conversation that is already encoded, so it skips the template stack and
     * lowers straight to a {@link ChatEngine.Prepared} - the same generation pass, sampling policy,
     * stop handling and telemetry as chat, with no framing of its own.
     */
    private Reply generate(Map<String, Object> request, IntSequence promptTokens, Sinks sinks) {
        int maxTokens = maxTokens(request);
        boolean think = requestThink(request);
        Grammar.Spec grammar = grammarSpec(request);
        Sampler sampler =
                RequestPolicy.sampler(
                        model,
                        Values.floatValue(request.get("temperature"), options.temperature()),
                        Values.floatValue(request.get("top_p"), options.topp()),
                        Values.longValue(request.get("seed"), options.seed()),
                        think,
                        maxTokens,
                        reasoningMax(request));
        if (grammar != null) {
            sampler = RequestPolicy.constrained(model, sampler, grammar.cursor(), think);
        }
        List<Batch> prompt = List.of(Batch.prefill(promptTokens.toArray()));
        ChatEngine.Prepared prepared =
                new ChatEngine.Prepared(
                        new ChatEngine.Encoded(prompt, java.util.Optional.empty()),
                        sampler,
                        maxTokens,
                        ServerFlags.SERVER_REQUEST_TIMEOUT_NANOS,
                        promptTokens.length(),
                        new int[0], // no template, so no reply-grammar tail to pre-feed
                        textStops(request.get("stop")),
                        false,
                        false); // a completion offers no tools, so call syntax stays text
        return complete(
                prepared, request, sinks, consumedPromptTokens(model.tokenizer(), promptTokens));
    }

    /** The server's per-request override of the half-budget reasoning default. */
    private static Integer reasoningMax(Map<String, Object> request) {
        Object rmt = request.get("reasoning_max_tokens");
        return rmt == null ? null : Values.intValue(rmt, -1);
    }

    /** Prompt size as billed to the client: a leading BOS is template overhead, not user input. */
    private static int consumedPromptTokens(Tokenizer tokenizer, IntSequence promptTokens) {
        int bos = SpecialTokens.findFirst(tokenizer, "<bos>", "<|startoftext|>").orElse(1);
        if (!promptTokens.isEmpty() && promptTokens.getFirst() == bos) {
            return promptTokens.length() - 1;
        }
        return promptTokens.length();
    }

    /**
     * Routes the parser's text fragments to the live sinks and accumulates the streamed reply:
     * content through the stop-string holdback, reasoning to its channel (or bracketed inline as
     * {@code <think>...</think>} content for reasoning_format "none"). Structure (tool calls) comes
     * from the parser's finished {@link Message}, not the stream - calls are atomic.
     */
    private static final class FragmentRouter {
        private final StringBuilder text = new StringBuilder();
        private final StringBuilder reasoning = new StringBuilder();
        private final TextStops.Holdback holdback; // null when neither streaming nor text stops
        private final Consumer<String> onReasoning;
        private final InlineThink inline; // null when reasoning routes to its own channel

        FragmentRouter(
                List<String> textStops,
                Consumer<String> onText,
                Consumer<String> onReasoning,
                boolean inline) {
            this.holdback =
                    onText != null || !textStops.isEmpty()
                            ? new TextStops.Holdback(textStops, onText != null ? onText : t -> {})
                            : null;
            this.onReasoning = onReasoning;
            this.inline = inline ? new InlineThink() : null;
        }

        void fragment(String fragment, boolean reasoningChannel) {
            if (reasoningChannel) {
                if (inline != null) {
                    content(inline.project(fragment, true));
                    return;
                }
                reasoning.append(fragment);
                if (onReasoning != null) onReasoning.accept(fragment);
            } else {
                content(inline != null ? inline.project(fragment, false) : fragment);
            }
        }

        private void content(String fragment) {
            text.append(fragment);
            if (holdback != null) holdback.accept(fragment);
        }

        boolean stopped() {
            return holdback != null && holdback.stopped();
        }

        void flush() {
            if (holdback != null) holdback.flush();
        }

        /** The coalesced {@link Reply}, with stop strings applied and finish_reason mapped. */
        Reply reply(
                GenerationResult result,
                Message structured,
                int promptTokens,
                int cachedTokens,
                List<String> textStops) {
            List<Part.ToolCall> toolCalls = collectCalls(structured.content());
            TextStops.Result stopResult = TextStops.apply(text.toString(), textStops);
            boolean textStopped = stopResult.stopped() || stopped();
            String finishReason =
                    !toolCalls.isEmpty()
                            ? "tool_calls"
                            : result.stopToken() >= 0 || textStopped
                                    ? "stop"
                                    : "length".equals(result.finishReason()) ? "length" : "stop";
            return new Reply(
                    result,
                    promptTokens,
                    cachedTokens,
                    stopResult.text(),
                    reasoning.isEmpty() ? null : reasoning.toString(),
                    toolCalls,
                    finishReason);
        }

        /** Every tool call in the reply, in order, including calls made inside a think span. */
        private static List<Part.ToolCall> collectCalls(List<Part> parts) {
            List<Part.ToolCall> calls = new ArrayList<>();
            for (Part part : parts) {
                if (part instanceof Part.ToolCall c) calls.add(c);
                else if (part instanceof Part.Reasoning r) calls.addAll(collectCalls(r.content()));
            }
            return calls;
        }
    }

    // ---- sampler / grammar / stop / think wiring ---------------------------

    /**
     * Builds a grammar cursor from request params: {@code grammar} (GBNF string) or {@code
     * response_format: {type: "json_object"}}. Returns null when no constraint.
     */
    /** The request's output grammar, if any; the engine turns it into a cursor. */
    private Grammar.Spec grammarSpec(Map<String, Object> request) {
        if (!RuntimeFlags.GRAMMAR) return null;
        Tokenizer tokenizer = model.tokenizer();
        Object gbnf = request.get("grammar");
        if (gbnf instanceof String g && !g.isBlank()) {
            return Grammar.of(g, tokenizer);
        }
        Object fmt = request.get("response_format");
        if (fmt instanceof Map<?, ?> f && "json_object".equals(f.get("type"))) {
            return Grammar.json(tokenizer);
        }
        return null;
    }

    /** Request budget under the server's own ceiling ({@code jinfer.serverMaxTokens}). */
    private int maxTokens(Map<String, Object> request) {
        int maxTokens =
                Values.intValue(
                        request.getOrDefault("max_tokens", request.get("max_completion_tokens")),
                        options.maxTokens());
        LLMOptions.require(Values.intValue(request.get("n"), 1) == 1, "Only n=1 is supported");
        LLMOptions.require(0 <= maxTokens, "Invalid argument: max_tokens must be non-negative");
        if (ServerFlags.SERVER_MAX_TOKENS > 0) {
            maxTokens =
                    maxTokens < 0
                            ? ServerFlags.SERVER_MAX_TOKENS
                            : Math.min(maxTokens, ServerFlags.SERVER_MAX_TOKENS);
        }
        return maxTokens;
    }

    /** Prompt size as billed: a leading BOS is template overhead, not user input. */
    private static int consumedPromptTokens(Tokenizer tokenizer, List<Batch> prompt) {
        return consumedPromptTokens(tokenizer, IntSequence.wrap(Batch.tokenIds(prompt)));
    }

    /**
     * User stop strings stay TEXT stops only: token stops end generation anywhere, including inside
     * the think span, while text stops are matched against content alone.
     */
    private static List<String> textStops(Object value) {
        List<String> textStops = new ArrayList<>();
        if (value instanceof String s) {
            if (!s.isEmpty()) textStops.add(s);
        } else if (value instanceof List<?> values) {
            for (Object item : values) {
                String stop = Values.stringValue(item, "");
                if (!stop.isEmpty()) textStops.add(stop);
            }
        } else if (value != null) {
            throw new IllegalArgumentException("stop must be a string or an array of strings");
        }
        return List.copyOf(textStops);
    }

    /**
     * Effective thinking switch for a server request: chat_template_kwargs.enable_thinking
     * (llama.cpp convention) overrides the CLI --think flag. Forced tool calls never think - the
     * call marker is seeded as the first assistant token.
     */
    private boolean requestThink(Map<String, Object> request) {
        if (ToolUse.forced(request) != null) {
            return false;
        }
        if (request.get("chat_template_kwargs") instanceof Map<?, ?> kwargs
                && kwargs.get("enable_thinking") instanceof Boolean enabled) {
            return enabled;
        }
        return options.think();
    }

    /**
     * llama.cpp-compatible reasoning_format: "none" = leave thinking inline in content (with
     * literal <think> markers) instead of routing it to the reasoning_content channel - lets
     * vanilla OpenAI clients that only render content show thinking live.
     */
    private static boolean inlineReasoning(Map<String, Object> request) {
        return "none".equals(Values.stringValue(request.get("reasoning_format"), null));
    }
}
