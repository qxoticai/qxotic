package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
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
 * The OpenAI wire lowered onto the shared runtime: parses request maps into {@link
 * ChatEngine.Request}s (chat) or raw token prompts (completions), drives them through {@link
 * ChatEngine#complete}, and shapes the outcome into the {@link Reply} the schema wants - including
 * the llama.cpp-compatible knobs (reasoning_format, chat_template_kwargs, bare-call recovery).
 * Generation semantics - templating, caching tiers, sampling policy, stop holdback, reply parsing -
 * all live in the engine; transport (endpoint handlers, SSE) lives in {@link Server}.
 */
final class Generation {

    private final LoadedModel<?> model;
    private final LLMOptions options;
    private final ChatTemplate template; // memoized model framing, null when the model has none
    // raw-prompt mode encodes text special-token-aware; nothing renders a template for it
    private final com.qxotic.toknroll.Specials specials;
    // The shared runtime: template stack, sampling policy, and the one PromptCache. This
    // class predates all of it and had its own of each; what remains here is the OpenAI wire.
    private final ChatEngine engine;
    private final Metrics metrics;
    // whether close() appends the catalog (a read-write --cache): the blocks a server
    // accumulates survive its restarts
    private final boolean saveCatalog;

    Generation(LoadedModel<?> chatModel, LLMOptions options, Metrics metrics) {
        this.metrics = metrics;
        this.model = chatModel;
        this.options = options;
        this.template = chatModel.template().orElse(null);
        this.specials = SpecialTokens.encoder(model.tokenizer());
        java.nio.file.Path catalog = options.promptCache();
        boolean mounted = catalog != null && java.nio.file.Files.exists(catalog);
        // borrowed weights: the server loaded the model and keeps its arena. The engine's cache
        // owns the whole catalog policy - a codec-less model warns and ignores the file,
        // read-only problems degrade, read-write fail loudly.
        this.engine =
                new ChatEngine(
                        chatModel,
                        options.modelPath().getFileName().toString(),
                        catalog,
                        options.promptCacheReadOnly(),
                        RuntimeFlags.SESSIONS);
        boolean usable = catalog != null && engine.blockCaching();
        this.saveCatalog = usable && !options.promptCacheReadOnly();
        if (usable) {
            System.out.printf(
                    "prompt cache: %s%s%s%n",
                    catalog,
                    mounted ? " (mounted)" : " (new)",
                    options.promptCacheReadOnly() ? " read-only" : ", saved at shutdown");
        }
    }

    /** The block tree's health reading for {@code /props}; null = no tree behind this model. */
    com.qxotic.jinfer.cache.PromptCache.Sample cacheSample() {
        return engine.cacheSample();
    }

    boolean blockCaching() {
        return engine.blockCaching();
    }

    private final java.util.concurrent.atomic.AtomicBoolean closed =
            new java.util.concurrent.atomic.AtomicBoolean();

    /**
     * Frees the engine's states and blocks; the weights arena stays the server's. Idempotent: the
     * shutdown hook re-runs close after an embedder's explicit one, and the second run must not
     * re-append the catalog or log a false "failed to save".
     */
    void close() {
        if (!closed.compareAndSet(false, true)) return;
        if (saveCatalog) {
            // best-effort: a failed write-back must never block the engine's shutdown
            try {
                engine.savePrompts();
                System.out.println("prompt cache saved: " + options.promptCache());
            } catch (RuntimeException e) {
                System.err.println(
                        "failed to save prompt cache " + options.promptCache() + ": " + e);
            }
        }
        engine.close();
    }

    // ---- chat / completion -------------------------------------------------

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

    @SuppressWarnings("unchecked")
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
                        reasoningMax(request),
                        ServerFlags.SERVER_REQUEST_TIMEOUT_NANOS,
                        Values.floatValue(request.get("temperature"), options.temperature()),
                        Values.floatValue(request.get("top_p"), options.topp()),
                        Values.longValue(request.get("seed"), options.seed()),
                        grammarSpec(request),
                        nativeForcedOk() ? ToolUse.forced(request) : null,
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
                        consumedPromptTokens(model.tokenizer(), prepared));
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
        metrics.recordPromptCache(
                done.tier() == com.qxotic.jinfer.cache.PromptCache.Tier.SESSION,
                done.restoredTokens());
        return router.reply(
                done.result(), done.reply(), billed, done.restoredTokens(), prepared.stops());
    }

    /**
     * Whether this model can force a call at all (a native codec that declares a call seed). When
     * it cannot, the request degrades to an unforced generation rather than failing - forcing is
     * best-effort on the wire. A forced name the request never offered is the engine's law now: the
     * {@link ChatEngine.Request} constructor rejects it, and the 400 mapping answers.
     */
    private boolean nativeForcedOk() {
        return template != null && template.callSeed().length != 0;
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
            List<Part> contentParts = null; // typed lowering when the array carries images
            if (raw2 instanceof List<?> parts) {
                boolean hasTyped =
                        parts.stream()
                                .anyMatch(
                                        part ->
                                                part instanceof Map<?, ?> pm
                                                        && pm.get("type") != null
                                                        && !"text".equals(pm.get("type")));
                if (hasTyped) {
                    // non-text parts can never reach the whole-render fallback (media needs the
                    // native template; unknown types deserve their precise error, not the
                    // generic empty-request 400), so this path either builds typed parts or
                    // throws - it must not return null
                    contentParts = new ArrayList<>();
                    for (Object part : parts) {
                        Map<String, Object> pm = Values.asObject(part, "content part");
                        switch (Values.stringValue(pm.get("type"), "")) {
                            case "text" -> {
                                String t = Values.stringValue(pm.get("text"), "");
                                if (!t.isEmpty()) contentParts.add(new Part.Text(t));
                            }
                            case "image_url" -> contentParts.add(imagePart(pm.get("image_url")));
                            default ->
                                    LLMOptions.require(
                                            false,
                                            "unsupported content part type: %s",
                                            pm.get("type"));
                        }
                    }
                } else {
                    for (Object part : parts) { // pure-text flattens faithfully
                        if (!(part instanceof Map<?, ?> pm) || !"text".equals(pm.get("type")))
                            return null;
                    }
                }
            }
            String content = Values.messageContent(raw2);
            List<Part> callParts = toolCallParts(m.get("tool_calls"));
            if (callParts == null) return null; // malformed tool_calls: whole-render
            if (contentParts != null) {
                List<Part> all = new ArrayList<>(contentParts);
                all.addAll(callParts);
                out.add(new Message(new Role(role), all));
            } else if (!callParts.isEmpty()) {
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
     * One {@code image_url} content item to a typed media part. Vision is a capability the server
     * OPTS INTO: without {@code --mmproj} every image request is refused with a clear 400 (the
     * encoder is not loaded - silently dropping the image would answer a question the model never
     * saw). Only {@code data:} URIs are accepted - fetching remote URLs from a server is an SSRF
     * surface, and the OpenAI wire supports inline base64 everywhere.
     */
    private Part imagePart(Object imageUrl) {
        LLMOptions.require(
                options.mediaProjector() != null,
                "image input is not enabled on this server (start it with --mmproj"
                        + " <projector.gguf>)");
        LLMOptions.require(
                model.model() instanceof com.qxotic.jinfer.MultiModal mm
                        && mm.modalities().contains(com.qxotic.jinfer.Media.Image.class),
                "the loaded projector provides no image encoder for this model");
        String url =
                imageUrl instanceof Map<?, ?> mu
                        ? Values.stringValue(mu.get("url"), "")
                        : Values.stringValue(imageUrl, "");
        LLMOptions.require(
                url.startsWith("data:"),
                "image_url must be a data: URI (the server does not fetch remote URLs)");
        int comma = url.indexOf(',');
        LLMOptions.require(
                comma > 0 && url.substring(0, comma).endsWith(";base64"),
                "image_url data: URI must be base64-encoded (data:<mime>;base64,<payload>)");
        byte[] encoded;
        try {
            encoded = java.util.Base64.getDecoder().decode(url.substring(comma + 1));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("image_url base64 payload is malformed");
        }
        try {
            byte[] key = java.security.MessageDigest.getInstance("SHA-256").digest(encoded);
            return new Part.Blob(com.qxotic.jinfer.media.ImageCodec.decode(encoded), key);
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        } catch (java.io.IOException e) {
            throw new IllegalArgumentException("image could not be decoded: " + e.getMessage());
        }
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

    Reply completion(Map<String, Object> request, String prompt, Sinks sinks) {
        return recorded(completion0(request, prompt, sinks));
    }

    private Reply completion0(Map<String, Object> request, String prompt, Sinks sinks) {
        Tokenizer tokenizer = model.tokenizer();
        IntSequence promptTokens =
                options.rawPrompt() ? specials.encode(tokenizer, prompt) : tokenizer.encode(prompt);
        return generate(request, promptTokens, sinks);
    }

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
        ChatEngine.Prepared prepared =
                ChatEngine.Prepared.raw(
                        promptTokens.toArray(),
                        sampler,
                        maxTokens,
                        ServerFlags.SERVER_REQUEST_TIMEOUT_NANOS,
                        textStops(request.get("stop")));
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
     * The request's output grammar - {@code grammar} (GBNF string) or {@code response_format:
     * {type: "json_object"}} - or null when unconstrained; the engine turns it into a cursor.
     */
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

    /**
     * Same rule for an encoded prompt, without flattening it: the count is already on {@link
     * ChatEngine.Prepared}, and BOS can only be the very first token of the first text batch.
     */
    private static int consumedPromptTokens(Tokenizer tokenizer, ChatEngine.Prepared prepared) {
        List<Batch> prompt = prepared.encoded().prompt();
        if (!prompt.isEmpty()
                && prompt.get(0).input() instanceof Batch.Input.Tokens first
                && first.ids().length > 0) {
            int bos = SpecialTokens.findFirst(tokenizer, "<bos>", "<|startoftext|>").orElse(1);
            if (first.ids()[0] == bos) return prepared.promptTokens() - 1;
        }
        return prepared.promptTokens();
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
