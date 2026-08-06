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
import com.qxotic.jinfer.chat.Thinking;
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
    private final ServerConfig config;
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
    // this server's configured stack, resolved once; every request overrides a copy of it
    private final Sampling defaultSampling;

    Generation(LoadedModel<?> chatModel, ServerConfig config, Metrics metrics) {
        this.metrics = metrics;
        this.model = chatModel;
        this.config = config;
        this.defaultSampling = config.defaults().sampling();
        this.template = chatModel.template().orElse(null);
        this.specials = SpecialTokens.encoder(model.tokenizer());
        java.nio.file.Path catalog = config.cache().catalog();
        boolean mounted = catalog != null && java.nio.file.Files.exists(catalog);
        // borrowed weights: the server loaded the model and keeps its arena. The engine's cache
        // owns the whole catalog policy - a codec-less model warns and ignores the file,
        // read-only problems degrade, read-write fail loudly.
        this.engine =
                new ChatEngine(
                        chatModel,
                        config.modelName(),
                        catalog,
                        config.cache().readOnly(),
                        RuntimeFlags.SESSIONS);
        boolean usable = catalog != null && engine.blockCaching();
        this.saveCatalog = usable && !config.cache().readOnly();
        if (usable) {
            Log.LOG.log(
                    System.Logger.Level.INFO,
                    () ->
                            "prompt cache: %s%s%s"
                                    .formatted(
                                            catalog,
                                            mounted ? " (mounted)" : " (new)",
                                            config.cache().readOnly()
                                                    ? " read-only"
                                                    : ", saved at shutdown"));
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
                Log.LOG.log(
                        System.Logger.Level.INFO,
                        () -> "prompt cache saved: " + config.cache().catalog());
            } catch (RuntimeException e) {
                Log.LOG.log(
                        System.Logger.Level.WARNING,
                        () -> "failed to save prompt cache " + config.cache().catalog(),
                        e);
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
                        config.limits().requestTimeout().toNanos(),
                        sampling(request),
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
            if ("tool".equals(role)) {
                // typed lowering: the wire carries the result text and its tool_call_id, and
                // every native template folds Part.ToolResult - five of them accept ONLY that
                // shape, and lowering as plain Text silently rendered their served tool results
                // empty (templates on the generic per-turn path punt to the whole render, which
                // reads the raw maps and never sees this part)
                out.add(
                        new Message(
                                new Role(role),
                                List.of(
                                        new Part.ToolResult(
                                                Values.stringValue(m.get("tool_call_id"), ""),
                                                Values.messageContent(m.get("content"))))));
                continue;
            }
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
                            case "input_audio" ->
                                    contentParts.add(audioPart(pm.get("input_audio")));
                            case "video_url" -> contentParts.add(videoPart(pm.get("video_url")));
                            default ->
                                    Validation.require(
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
            // an assistant echo may return its reasoning (reasoning/reasoning_content) - typed as
            // Part.Reasoning so templates that preserve thought in the tool loop can re-render it
            Part reasoning = null;
            if ("assistant".equals(role)) {
                String rt = Values.stringValue(m.get("reasoning"), null);
                if (rt == null) rt = Values.stringValue(m.get("reasoning_content"), null);
                if (rt != null && !rt.isBlank()) {
                    reasoning = new Part.Reasoning(List.of(new Part.Text(rt)), null);
                }
            }
            if (contentParts != null || !callParts.isEmpty() || reasoning != null) {
                List<Part> all = new ArrayList<>();
                if (reasoning != null) all.add(reasoning);
                if (contentParts != null) all.addAll(contentParts);
                else if (!content.isEmpty()) all.add(new Part.Text(content));
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
        Validation.require(
                model.model() instanceof com.qxotic.jinfer.MultiModal mm
                        && mm.modalities().contains(com.qxotic.jinfer.Media.Image.class),
                "this model cannot read images: no vision encoder is loaded. Start the server with"
                        + " a projector that carries one, e.g. --with media=<mmproj.gguf>");
        byte[] encoded = dataUriBytes(urlOf(imageUrl), "image_url");
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
     * One {@code input_audio} content item ({@code {data: <base64>, format: wav|mp3}}) to a typed
     * media part - the same opt-in gate and source-keying as images. The audio encoders ride the
     * larger mmproj sidecars (gemma 12B+); a projector without them gets a clear 400.
     */
    private Part audioPart(Object inputAudio) {
        Validation.require(
                model.model() instanceof com.qxotic.jinfer.MultiModal mm
                        && mm.modalities().contains(com.qxotic.jinfer.Media.Audio.class),
                "this model cannot read audio: no audio encoder is loaded. Start the server with a"
                        + " projector that carries one, e.g. --with media=<mmproj.gguf>");
        Map<String, Object> audio = Values.asObject(inputAudio, "input_audio");
        byte[] encoded;
        try {
            encoded =
                    java.util.Base64.getDecoder().decode(Values.stringValue(audio.get("data"), ""));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("input_audio data is not valid base64");
        }
        try {
            byte[] key = java.security.MessageDigest.getInstance("SHA-256").digest(encoded);
            return new Part.Blob(com.qxotic.jinfer.media.AudioCodec.decode(encoded), key);
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        } catch (java.io.IOException e) {
            throw new IllegalArgumentException("audio could not be decoded: " + e.getMessage());
        }
    }

    /**
     * One {@code video_url} content item (vLLM's content-part convention) to a typed media part -
     * the same opt-in gate and source-keying as images. The gate is the VISION encoder: video
     * decomposes into timestamped image frames in the template, so a projector without vision gets
     * a clear 400. The payload lands in a temp file because ffmpeg samples files; sampling is the
     * reference processor's policy ({@code uniform}, {@link
     * com.qxotic.jinfer.media.VideoCodec#DEFAULT_NUM_FRAMES} frames across the whole duration), and
     * the template derives each frame's cache key from this blob's digest plus its timestamp.
     */
    private Part videoPart(Object videoUrl) {
        Validation.require(
                model.model() instanceof com.qxotic.jinfer.MultiModal mm
                        && mm.modalities().contains(com.qxotic.jinfer.Media.Image.class),
                "this model cannot read video: no vision encoder is loaded (video decomposes into"
                        + " image frames). Start the server with a projector that carries one,"
                        + " e.g. --with media=<mmproj.gguf>");
        byte[] encoded = dataUriBytes(urlOf(videoUrl), "video_url");
        try {
            byte[] key = java.security.MessageDigest.getInstance("SHA-256").digest(encoded);
            if (template != null && template.mediaEncodingCached(key)) {
                // full skip: no temp file, no ffmpeg, no encoder - the template replays its
                // cached rows for this digest (a frameless keyed blob is that contract)
                return new Part.Blob(new com.qxotic.jinfer.Media.Video(java.util.List.of()), key);
            }
            java.nio.file.Path tmp = java.nio.file.Files.createTempFile("jinfer-video", ".bin");
            try {
                java.nio.file.Files.write(tmp, encoded);
                return new Part.Blob(com.qxotic.jinfer.media.VideoCodec.ffmpeg().uniform(tmp), key);
            } finally {
                java.nio.file.Files.deleteIfExists(tmp);
            }
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        } catch (java.io.IOException e) {
            throw new IllegalArgumentException("video could not be decoded: " + e.getMessage());
        }
    }

    /** The url from either the OpenAI object form ({@code {url: ...}}) or a bare string. */
    private static String urlOf(Object part) {
        return part instanceof Map<?, ?> mu
                ? Values.stringValue(mu.get("url"), "")
                : Values.stringValue(part, "");
    }

    /**
     * The base64 payload of a {@code data:} URI - the only URL form served media accepts (fetching
     * remote URLs from a server is an SSRF surface, and the OpenAI wire supports inline base64
     * everywhere).
     */
    private static byte[] dataUriBytes(String url, String what) {
        Validation.require(
                url.startsWith("data:"),
                "%s must be a data: URI (the server does not fetch remote URLs)",
                what);
        int comma = url.indexOf(',');
        Validation.require(
                comma > 0 && url.substring(0, comma).endsWith(";base64"),
                "%s data: URI must be base64-encoded (data:<mime>;base64,<payload>)",
                what);
        try {
            return java.util.Base64.getDecoder().decode(url.substring(comma + 1));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(what + " base64 payload is malformed");
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
                // upstream template fix 35b4173: arguments must deserialize to a JSON object -
                // the reference raises rather than silently rendering an empty argument map
                Object parsed;
                try {
                    parsed = JsonCodec.parse(s);
                } catch (RuntimeException notJson) {
                    parsed = null;
                }
                Validation.require(
                        parsed instanceof Map<?, ?>,
                        "tool_calls[].function.arguments must be a JSON object (mapping), got:"
                                + " %s",
                        s.length() > 80 ? s.substring(0, 80) + "..." : s);
                argMap.putAll((Map<String, Object>) parsed);
            } else if (args instanceof Map<?, ?> parsed) {
                argMap.putAll((Map<String, Object>) parsed);
            } else if (args != null && !(args instanceof String)) {
                Validation.require(
                        false,
                        "tool_calls[].function.arguments must be a JSON object (mapping), got a"
                                + " %s",
                        args.getClass().getSimpleName());
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
                config.defaults().rawPrompt()
                        ? specials.encode(tokenizer, prompt)
                        : tokenizer.encode(prompt);
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
        // a raw prompt has no conversation, so the template's STATIC seed is the only possible
        // tail knowledge (the chat path gets the conversation-aware seed from encodePrompt)
        int[] replySeed = model.template().map(t -> t.replySeed(think)).orElseGet(() -> new int[0]);
        Sampler sampler =
                RequestPolicy.sampler(
                        model,
                        sampling(request),
                        think,
                        maxTokens,
                        reasoningMax(request),
                        replySeed);
        if (grammar != null) {
            sampler = RequestPolicy.constrained(model, sampler, grammar.cursor(), replySeed);
        }
        ChatEngine.Prepared prepared =
                ChatEngine.Prepared.raw(
                        promptTokens.toArray(),
                        sampler,
                        maxTokens,
                        config.limits().requestTimeout().toNanos(),
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
        private final Thinking.Inline inline; // null when reasoning routes to its own channel

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
            this.inline = inline ? new Thinking.Inline() : null;
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

    /**
     * The server's configured stack with this request's overrides on top. Both framings (chat and
     * raw prompt) read it here rather than each spelling out the same five fallbacks, which is how
     * they came to disagree: one resolved a missing seed per request, the other per process.
     */
    private Sampling sampling(Map<String, Object> request) {
        // a STATEMENT, not a ternary: mixing Long and long in one conditional makes javac unbox
        // the Long branch, so a server without --seed (the default, meaning fresh randomness per
        // request) failed every request with a NullPointerException
        Long seed = defaultSampling.seed();
        if (request.get("seed") != null) {
            seed = Values.longValue(request.get("seed"), 0L);
        }
        return new Sampling(
                Values.floatValue(request.get("temperature"), defaultSampling.temperature()),
                Values.floatValue(request.get("top_p"), defaultSampling.topP()),
                Values.intValue(request.get("top_k"), defaultSampling.topK()),
                Values.floatValue(request.get("min_p"), defaultSampling.minP()),
                seed);
    }

    /** Request budget under the server's own ceiling ({@link ServerConfig.Limits#maxTokens}). */
    private int maxTokens(Map<String, Object> request) {
        int maxTokens =
                Values.intValue(
                        request.getOrDefault("max_tokens", request.get("max_completion_tokens")),
                        config.defaults().maxTokens());
        Validation.require(Values.intValue(request.get("n"), 1) == 1, "Only n=1 is supported");
        Validation.require(0 <= maxTokens, "Invalid argument: max_tokens must be non-negative");
        int ceiling = config.limits().maxTokens();
        if (ceiling > 0) {
            maxTokens = maxTokens < 0 ? ceiling : Math.min(maxTokens, ceiling);
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
        return config.defaults().think();
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
