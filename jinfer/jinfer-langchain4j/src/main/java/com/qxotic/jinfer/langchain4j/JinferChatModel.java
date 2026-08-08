package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.chat.CachedPrompt;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.TextStops;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * langchain4j {@link ChatModel} backed by jinfer: in-process CPU inference over a local GGUF.
 * Prompting goes native-first through the model's hand-written, oracle-validated chat-template
 * codec (token-exact, injection-inert) and falls back to a scrubbed Jinja whole-render for unported
 * models or unframeable requests.
 *
 * <p>Concurrency contract: an instance is ONE serial inference pipeline - concurrent requests queue
 * fairly on it. For a second pipeline, build a second model: the weight PAGES are shared by the OS
 * page cache, so the added cost is one context plus one load. Footprint: an instance holds its
 * weights, ONE full-context state reused for every request (extended on a prefix hit when {@code
 * cachedSessions} is set, reset otherwise - never re-allocated per request), plus the block layer's
 * KV (every served conversation, best-effort, bounded by {@code jinfer.promptCacheMB}; defined
 * prompts are pinned intent within it).
 *
 * <p>Three caching tiers, near-homonyms with distinct jobs: {@code withCachedPrompt} defines a LIVE
 * shared prefix (prefilled once, restored per request - the system-prompt/tools/few-shot case);
 * {@code Builder.cachedSessions} keeps finished CONVERSATION states warm for append-only multi-turn
 * reuse (in-RAM, gone at close); {@code saveCachedPrompts}/{@code Builder.loadCachedPrompts}
 * persist the defined prompts as an immutable ARTIFACT that mounts zero-prefill in later processes.
 * None changes output - byte-identity to a cold run is the law.
 *
 * <p>Run with jinfer's JVM flags: {@code --enable-preview --add-modules jdk.incubator.vector
 * --enable-native-access=ALL-UNNAMED}.
 */
public final class JinferChatModel implements ChatModel, AutoCloseable {

    final ChatEngine engine;
    final ChatRequestParameters defaults;
    final boolean thinking;
    final Long seed;
    final long timeoutNanos;
    final List<ChatModelListener> listeners;
    final com.qxotic.jinfer.media.VideoSampler videoSampler;
    // cached-prompt view state: EMPTY for the base model. Converted to jinfer types ONCE at view
    // creation (media decoded once, not per request); a view's conversations all start with this
    // prefix, its KV restored from the engine's block tree instead of re-prefilled.
    final CachedPrompt prefix;

    /**
     * The builder's two cache knobs as the cache's own record. Read-only: this mounts a catalog to
     * SERVE, never to write - a provider embedded in an application must not append to a file the
     * application did not ask it to write.
     */
    private static com.qxotic.jinfer.cache.PromptCache.Options cacheOptions(
            java.nio.file.Path cachedPrompts, int cachedSessions, int contextLength) {
        var defaults = com.qxotic.jinfer.cache.PromptCache.Options.DEFAULTS;
        return defaults.withHotSessions(cachedSessions)
                .withContextCapacity(
                        contextLength <= 0 ? defaults.contextCapacity() : contextLength)
                .withCatalog(cachedPrompts, true);
    }

    private JinferChatModel(Builder b) {
        this.engine =
                b.loaded == null
                        ? new ChatEngine(
                                b.modelPath,
                                java.util.Map.copyOf(b.companions),
                                cacheOptions(b.cachedPrompts, b.cachedSessions, b.contextLength))
                        : new ChatEngine(
                                b.loaded,
                                b.modelName == null
                                        ? b.loaded.model().getClass().getSimpleName()
                                        : b.modelName,
                                cacheOptions(b.cachedPrompts, b.cachedSessions, b.contextLength));
        this.thinking = b.thinking;
        this.seed = b.seed;
        this.timeoutNanos = b.timeout == null ? 0 : b.timeout.toNanos();
        this.listeners = List.copyOf(b.listeners);
        this.videoSampler = b.videoSampler;
        this.prefix = CachedPrompt.NONE;
        // Jinfer-typed ALWAYS: ChatModel.chat merges defaults.overrideWith(request), and only a
        // jinfer-typed receiver preserves grammar/seed from either side of the merge
        // precedence: request > builder > the container's recommendation (general.sampling.*)
        // > port author recommendation > the engine baseline (SamplingDefaults.DEFAULT_*)
        var recommended = engine.loaded().samplingDefaults();
        JinferChatRequestParameters base =
                JinferChatRequestParameters.builder()
                        .modelName(engine.modelName())
                        .temperature(
                                b.temperature != null
                                        ? b.temperature
                                        : toDouble(recommended.temperature()))
                        .topP(b.topP != null ? b.topP : toDouble(recommended.topP()))
                        .topK(b.topK != null ? b.topK : recommended.topK())
                        .minP(b.minP != null ? b.minP : toDouble(recommended.minP()))
                        .maxOutputTokens(b.maxOutputTokens)
                        .build();
        // caller-supplied defaults win field-by-field; unsupported ones reject HERE, eagerly - a
        // model whose defaults can never serve a request should fail at build, not at first use
        this.defaults = b.defaultParameters == null ? base : base.overrideWith(b.defaultParameters);
        if (b.defaultParameters != null) {
            rejectUnsupported(this.defaults);
            rejectModelSwitch(engine, this.defaults);
        }
    }

    private JinferChatModel(JinferChatModel base, CachedPrompt prefix) {
        this.engine = base.engine;
        this.defaults = base.defaults;
        this.thinking = base.thinking;
        this.seed = base.seed;
        this.timeoutNanos = base.timeoutNanos;
        this.listeners = base.listeners;
        this.videoSampler = base.videoSampler;
        this.prefix = prefix;
    }

    /**
     * A model view whose conversations all start with {@code prefix} (+ welded {@code tools}) -
     * prefilled ONCE into the engine's block tree, restored (not recomputed) on every chat.
     * Composable: calling this on a view branches on its prefix. Immutable, shares the base engine;
     * a view's prefix is pinned intent, where the base model's traffic is cached best-effort.
     *
     * <p>(The tree serves the BASE model too: under the default {@code jinfer.promptCache=true},
     * every conversation on a codec model is resumed from and committed to it, bounded by {@code
     * jinfer.promptCacheMB} with LRU eviction. {@code -Djinfer.promptCache=false} turns that
     * retention off; defined views still work through an explicitly mounted artifact.)
     *
     * <p>Typical shape:
     *
     * <pre>{@code
     * var base = JinferChatModel.builder().modelPath(gguf).build();
     * var support = base.withCachedPrompt(List.of(SYSTEM_PROMPT), TOOLS); // prefilled ONCE
     * support.chat(...);            // restores the prefix, ingests only the request
     * var billing = support.withCachedPrompt(List.of(BILLING_ADDENDUM), List.of()); // branches
     * }</pre>
     */
    public JinferChatModel withCachedPrompt(
            List<ChatMessage> prefixMessages, List<ToolSpecification> tools) {
        return withPrefix(
                prefix.merge(
                        Mappings.toMessages(prefixMessages, videoSampler),
                        tools == null ? List.of() : Mappings.toTools(tools)));
    }

    private JinferChatModel withPrefix(CachedPrompt merged) {
        framed(() -> engine.define(merged.conversation(thinking)));
        return new JinferChatModel(this, merged);
    }

    /** Freezes every prompt defined so far (plus any mounted base) into one artifact. */
    public void saveCachedPrompts(Path out) {
        engine.freezePrompts(out);
    }

    /**
     * Blocking, idempotent: waits out any in-flight request (including a live stream), then frees
     * the pooled session states' arenas and the cached-prompt blobs deterministically; later use of
     * this model (or any view sharing its engine) fails with IllegalStateException.
     *
     * <p>Weights are freed too, LAST and only if this model loaded them: mapped tensor pages are
     * kernel-reclaimable, but load-time conversions and repacks are anonymous memory that a
     * GC-managed arena would free only at a GC a native-heavy JVM never runs. A model built with
     * {@code model(...)} borrows its weights instead - close YOUR arena after this, never before.
     */
    @Override
    public void close() {
        engine.close();
    }

    @Override
    public ChatRequestParameters defaultRequestParameters() {
        return defaults;
    }

    @Override
    public List<ChatModelListener> listeners() {
        return listeners; // core's chat() dispatches onRequest/onResponse/onError
    }

    @Override
    public java.util.Set<dev.langchain4j.model.chat.Capability> supportedCapabilities() {
        // grammar-constrained decoding honors JSON schemas natively - AiServices reads this to
        // use structured output for POJO extraction instead of prompt-based JSON begging
        return java.util.Set.of(dev.langchain4j.model.chat.Capability.RESPONSE_FORMAT_JSON_SCHEMA);
    }

    @Override
    public ChatResponse doChat(ChatRequest request) {
        ChatEngine.Prepared p = prepare(request);
        ChatEngine.Completion done = engine.complete(p, ChatEngine.ReplySink.NONE);
        AiMessage ai = Mappings.toAiMessage(done.reply());
        if (done.stopped()) {
            ai = Mappings.withText(ai, TextStops.apply(ai.text(), p.stops()).text());
        }
        return Mappings.response(
                engine.modelName(), ai, p.promptTokens(), done.result(), done.stopped());
    }

    /**
     * Token counting over THIS model's tokenizer: exact on text, message counts summing visible
     * text (chat scaffold excluded - a few percent that consumer margins absorb). For {@code
     * TokenWindowChatMemory} budgets and token-aware splitters.
     */
    public dev.langchain4j.model.TokenCountEstimator tokenCountEstimator() {
        var template = engine.loaded().template().orElse(null);
        return new Estimators(
                engine.loaded().tokenizer(), template == null ? null : template::mediaPositions);
    }

    /** A streaming twin sharing this model's engine and cached prefix (the GGUF loads once). */
    public JinferStreamingChatModel streaming() {
        return new JinferStreamingChatModel(this);
    }

    // ---- shared request preparation (also used by the streaming twin) ----

    /** Every request-shape rejection, synchronously; both entry points reach it via prepare(). */
    void validate(ChatRequest request) {
        ChatRequestParameters p = request.parameters();
        rejectUnsupported(p);
        rejectModelSwitch(engine, p);
        boolean requestHasTools =
                p.toolSpecifications() != null && !p.toolSpecifications().isEmpty();
        if (!prefix.isEmpty() && requestHasTools) {
            throw new UnsupportedFeatureException(
                    "a cached-prompt view welds its tools into the cached prefix; per-request"
                            + " toolSpecifications would silently forfeit the cache - put tools on"
                            + " withCachedPrompt(...) instead");
        }
        if (p.toolChoice() == ToolChoice.REQUIRED && !requestHasTools && prefix.tools().isEmpty()) {
            throw new IllegalArgumentException("toolChoice REQUIRED without any tools");
        }
        if (p.toolChoice() == ToolChoice.NONE && !prefix.tools().isEmpty()) {
            throw new UnsupportedFeatureException(
                    "toolChoice NONE on a cached-prompt view is not supported: the view's tools"
                            + " are welded into its cached prefix and cannot be un-offered");
        }
    }

    /** Framework types mapped away; every policy below this line lives in {@link ChatEngine}. */
    ChatEngine.Prepared prepare(ChatRequest request) {
        validate(request);
        ChatRequestParameters p = request.parameters();
        boolean cached = !prefix.isEmpty();
        // NONE = the model cannot use tools: never offer them, and there is nothing to call
        List<ToolSpecification> requestTools =
                p.toolChoice() == ToolChoice.NONE || p.toolSpecifications() == null
                        ? List.of()
                        : p.toolSpecifications();
        List<Message> messages = new ArrayList<>(prefix.messages());
        messages.addAll(Mappings.toMessages(request.messages(), videoSampler));
        JinferChatRequestParameters j = p instanceof JinferChatRequestParameters jp ? jp : null;
        List<ChatMessage> requestMessages = request.messages();
        ChatEngine.Request lowered =
                new ChatEngine.Request(
                        messages,
                        cached ? prefix.tools() : Mappings.toTools(requestTools),
                        thinking,
                        p.maxOutputTokens() == null ? -1 : p.maxOutputTokens(),
                        null, // langchain4j has no reasoning-budget knob
                        timeoutNanos,
                        engine.loaded()
                                .samplingDefaults()
                                .resolve(
                                        p.temperature() == null
                                                ? null
                                                : p.temperature().floatValue(),
                                        p.topP() == null ? null : p.topP().floatValue(),
                                        p.topK(),
                                        j == null || j.minP() == null
                                                ? null
                                                : j.minP().floatValue(),
                                        j != null && j.seed() != null ? j.seed() : seed),
                        grammar(p, j),
                        p.toolChoice() == ToolChoice.REQUIRED ? "" : null,
                        cached,
                        p.stopSequences(),
                        null); // langchain4j has no chat_template_kwargs equivalent
        return framed(
                () ->
                        engine.prepare(
                                lowered,
                                () -> Mappings.toMessageMaps(requestMessages),
                                () -> Mappings.toToolMaps(requestTools)));
    }

    /**
     * The request's decoding constraint, if any - the one piece of the sampling stack that is
     * genuinely framework-shaped: langchain4j spells it as a response format (schemaless JSON or a
     * typed schema) or, jinfer-typed, as raw GBNF. Specs cache per (source, vocab), so repeated
     * schemas reuse the compiled masks.
     */
    private Grammar.Spec grammar(ChatRequestParameters p, JinferChatRequestParameters j) {
        var tokenizer = engine.loaded().tokenizer();
        ResponseFormat rf = p.responseFormat();
        if (rf != null && rf.type() == ResponseFormatType.JSON) {
            return rf.jsonSchema() == null
                    ? Grammar.json(tokenizer)
                    : Grammar.fromSchema(
                            Mappings.toSchemaMap(rf.jsonSchema().rootElement()), tokenizer);
        }
        // raw GBNF: the JSON format's generalization (validate() guaranteed they are not combined)
        return j == null || j.grammar() == null ? null : Grammar.of(j.grammar(), tokenizer);
    }

    private static <T> T framed(java.util.function.Supplier<T> op) {
        try {
            return op.get();
        } catch (UnsupportedOperationException e) {
            throw new UnsupportedFeatureException(e.getMessage());
        }
    }

    private static void framed(Runnable op) {
        framed(
                () -> {
                    op.run();
                    return null;
                });
    }

    /** One loaded GGUF per instance: a different {@code modelName} cannot be served. */
    private static void rejectModelSwitch(ChatEngine engine, ChatRequestParameters p) {
        if (p.modelName() != null && !p.modelName().equals(engine.modelName())) {
            throw new UnsupportedFeatureException(
                    "per-request modelName is not supported: this model IS '"
                            + engine.modelName()
                            + "' (one loaded GGUF per instance)");
        }
    }

    // topK is NOT rejected here: it is a supported sampling knob (the builder exposes it, the
    // port's recommendation seeds it, prepare() feeds it to the sampler). A guard here once
    // predated that support and, because chat() merges the defaults over the request, it rejected
    // EVERY request on a model whose port recommends a top_k - gemma4 does.
    private static void rejectUnsupported(ChatRequestParameters p) {
        if (p.frequencyPenalty() != null)
            throw new UnsupportedFeatureException("frequencyPenalty is not supported");
        if (p.presencePenalty() != null)
            throw new UnsupportedFeatureException("presencePenalty is not supported");
        ResponseFormat rf = p.responseFormat();
        boolean tools = p.toolSpecifications() != null && !p.toolSpecifications().isEmpty();
        if (rf != null && rf.type() == ResponseFormatType.JSON && tools)
            throw new UnsupportedFeatureException(
                    "tools together with a JSON response format are not supported:"
                            + " grammar-constrained output cannot admit tool-call syntax");
        String grammar = p instanceof JinferChatRequestParameters j ? j.grammar() : null;
        if (grammar != null && tools)
            throw new UnsupportedFeatureException(
                    "tools together with a grammar are not supported: grammar-constrained output"
                            + " cannot admit tool-call syntax");
        if (grammar != null && rf != null && rf.type() == ResponseFormatType.JSON)
            throw new UnsupportedFeatureException(
                    "grammar and a JSON response format are mutually exclusive: both constrain the"
                            + " same reply - pick one");
    }

    private static Double toDouble(Float f) {
        return f == null ? null : f.doubleValue();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Path modelPath;
        private LoadedModel<?> loaded;
        private String modelName;
        private final java.util.Map<String, Path> companions = new java.util.LinkedHashMap<>();
        private com.qxotic.jinfer.media.VideoSampler videoSampler =
                com.qxotic.jinfer.media.VideoSampler.UNIFORM;
        private Path cachedPrompts;
        private int cachedSessions;
        private int contextLength;
        private Double temperature;
        private Double topP;
        private Integer topK;
        private Double minP;
        private Integer maxOutputTokens;
        private ChatRequestParameters defaultParameters;
        private List<ChatModelListener> listeners = List.of();
        private boolean thinking = true;
        private Long seed;
        private Duration timeout;

        /** The GGUF to load. Required unless {@link #model}. */
        public Builder modelPath(Path modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        /**
         * A model you loaded yourself - the seam for a hand-built {@link LoadedModel}, e.g. one
         * carrying your own tokenizer via {@code LoadedModel.withTokenizer(...)}. Mutually
         * exclusive with {@link #modelPath}.
         *
         * <p>You own its weights arena: {@link JinferChatModel#close()} quiesces this model but
         * frees only what it allocated, so close your arena after it, never before.
         */
        public Builder model(LoadedModel<?> loaded) {
            this.loaded = loaded;
            return this;
        }

        /**
         * Reported as the response's model name; defaults to the model class. {@link #model} only.
         */
        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        /**
         * Mounts a cached-prompt artifact read-only; model-seed-checked. An incompatible file fails
         * the build loudly; a MISSING file degrades to serving without it (stderr warning) - check
         * the path if TTFT looks cold.
         */
        public Builder loadCachedPrompts(Path cachedPrompts) {
            this.cachedPrompts = cachedPrompts;
            return this;
        }

        /**
         * How video content becomes frames - default {@link
         * com.qxotic.jinfer.media.VideoSampler#UNIFORM} (the reference policy: 32 frames uniform
         * across the whole duration). Any policy composes: {@code v -> VideoCodec.ffmpeg().span(v,
         * 8)}, a window of a long source, caller-curated timestamps.
         */
        public Builder videoSampler(com.qxotic.jinfer.media.VideoSampler videoSampler) {
            this.videoSampler = java.util.Objects.requireNonNull(videoSampler);
            return this;
        }

        /**
         * Attaches a COMPANION: an auxiliary file that gives the model a capability it does not
         * have alone, keyed by that capability - {@code "media"} for the mmproj GGUF carrying the
         * vision and audio encoders. What an architecture accepts is {@code Models.companions}.
         */
        public Builder companion(String capability, Path file) {
            this.companions.put(capability, file);
            return this;
        }

        /**
         * Keeps the last {@code n} finished conversations' KV states LIVE for append-only reuse: a
         * request whose prompt strictly extends a kept conversation (its echoed turns re-encoding
         * to the exact generated tokens - the native codec's verbatim splice guarantees this for
         * unmodified echoes) pays prefill only for the delta. Purely a runtime warmth knob: output
         * is byte-identical to a cold run and nothing persists. Each kept conversation holds a full
         * context of KV.
         *
         * <p>The default 0 keeps the model stateless between requests - its state is wiped the
         * moment a reply ends, so no conversation survives the call - but the ALLOCATION is still
         * reused: a pipeline allocates its context once and never per request, whatever this is set
         * to. This knob buys warmth, not memory reuse.
         */
        public Builder cachedSessions(int cachedSessions) {
            this.cachedSessions = cachedSessions;
            return this;
        }

        /** Context window; 0 = the model's own maximum. */
        public Builder contextLength(int contextLength) {
            this.contextLength = contextLength;
            return this;
        }

        /**
         * Sampling temperature; default: the model's recommended value (the GGUF's {@code
         * general.sampling.temp}, or the model author's recommendation shipped with the port), else
         * 0.8. Per-request values override; pass 0 for greedy argmax.
         */
        public Builder temperature(Double temperature) {
            this.temperature = temperature;
            return this;
        }

        /**
         * Nucleus sampling mass, effective only at temperature &gt; 0; default: the model's
         * recommended value (the GGUF's {@code general.sampling.top_p}, or the port's), else 0.95.
         */
        public Builder topP(Double topP) {
            this.topP = topP;
            return this;
        }

        /**
         * Top-k cutoff (0 disables); default: the model's recommended value (the GGUF's {@code
         * general.sampling.top_k}, or the port's), else 40. Per-request values override.
         */
        public Builder topK(Integer topK) {
            this.topK = topK;
            return this;
        }

        /**
         * Min-p cutoff relative to the top token, in [0,1] (0 disables); default: the model's
         * recommended value ({@code general.sampling.min_p}, or the port's), else 0.05. Per-request
         * {@link JinferChatRequestParameters#minP} overrides.
         */
        public Builder minP(Double minP) {
            this.minP = minP;
            return this;
        }

        /**
         * Completion budget; default unlimited (the context bounds it). Values below 16 also
         * disable thinking - a think span cannot fit such a budget, and silently spending it on
         * scaffold would return empty text.
         */
        public Builder maxOutputTokens(Integer maxOutputTokens) {
            this.maxOutputTokens = maxOutputTokens;
            return this;
        }

        /** langchain4j listeners; core dispatches onRequest/onResponse/onError around chat. */
        public Builder listeners(List<ChatModelListener> listeners) {
            this.listeners = listeners;
            return this;
        }

        /**
         * Default request parameters, merged under each request's own (standard langchain4j
         * semantics). Unsupported parameters are rejected eagerly at build.
         */
        public Builder defaultRequestParameters(ChatRequestParameters defaultParameters) {
            this.defaultParameters = defaultParameters;
            return this;
        }

        /**
         * The model's reasoning scaffold toggle (templates without one ignore it). Default on.
         * Forced tool calls and completion budgets below 16 tokens disable it per request
         * regardless - the reply is seeded into the call block, or the budget cannot fit a span.
         */
        public Builder thinking(boolean thinking) {
            this.thinking = thinking;
            return this;
        }

        /**
         * RNG seed for temperature sampling; default: a fresh random seed per request. Set one to
         * pin sampling - a per-request {@link JinferChatRequestParameters#seed} wins over this.
         * Same seed does NOT guarantee byte-identical replay across processes: the CPU backend's
         * run-to-run FP jitter can flip a near-tie.
         */
        public Builder seed(Long seed) {
            this.seed = seed;
            return this;
        }

        /** Wall-clock deadline per request; unset = none. Exceeding it finishes with LENGTH. */
        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public JinferChatModel build() {
            if ((modelPath == null) == (loaded == null))
                throw new IllegalArgumentException(
                        "exactly one of modelPath(...) or model(...) is required");
            if (loaded != null && (!companions.isEmpty() || contextLength != 0))
                throw new IllegalArgumentException(
                        "companions/contextLength are load-time settings; apply them when you"
                                + " build the LoadedModel passed to model(...)");
            return new JinferChatModel(this);
        }
    }
}
