package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.chat.CachedPrompt;
import com.qxotic.jinfer.chat.ChatEngine;
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
 * cachedSessions} is set, reset otherwise - never re-allocated per request), plus one KV block set
 * per defined cached prompt (explicit and deliberately paid for).
 *
 * <p>Three caching tiers, near-homonyms with distinct jobs: {@code withCachedPrompt} defines a LIVE
 * shared prefix (prefilled once, restored per request - the system-prompt/tools/few-shot case);
 * {@code Builder.cachedSessions} keeps finished CONVERSATION states warm for append-only multi-turn
 * reuse (ephemeral, nothing persists); {@code saveCachedPrompts}/{@code Builder.loadCachedPrompts}
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
    final long seed;
    final long timeoutNanos;
    final List<ChatModelListener> listeners;
    // cached-prompt view state: EMPTY for the base model. Converted to jinfer types ONCE at view
    // creation (media decoded once, not per request); a view's conversations all start with this
    // prefix, its KV restored from the engine's block tree instead of re-prefilled.
    final CachedPrompt prefix;

    private JinferChatModel(Builder b) {
        this.engine =
                new ChatEngine(
                        b.modelPath,
                        b.mediaProjector,
                        b.contextLength,
                        b.cachedPrompts,
                        b.cachedSessions);
        this.thinking = b.thinking;
        this.seed = b.seed;
        this.timeoutNanos = b.timeout == null ? 0 : b.timeout.toNanos();
        this.listeners = List.copyOf(b.listeners);
        this.prefix = CachedPrompt.NONE;
        // Jinfer-typed ALWAYS: ChatModel.chat merges defaults.overrideWith(request), and only a
        // jinfer-typed receiver preserves grammar/seed from either side of the merge
        JinferChatRequestParameters base =
                JinferChatRequestParameters.builder()
                        .modelName(engine.modelName())
                        .temperature(b.temperature)
                        .topP(b.topP)
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
        this.prefix = prefix;
    }

    /**
     * A model view whose conversations all start with {@code prefix} (+ welded {@code tools}) -
     * prefilled ONCE into the engine's block tree, restored (not recomputed) on every chat.
     * Composable: calling this on a view branches on its prefix. Immutable, shares the base engine;
     * the base model itself never touches the tree.
     *
     * <p>Typical shape - the base stays cold and stateless; only views touch the tree:
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
                        Mappings.toMessages(prefixMessages),
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
     * this model (or any view sharing its engine) fails with IllegalStateException. Weights are a
     * GC-managed READ_ONLY mmap (kernel-reclaimable; never pin memory) and need no close.
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
        messages.addAll(Mappings.toMessages(request.messages()));
        JinferChatRequestParameters j = p instanceof JinferChatRequestParameters jp ? jp : null;
        List<ChatMessage> requestMessages = request.messages();
        ChatEngine.Request lowered =
                new ChatEngine.Request(
                        messages,
                        cached ? prefix.tools() : Mappings.toTools(requestTools),
                        thinking,
                        p.maxOutputTokens() == null ? -1 : p.maxOutputTokens(),
                        timeoutNanos,
                        p.temperature() == null ? 0.0f : p.temperature().floatValue(),
                        p.topP() == null ? 0.95f : p.topP().floatValue(),
                        j != null && j.seed() != null ? j.seed() : seed,
                        grammar(p, j),
                        p.toolChoice() == ToolChoice.REQUIRED,
                        cached,
                        p.stopSequences());
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

    private static void rejectUnsupported(ChatRequestParameters p) {
        if (p.topK() != null) throw new UnsupportedFeatureException("topK is not supported");
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

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Path modelPath;
        private Path mediaProjector;
        private Path cachedPrompts;
        private int cachedSessions;
        private int contextLength;
        private Double temperature;
        private Double topP;
        private Integer maxOutputTokens;
        private ChatRequestParameters defaultParameters;
        private List<ChatModelListener> listeners = List.of();
        private boolean thinking = true;
        private long seed = 42;
        private Duration timeout;

        /** The GGUF to load. Required. */
        public Builder modelPath(Path modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        /** Mounts a cached-prompt artifact ({@link #saveCachedPrompts}); model-seed-checked. */
        public Builder loadCachedPrompts(Path cachedPrompts) {
            this.cachedPrompts = cachedPrompts;
            return this;
        }

        /** The media sidecar (mmproj GGUF: vision/audio encoders) for multimodal models. */
        public Builder mediaProjector(Path mediaProjector) {
            this.mediaProjector = mediaProjector;
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

        /** Sampling temperature; default 0 (greedy argmax). Per-request values override. */
        public Builder temperature(Double temperature) {
            this.temperature = temperature;
            return this;
        }

        /** Nucleus sampling mass, effective only at temperature &gt; 0; default 0.95. */
        public Builder topP(Double topP) {
            this.topP = topP;
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
         * RNG seed for temperature sampling; default 42. A per-request {@link
         * JinferChatRequestParameters#seed} wins over this. Same seed does NOT guarantee
         * byte-identical replay at temperature &gt; 0: the CPU backend's run-to-run FP jitter flips
         * near-tie samples.
         */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /** Wall-clock deadline per request; unset = none. Exceeding it finishes with LENGTH. */
        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public JinferChatModel build() {
            if (modelPath == null) throw new IllegalArgumentException("modelPath is required");
            return new JinferChatModel(this);
        }
    }
}
