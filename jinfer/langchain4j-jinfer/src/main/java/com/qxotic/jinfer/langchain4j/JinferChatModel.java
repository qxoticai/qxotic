package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.ReplyLanes;
import com.qxotic.jinfer.chat.RequestPolicy;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.TextStops;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.internal.JsonSchemaElementUtils;
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
 * fairly on it. For parallel pipelines, build several instances: same-model instances share weights
 * through the OS page cache (read-only mmap), so each extra instance costs only its own mutable
 * state (KV, caches, sessions), not another copy of the model.
 *
 * <p>Run with jinfer's JVM flags: {@code --enable-preview --add-modules jdk.incubator.vector
 * --enable-native-access=ALL-UNNAMED}.
 */
public final class JinferChatModel implements ChatModel, AutoCloseable {

    final JinferEngine engine;
    final ChatRequestParameters defaults;
    final boolean thinking;
    final long seed;
    final long timeoutNanos;
    final List<ChatModelListener> listeners;
    // cached-prompt view state: EMPTY for the base model. Converted to jinfer types ONCE at view
    // creation (media decoded once, not per request); a view's conversations all start with this
    // prefix, its KV restored from the engine's block tree instead of re-prefilled.
    final CachedPrompt prefix;

    /** A view's prefix in jinfer types; {@link #EMPTY} for the base model. */
    record CachedPrompt(List<Message> messages, List<Tool> tools) {
        static final CachedPrompt EMPTY = new CachedPrompt(List.of(), List.of());

        boolean isEmpty() {
            return messages.isEmpty() && tools.isEmpty();
        }
    }

    private JinferChatModel(Builder b) {
        this.engine =
                new JinferEngine(
                        b.modelPath,
                        b.mediaProjector,
                        b.contextLength,
                        b.cachedPrompts,
                        b.cachedSessions);
        this.thinking = b.thinking;
        this.seed = b.seed;
        this.timeoutNanos = b.timeout == null ? 0 : b.timeout.toNanos();
        this.listeners = List.copyOf(b.listeners);
        this.prefix = CachedPrompt.EMPTY;
        // Jinfer-typed ALWAYS: ChatModel.chat merges defaults.overrideWith(request), and only a
        // jinfer-typed receiver preserves grammar/seed from either side of the merge
        JinferChatRequestParameters base =
                JinferChatRequestParameters.builder()
                        .modelName(engine.modelName)
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
     */
    public JinferChatModel withCachedPrompt(
            List<ChatMessage> prefixMessages, List<ToolSpecification> tools) {
        List<Message> messages = new ArrayList<>(prefix.messages());
        messages.addAll(Mappings.toMessages(prefixMessages)); // converted ONCE, media decoded here
        List<Tool> welded = new ArrayList<>(prefix.tools());
        if (tools != null) {
            welded.addAll(Mappings.toTools(tools));
        }
        CachedPrompt merged = new CachedPrompt(List.copyOf(messages), List.copyOf(welded));
        engine.define(new Conversation(merged.messages(), merged.tools(), thinking, ""));
        return new JinferChatModel(this, merged);
    }

    /** Freezes every prompt defined so far (plus any mounted base) into one artifact. */
    public void saveCachedPrompts(Path out) {
        engine.freezePrompts(out);
    }

    /**
     * Releases the engine's cached-prompt blobs and pooled session states; later use of this model
     * (or any view sharing its engine) fails with IllegalStateException. Idempotent. Weights still
     * release with reachability - close() is for eagerly dropping cache memory.
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
        Prepared p = prepare(this, request);
        List<String> stops = p.stops() == null ? List.of() : p.stops();
        TextStops.Holdback watch =
                stops.isEmpty() ? null : new TextStops.Holdback(stops, ignored -> {});
        ReplyLanes lanes =
                new ReplyLanes(p.encoded().template(), engine.loaded.tokenizer(), p.parserSeed());
        Generator.TokenSink sink =
                token -> {
                    String fragment = lanes.feed(token);
                    if (watch != null && !lanes.reasoning() && !fragment.isEmpty()) {
                        watch.accept(fragment); // stop strings match the content lane only
                    }
                    return watch == null || !watch.stopped();
                };
        Generator.GenerationResult result =
                engine.generate(
                                p.encoded().prompt(),
                                p.sampler(),
                                p.maxTokens(),
                                timeoutNanos,
                                sink,
                                p.cached())
                        .result();
        AiMessage ai = Mappings.toAiMessage(lanes.finish());
        boolean stopHit = watch != null && watch.stopped();
        if (stopHit) {
            ai = Mappings.withText(ai, TextStops.apply(ai.text(), stops).text());
        }
        return Mappings.response(engine.modelName, ai, p.promptTokens(), result, stopHit);
    }

    /**
     * Token counting over THIS model's tokenizer: exact on text, message counts summing visible
     * text (chat scaffold excluded - a few percent that consumer margins absorb). For {@code
     * TokenWindowChatMemory} budgets and token-aware splitters.
     */
    public dev.langchain4j.model.TokenCountEstimator tokenCountEstimator() {
        var template = engine.loaded.template().orElse(null);
        return new Estimators(
                engine.loaded.tokenizer(), template == null ? null : template::mediaPositions);
    }

    /** A streaming twin sharing this model's engine and cached prefix (the GGUF loads once). */
    public JinferStreamingChatModel streaming() {
        return new JinferStreamingChatModel(this);
    }

    // ---- shared request preparation (also used by the streaming twin) ----

    record Prepared(
            ChatEngine.Encoded encoded,
            Sampler sampler,
            int maxTokens,
            int promptTokens,
            boolean cached,
            int[] parserSeed,
            List<String> stops) {}

    /** Every request-shape rejection, synchronously (streaming calls this before its thread). */
    static void validate(JinferChatModel m, ChatRequest request) {
        ChatRequestParameters p = request.parameters();
        rejectUnsupported(p);
        rejectModelSwitch(m.engine, p);
        boolean requestHasTools =
                p.toolSpecifications() != null && !p.toolSpecifications().isEmpty();
        if (!m.prefix.isEmpty() && requestHasTools) {
            throw new UnsupportedFeatureException(
                    "a cached-prompt view welds its tools into the cached prefix; per-request"
                            + " toolSpecifications would silently forfeit the cache - put tools on"
                            + " withCachedPrompt(...) instead");
        }
        if (p.toolChoice() == ToolChoice.REQUIRED
                && !requestHasTools
                && m.prefix.tools().isEmpty()) {
            throw new IllegalArgumentException("toolChoice REQUIRED without any tools");
        }
        if (p.toolChoice() == ToolChoice.NONE && !m.prefix.tools().isEmpty()) {
            throw new UnsupportedFeatureException(
                    "toolChoice NONE on a cached-prompt view is not supported: the view's tools"
                            + " are welded into its cached prefix and cannot be un-offered");
        }
    }

    static Prepared prepare(JinferChatModel m, ChatRequest request) {
        validate(m, request);
        ChatRequestParameters p = request.parameters();
        boolean cached = !m.prefix.isEmpty();
        // NONE = the model cannot use tools: never offer them, and there is nothing to call
        List<ToolSpecification> requestTools =
                p.toolChoice() == ToolChoice.NONE || p.toolSpecifications() == null
                        ? List.of()
                        : p.toolSpecifications();
        JinferEngine engine = m.engine;
        int maxTokens = p.maxOutputTokens() == null ? -1 : p.maxOutputTokens();
        boolean required = p.toolChoice() == ToolChoice.REQUIRED;
        // a think span cannot fit a tiny completion budget: below the floor, reasoning is
        // disabled outright (scaffold and sampler both) so the budget buys VISIBLE text.
        // A forced call also skips thinking: the reply is seeded INTO the call block.
        boolean think = m.thinking && (maxTokens < 0 || maxTokens >= 16) && !required;
        List<Message> messages = new ArrayList<>(m.prefix.messages());
        messages.addAll(Mappings.toMessages(request.messages()));
        Conversation conversation =
                new Conversation(
                        messages,
                        cached ? m.prefix.tools() : Mappings.toTools(requestTools),
                        think,
                        "");
        // cached views are native-only (define enforced it); the base keeps the Jinja fallback
        ChatEngine.Encoded encoded =
                cached
                        ? engine.encodeNative(conversation)
                        : engine.encode(conversation, request.messages(), requestTools);
        Sampler sampler = sampler(m, p, think, maxTokens);
        // the parser pre-feed: the generation prompt's reply-grammar tail (a prompt-opened think
        // span); a forced call replaces it with the recipe's own (reply seeded into the call block)
        int[] parserSeed = encoded.template().map(t -> t.replySeed(think)).orElse(new int[0]);
        if (required) {
            // the shared recipe: seed the family's call marker into the prompt, prefix-pin the
            // offered names + header epilogue, pre-feed the parser - one unsplittable value
            RequestPolicy.ForcedCall f =
                    RequestPolicy.forceCall(engine.loaded, conversation.tools(), sampler)
                            .orElseThrow(
                                    () ->
                                            new UnsupportedFeatureException(
                                                    "ToolChoice.REQUIRED is not supported by this"
                                                        + " model: forcing seeds the reply with the"
                                                        + " family's call marker, which needs a"
                                                        + " native codec that declares one"));
            List<Batch> prompt = new ArrayList<>(encoded.prompt());
            prompt.add(f.seed());
            encoded = new ChatEngine.Encoded(List.copyOf(prompt), encoded.template());
            sampler = f.sampler();
            parserSeed = f.parserSeed();
        }
        int promptTokens = encoded.prompt().stream().mapToInt(Batch::count).sum();
        return new Prepared(
                encoded, sampler, maxTokens, promptTokens, cached, parserSeed, p.stopSequences());
    }

    /**
     * The request's sampling stack via the shared {@link ChatEngine} policy, plus
     * grammar-constrained JSON when the request asks for it (schema compiled here - the framework
     * conversion is this adapter's; specs are cached per grammar source, so repeated schemas reuse
     * the compiled masks).
     */
    private static Sampler sampler(
            JinferChatModel m, ChatRequestParameters p, boolean think, int maxTokens) {
        var loaded = m.engine.loaded;
        JinferChatRequestParameters j = p instanceof JinferChatRequestParameters jp ? jp : null;
        long seed = j != null && j.seed() != null ? j.seed() : m.seed;
        Sampler sampler =
                RequestPolicy.sampler(
                        loaded,
                        p.temperature() == null ? 0.0f : p.temperature().floatValue(),
                        p.topP() == null ? 0.95f : p.topP().floatValue(),
                        seed,
                        think,
                        maxTokens,
                        null);
        ResponseFormat rf = p.responseFormat();
        if (rf != null && rf.type() == ResponseFormatType.JSON) {
            Grammar.Spec spec =
                    rf.jsonSchema() == null
                            ? Grammar.json(loaded.tokenizer())
                            : Grammar.fromSchema(
                                    JsonSchemaElementUtils.toMap(rf.jsonSchema().rootElement()),
                                    loaded.tokenizer());
            sampler = RequestPolicy.constrained(loaded, sampler, spec.cursor(), think);
        } else if (j != null && j.grammar() != null) {
            // raw GBNF - the JSON format's generalization, same think gating (validate()
            // guaranteed the two are not combined); specs cache by source, repeats are free
            sampler =
                    RequestPolicy.constrained(
                            loaded,
                            sampler,
                            Grammar.of(j.grammar(), loaded.tokenizer()).cursor(),
                            think);
        }
        return sampler;
    }

    /** One loaded GGUF per instance: a different {@code modelName} cannot be served. */
    private static void rejectModelSwitch(JinferEngine engine, ChatRequestParameters p) {
        if (p.modelName() != null && !p.modelName().equals(engine.modelName)) {
            throw new UnsupportedFeatureException(
                    "per-request modelName is not supported: this model IS '"
                            + engine.modelName
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
         * is byte-identical to a cold run, nothing persists, and the default 0 keeps the model
         * fully stateless. Each kept state holds a full context of KV.
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

        public Builder temperature(Double temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder topP(Double topP) {
            this.topP = topP;
            return this;
        }

        public Builder maxOutputTokens(Integer maxOutputTokens) {
            this.maxOutputTokens = maxOutputTokens;
            return this;
        }

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

        /** The model's reasoning scaffold toggle (templates without one ignore it). Default on. */
        public Builder thinking(boolean thinking) {
            this.thinking = thinking;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

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
