package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.TextStops;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.codecs.VideoSampler;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.llm.Grammar;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.TokenCountEstimator;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * langchain4j {@link ChatModel} backed by jinfer: in-process CPU inference over a local GGUF.
 * Prompting goes native-first through the model's hand-written, oracle-validated chat-template
 * codec (token-exact, injection-inert) and falls back to a scrubbed Jinja whole-render for unported
 * models or unframeable requests.
 *
 * <p>Concurrency contract: an instance is ONE serial inference pipeline - concurrent requests queue
 * fairly on it. For a second pipeline, load the weights once into YOUR arena ({@code
 * Models.load(path, arena)}), build with {@code model(loaded)}, and {@code fork()} pipelines for
 * the price of a context each. Footprint: an instance holds its weights, up to the configured
 * number of retained full-context states, plus the block layer's KV (every served conversation,
 * best-effort within a 2 GiB LRU-evicted RAM-only budget; defined prompts are pinned intent within
 * it).
 *
 * <p>Lifecycle: base model, cached-prompt views and the streaming twin share ONE engine, so {@code
 * close()} on ANY of them closes ALL - later use of any of them fails with IllegalStateException.
 * The component that built the base model owns {@code close()}; code handed a view or a twin never
 * calls it. ({@code fork()} is the exception: an independent pipeline with its own lifecycle.)
 *
 * <p>Three cache controls with distinct jobs: {@code withCachedPrompt} defines a shared prefix
 * (prefilled once, restored per request - the system-prompt/tools/few-shot case); {@code
 * Builder.retainSessions} keeps finished CONVERSATION states warm for append-only multi-turn reuse
 * (in-RAM, gone at close); {@code saveCachedPrompts}/{@code Builder.promptCache} persist the
 * defined prompts as an immutable artifact that mounts zero-prefill in later processes. None
 * changes output - byte-identity to a cold run is the law. Every response accounts for what the
 * cache did: {@link JinferTokenUsage#cachedInputTokens} is the read, per request.
 *
 * <p>Run with jinfer's JVM flags: {@code --add-modules jdk.incubator.vector
 * --enable-native-access=ALL-UNNAMED}.
 */
public final class JinferChatModel implements ChatModel, AutoCloseable {

    private static final Logger LOG =
            LoggerFactory.getLogger(JinferChatModel.class);

    final ChatEngine engine;
    final ChatRequestParameters defaults;
    final boolean thinking;
    final Duration timeout;
    final List<ChatModelListener> listeners;
    final VideoSampler videoSampler;
    // cached-prompt view state: EMPTY for the base model. Converted to jinfer types ONCE at view
    // creation (media decoded once, not per request); a view's conversations all start with this
    // prefix, its KV restored from the engine's block tree instead of re-prefilled.
    final CachedPrompt prefix;
    private final PromptCache.Options cacheOptions; // the builder's cache knobs, carried for fork()
    private final boolean ownsWeights; // false = the caller loaded the model and keeps the arena
    // per view (never copied): the first tools override warns once, then stays quiet
    private final AtomicBoolean warnedToolsOverride = new AtomicBoolean();

    /**
     * The builder's two cache knobs as the cache's own record. Read-only: this mounts a catalog to
     * SERVE, never to write - a provider embedded in an application must not append to a file the
     * application did not ask it to write.
     */
    private static PromptCache.Options cacheOptions(
            Path promptCache, int retainedSessions, Integer contextLength) {
        var options = PromptCache.Options.DEFAULTS.withRetainedSessions(retainedSessions);
        // unset stays the engine's bounded default (min(4096, model)); an explicit value above
        // the model's context length is refused at build
        if (contextLength != null) options = options.withContextCapacity(contextLength);
        return options.withCatalog(promptCache, true);
    }

    private JinferChatModel(Builder b) {
        // unsupported DEFAULTS reject before anything is mapped: core merges defaults under each
        // request, so a request can ADD what defaults lack (tools curing a REQUIRED default at
        // request time) but can never UNSET a knob this engine does not serve - only the latter
        // is build-fatal, and it need not cost a model load to find out
        if (b.defaultParameters != null) {
            List<Tool> stated = statedTools(b.defaultParameters);
            rejectUnsupported(b.defaultParameters, stated != null && !stated.isEmpty());
        }
        PromptCache.Options requestedCacheOptions =
                cacheOptions(b.promptCache, b.retainedSessions, b.contextLength);
        this.ownsWeights = b.loaded == null;
        this.engine =
                b.loaded == null
                        ? new ChatEngine(b.modelPath, b.companionPaths, requestedCacheOptions)
                        : new ChatEngine(
                                b.loaded,
                                b.modelName == null
                                        ? b.loaded.model().getClass().getSimpleName()
                                        : b.modelName,
                                requestedCacheOptions);
        this.cacheOptions = requestedCacheOptions.withContextCapacity(engine.contextCapacity());
        // the engine above is live (weights mapped) - anything that throws from here on must
        // free it, or a failed build() leaks a GB-scale ofShared arena with no backstop
        try {
            this.thinking = b.thinking;
            this.timeout = b.timeout == null ? Duration.ZERO : b.timeout;
            this.listeners = List.copyOf(b.listeners);
            this.videoSampler = b.videoSampler;
            this.prefix = CachedPrompt.NONE;
            // Jinfer-typed ALWAYS: ChatModel.chat merges defaults.overrideWith(request), and only a
            // jinfer-typed receiver preserves grammar/seed from either side of the merge
            // precedence: request > explicit builder setter > defaultRequestParameters > the
            // container's recommendation > port recommendation > engine baseline
            this.defaults =
                    resolveDefaults(engine.modelName(), engine.loaded().samplingDefaults(), b);
            rejectModelSwitch(engine, this.defaults);
        } catch (RuntimeException | Error e) {
            close(
                    engine::close,
                    e); // frees the engine-owned weights arena; borrowed weights stay alive
            throw e;
        }
    }

    /** Close-on-failure that keeps the ORIGINAL failure primary if close() itself throws. */
    private static void close(Runnable close, Throwable failure) {
        try {
            close.run();
        } catch (RuntimeException | Error e) {
            failure.addSuppressed(e);
        }
    }

    static JinferChatRequestParameters resolveDefaults(
            String model, LoadedModel.SamplingDefaults recommended, Builder b) {
        JinferChatRequestParameters.Builder resolved =
                JinferChatRequestParameters.builder()
                        .modelName(model)
                        .temperature(toDouble(recommended.temperature()))
                        .topP(toDouble(recommended.topP()))
                        .topK(recommended.topK())
                        .minP(toDouble(recommended.minP()));
        if (b.defaultParameters != null) resolved.overrideWith(b.defaultParameters);
        // LangChain4j's provider builders expose both idiomatic convenience setters and a
        // fallback parameters object. Explicit setters are the narrower, later provenance.
        if (b.temperature != null) resolved.temperature(b.temperature);
        if (b.topP != null) resolved.topP(b.topP);
        if (b.topK != null) resolved.topK(b.topK);
        if (b.minP != null) resolved.minP(b.minP);
        if (b.maxOutputTokens != null) resolved.maxOutputTokens(b.maxOutputTokens);
        if (b.seed != null) resolved.seed(b.seed);
        return resolved.build();
    }

    /** The fork constructor: a fresh engine over the same borrowed weights, every knob carried. */
    private JinferChatModel(JinferChatModel base, ChatEngine engine) {
        this.engine = engine;
        this.defaults = base.defaults;
        this.thinking = base.thinking;
        this.timeout = base.timeout;
        this.listeners = base.listeners;
        this.videoSampler = base.videoSampler;
        this.cacheOptions = base.cacheOptions;
        this.ownsWeights = false;
        this.prefix = CachedPrompt.NONE;
    }

    /**
     * A parallel pipeline over the same weights: fresh engine, state and stream driver, every
     * builder knob carried (a mounted cached-prompts artifact is re-mounted read-only; a view's
     * prefix is re-defined on the fork's own tree). Only a model whose weights YOU loaded can fork
     * - the weights' lifetime is your arena's, so a fork can never dangle. A model that loaded its
     * own weights refuses: it frees them at {@link #close()}, and a fork would outlive them.
     */
    public JinferChatModel fork() {
        if (ownsWeights) {
            throw new IllegalStateException(
                    "this model owns its weights and frees them at close - a fork would dangle."
                            + " Load once into YOUR arena instead: Models.load(path, arena), build"
                            + " with model(loaded), then fork freely");
        }
        JinferChatModel forked =
                new JinferChatModel(
                        this, new ChatEngine(engine.loaded(), engine.modelName(), cacheOptions));
        if (prefix.isEmpty()) return forked;
        try {
            return forked.withPrefix(prefix);
        } catch (RuntimeException | Error e) {
            close(forked::close, e); // a failed re-define must not leak the fork's engine
            throw e;
        }
    }

    private JinferChatModel(JinferChatModel base, CachedPrompt prefix) {
        this.engine = base.engine;
        this.defaults = base.defaults;
        this.thinking = base.thinking;
        this.timeout = base.timeout;
        this.listeners = base.listeners;
        this.videoSampler = base.videoSampler;
        this.cacheOptions = base.cacheOptions;
        this.ownsWeights = base.ownsWeights;
        this.prefix = prefix;
    }

    /**
     * A model view whose conversations all start with {@code prefixMessages}, offering {@code
     * tools} as the view's DEFAULT tool set - both prefilled ONCE into the engine's block tree,
     * restored (not recomputed) on every chat. Composable: calling this on a view branches on its
     * prefix. Immutable, shares the base engine; a view's prefix is pinned intent, where the base
     * model's traffic is cached best-effort. Shares the base's lifecycle too: {@code close()} on
     * either closes both (see the class doc's Lifecycle paragraph).
     *
     * <p>Tools follow the standard parameter precedence, request over defaults: a request that
     * states none offers the welded set (an AiServices agent re-stating the SAME set lands on the
     * cache too); a request that states a different set (or {@code toolChoice NONE}) is served with
     * ITS tools, byte-identical to the base model - a cache changes latency, never behavior - but
     * forfeits the prepaid prefill for that call. {@link JinferTokenUsage#cachedInputTokens} on
     * every response tells which happened; the first override also warns once on stderr.
     *
     * <p>There is deliberately NO messages-only overload: the prepaid frame includes the tool
     * declarations, so passing {@code List.of()} is the caller acknowledging this view welds no
     * tools - the cache is not tools-independent, and the signature should not suggest it is.
     *
     * <p>(The tree serves the BASE model too: every conversation on a codec model is resumed from
     * and committed to it, best-effort within the block budget. Defined views still work through an
     * explicitly mounted artifact.)
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
        Objects.requireNonNull(prefixMessages, "prefixMessages");
        Objects.requireNonNull(tools, "tools");
        return withPrefix(
                prefix.merge(
                        Mappings.toMessages(prefixMessages, videoSampler),
                        Mappings.toTools(tools)));
    }

    private JinferChatModel withPrefix(CachedPrompt merged) {
        framed(() -> engine.definePrompt(merged.conversation(thinking)));
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
    // shared engine - any close() closes the base, every view and the streaming twin;
    // if independent closing is ever a demonstrated need, explicit engine owner + non-owning views
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
    public Set<Capability> supportedCapabilities() {
        // grammar-constrained decoding honors JSON schemas natively - AiServices reads this to
        // use structured output for POJO extraction instead of prompt-based JSON begging
        return Set.of(Capability.RESPONSE_FORMAT_JSON_SCHEMA);
    }

    @Override
    public ChatResponse doChat(ChatRequest request) {
        try (ChatEngine.Prepared p = prepare(request)) {
            ChatEngine.Completion done = engine.complete(p, ChatEngine.ReplySink.NONE);
            AiMessage ai = Mappings.toAiMessage(done.reply());
            if (done.stopped()) {
                ai = Mappings.withText(ai, TextStops.apply(ai.text(), p.stops()).text());
            }
            return Mappings.response(engine.modelName(), ai, done.promptTokens(), done);
        }
    }

    /**
     * Token counting over THIS model's tokenizer: exact on text, message counts summing visible
     * text (chat scaffold excluded - a few percent that consumer margins absorb). For {@code
     * TokenWindowChatMemory} budgets and token-aware splitters.
     */
    public TokenCountEstimator tokenCountEstimator() {
        var template = engine.loaded().template().orElse(null);
        return new Estimators(
                engine.loaded().tokenizer(),
                template == null ? null : template::mediaPositions,
                videoSampler);
    }

    /** A streaming twin sharing this model's engine and cached prefix (the GGUF loads once). */
    public JinferStreamingChatModel streaming() {
        return new JinferStreamingChatModel(this);
    }

    // ---- shared request preparation (also used by the streaming twin) ----

    /** Every request-shape rejection, synchronously; both entry points reach it via prepare(). */
    void validate(ChatRequestParameters p, List<Tool> effectiveTools) {
        rejectUnsupported(p, !effectiveTools.isEmpty());
        rejectModelSwitch(engine, p);
        if (p.toolChoice() == ToolChoice.REQUIRED && effectiveTools.isEmpty()) {
            throw new IllegalArgumentException("toolChoice REQUIRED without any tools");
        }
    }

    /** The request's stated tool set: NONE = explicitly none; null = unstated, defaults apply. */
    private static List<Tool> statedTools(ChatRequestParameters p) {
        if (p.toolChoice() == ToolChoice.NONE) return List.of();
        return p.toolSpecifications() == null || p.toolSpecifications().isEmpty()
                ? null
                : Mappings.toTools(p.toolSpecifications());
    }

    /** Framework types mapped away; every policy below this line lives in {@link ChatEngine}. */
    ChatEngine.Prepared prepare(ChatRequest request) {
        ChatRequestParameters p = request.parameters();
        // request > view default (CachedPrompt.resolveTools, THE precedence rule): an override
        // is served correctly (byte-identical to the base model) at full prefill - a cache
        // changes latency, never behavior - and warns once so a wiring bug stays discoverable
        List<Tool> tools = prefix.resolveTools(statedTools(p));
        validate(p, tools);
        boolean cached = prefix.serves(tools);
        if (!cached && !prefix.isEmpty() && warnedToolsOverride.compareAndSet(false, true)) {
            LOG.warn(prefix.toolsOverrideWarning(tools));
        }
        List<Message> messages = new ArrayList<>(prefix.messages());
        // a schema constrains the SHAPE through the grammar below; stating says its MEANING,
        // which no local model can infer from a mask - to the REQUEST's messages only, a cached
        // prefix keeps its bytes (see Mappings.stating). The engine derives the fallback's
        // render maps from this same conversation, so both encode paths state it identically.
        Map<String, Object> schema = schemaOf(p);
        messages.addAll(
                Mappings.stating(
                        Mappings.toMessages(request.messages(), videoSampler),
                        schema,
                        !tools.isEmpty()));
        JinferChatRequestParameters j = p instanceof JinferChatRequestParameters jp ? jp : null;
        ChatEngine.Request lowered =
                new ChatEngine.Request(
                        messages,
                        tools,
                        thinking,
                        p.maxOutputTokens() == null ? -1 : p.maxOutputTokens(),
                        null, // langchain4j has no reasoning-budget knob
                        null, // nor a reasoning-message one
                        timeout,
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
                                        j == null ? null : j.seed()),
                        contentGbnf(p, j, schema, !tools.isEmpty()),
                        p.toolChoice() == ToolChoice.REQUIRED
                                ? ChatEngine.ForcedTool.ANY
                                : ChatEngine.ForcedTool.NONE,
                        p.stopSequences(),
                        null); // langchain4j has no chat_template_kwargs equivalent
        return framed(() -> engine.prepare(lowered));
    }

    /** The request's JSON schema as a plain map, or null when it carries none. */
    private static Map<String, Object> schemaOf(ChatRequestParameters p) {
        ResponseFormat rf = p.responseFormat();
        if (rf == null || rf.type() != ResponseFormatType.JSON || rf.jsonSchema() == null)
            return null;
        return Mappings.toSchemaMap(rf.jsonSchema().rootElement());
    }

    /**
     * The request's decoding constraint as GBNF SOURCE, or null - the one currency every
     * constrained chat decode speaks: a typed schema (tools present = the leading-ws-free
     * content-hole form), schemaless JSON mode, or raw GBNF. The engine compiles it into the
     * family's constrained selection; specs cache per (source, vocab), so repeated schemas reuse
     * the compiled masks.
     */
    private static String contentGbnf(
            ChatRequestParameters p,
            JinferChatRequestParameters j,
            Map<String, Object> schema,
            boolean toolsOffered) {
        if (schema != null) {
            return toolsOffered ? Grammar.schemaHoleGbnf(schema) : Grammar.schemaGbnf(schema);
        }
        ResponseFormat rf = p.responseFormat();
        if (rf != null && rf.type() == ResponseFormatType.JSON) return Grammar.jsonGbnf();
        // raw GBNF: the JSON format's generalization (validate() guaranteed they are not combined)
        return j == null ? null : j.grammar();
    }

    private static <T> T framed(Supplier<T> op) {
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
    private static void rejectUnsupported(ChatRequestParameters p, boolean tools) {
        if (p.frequencyPenalty() != null)
            throw new UnsupportedFeatureException("frequencyPenalty is not supported");
        if (p.presencePenalty() != null)
            throw new UnsupportedFeatureException("presencePenalty is not supported");
        ResponseFormat rf = p.responseFormat();
        if (rf != null && rf.type() == ResponseFormatType.JSON && tools) {
            // WITH a schema the two compose: the schema rides the family's reply language
            // (calls stay the family's own syntax, visible text can only be the schema).
            // Schemaless JSON has no language to state, and a FORCED call plus a schema-shaped
            // answer cannot both be THE reply - those two stay loud.
            if (rf.jsonSchema() == null)
                throw new UnsupportedFeatureException(
                        "tools together with schemaless JSON format are not supported: state a"
                                + " schema, the composed selection needs one");
            if (p.toolChoice() == ToolChoice.REQUIRED)
                throw new UnsupportedFeatureException(
                        "toolChoice REQUIRED together with a JSON response format is not"
                                + " supported: a forced call and a schema-shaped answer cannot"
                                + " both be the reply");
        }
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
        private Object source; // Path | model-ref String | LoadedModel: the last setter wins
        private Path modelPath; // derived from source at build()
        private LoadedModel<?> loaded; // derived from source at build()
        private Map<String, Path> companionPaths; // resolved at build()
        private String modelName;
        private final Map<String, String> companionRefs = new LinkedHashMap<>();
        private final Map<String, Path> localCompanions = new LinkedHashMap<>();
        private VideoSampler videoSampler = VideoSampler.UNIFORM;
        private Path promptCache;
        private int retainedSessions = 1;
        private Integer contextLength; // null = unset -> min(4096, model)
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
            this.source = modelPath;
            return this;
        }

        /**
         * The model as a model ref, resolved - downloading to the local cache on first use - by
         * {@link #build()}.
         *
         * <pre>{@code
         * model("hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0");
         * }</pre>
         *
         * <p>The full grammar - the default quant, pinned revisions, a file inside a repository,
         * ModelScope - is documented once in {@link com.qxotic.jinfer.hub.ModelRef}. For a file
         * already on disk use {@link #modelPath(Path)}. A URL is not a model ref: download it
         * first, then pass the path.
         */
        public Builder model(String modelRef) {
            ModelStore.requireRef(modelRef);
            this.source = modelRef;
            return this;
        }

        /**
         * A model you loaded yourself. Its weights arena stays yours; close the arena only after
         * this model and every fork on it.
         */
        public Builder model(LoadedModel<?> loaded) {
            this.source = loaded;
            return this;
        }

        /**
         * Reported as the response's model name; defaults to the model class. Applies to {@code
         * model(LoadedModel)} only - path and ref loads name themselves from the file.
         */
        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        /**
         * Mounts one existing cached-prompt artifact read-only; missing or incompatible artifacts
         * fail the build loudly.
         */
        public Builder promptCache(Path promptCache) {
            this.promptCache = Objects.requireNonNull(promptCache, "promptCache");
            return this;
        }

        /**
         * How video content becomes frames - default {@link VideoSampler#UNIFORM} (the reference
         * policy: 32 frames uniform across the whole duration). Any policy composes: {@code v ->
         * VideoCodec.ffmpeg().span(v, 8)}, a window of a long source, caller-curated timestamps.
         */
        public Builder videoSampler(VideoSampler videoSampler) {
            this.videoSampler = Objects.requireNonNull(videoSampler);
            return this;
        }

        /** Attaches a local companion file. This method never touches the network. */
        public Builder companionPath(String capability, Path companionPath) {
            Objects.requireNonNull(capability, "capability");
            Objects.requireNonNull(companionPath, "companionPath");
            companionRefs.remove(capability);
            localCompanions.put(capability, companionPath);
            return this;
        }

        /**
         * Attaches a companion from a supported model repository. The reference is resolved at
         * {@link #build()}.
         */
        public Builder companion(String capability, String companionRef) {
            Objects.requireNonNull(capability, "capability");
            if (!ModelStore.isRef(companionRef)) {
                throw new IllegalArgumentException(
                        "'"
                                + companionRef
                                + "' is not a companion model ref. Use companionPath(...) for a"
                                + " local file; download plain URLs first.");
            }
            localCompanions.remove(capability);
            companionRefs.put(capability, companionRef);
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
         * <p>The default is 1. Zero retains no completed state: every request's state is closed
         * when the request ends and the next request allocates a fresh one. This does not disable
         * the separate block cache.
         */
        public Builder retainSessions(int retainedSessions) {
            if (retainedSessions < 0)
                throw new IllegalArgumentException("retainedSessions " + retainedSessions);
            this.retainedSessions = retainedSessions;
            return this;
        }

        /**
         * Upper bound on the context available to each conversation, in tokens. The default is
         * min(4096, the model's context length), deliberately bounded because a full-context state
         * can consume substantial memory. A value above the model's context length is refused at
         * build; {@code 0} asks for the model's maximum. {@code 0} uses the model's declared
         * context length; otherwise the effective capacity is the smaller of this value and that
         * length.
         *
         * @throws IllegalArgumentException if {@code contextLength < 0}
         */
        public Builder contextLength(int contextLength) {
            if (contextLength < 0)
                throw new IllegalArgumentException(
                        "contextLength must be >= 0 (0 uses the model maximum): " + contextLength);
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
            this.listeners = List.copyOf(listeners); // a null list fails here, not after the load
            return this;
        }

        /**
         * Fallback request parameters. Explicit builder setters override these, and each request
         * overrides both (standard langchain4j semantics). Unsupported parameters are rejected
         * eagerly at build.
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
            modelPath = null;
            loaded = null;
            if (source == null)
                throw new IllegalArgumentException(
                        "a model is required: model(\"hf.co/owner/repo:Q4_K_M\"),"
                                + " modelPath(...) or model(LoadedModel)");
            if (promptCache != null && !Files.isRegularFile(promptCache)) {
                throw new IllegalArgumentException("prompt cache does not exist: " + promptCache);
            }
            if (source instanceof LoadedModel<?> l) {
                // contextLength stays legal here: state capacity is an ENGINE setting resolved
                // from cacheOptions, not a load-time one - a forked 32k pipeline needs it
                if (!companionRefs.isEmpty() || !localCompanions.isEmpty())
                    throw new IllegalArgumentException(
                            "companions are load-time settings; apply them when you build the"
                                    + " LoadedModel passed to model(...)");
                loaded = l;
                companionPaths = Map.of();
                return new JinferChatModel(this);
            }
            // the model (when it is a string) and the companions resolve in ONE batch, so a cold
            // start pays the slowest download, not the sum
            List<String> wanted = new ArrayList<>();
            if (source instanceof String ref) wanted.add(ref);
            wanted.addAll(companionRefs.values());
            List<Path> resolved = ModelStore.standard().resolveAll(wanted);
            int at = 0;
            modelPath = source instanceof Path path ? path : resolved.get(at++);
            var resolvedCompanions = new LinkedHashMap<>(localCompanions);
            for (String capability : companionRefs.keySet()) {
                resolvedCompanions.put(capability, resolved.get(at++));
            }
            companionPaths = Collections.unmodifiableMap(resolvedCompanions);
            return new JinferChatModel(this);
        }

        /** As {@link #build()}, returning the streaming face - one load, shared engine. */
        public JinferStreamingChatModel buildStreaming() {
            return build().streaming();
        }
    }
}
