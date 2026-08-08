package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.chat.CachedPrompt;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.TextStops;
import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationRegistry;
import io.micrometer.observation.contextpropagation.ObservationThreadLocalAccessor;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.chat.metadata.EmptyRateLimit;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.MessageAggregator;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.chat.observation.ChatModelObservationConvention;
import org.springframework.ai.chat.observation.ChatModelObservationDocumentation;
import org.springframework.ai.chat.observation.DefaultChatModelObservationConvention;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.ToolCallback;
import reactor.core.publisher.Flux;
import reactor.core.publisher.FluxSink;

/**
 * Spring AI {@link ChatModel} backed by jinfer: in-process CPU inference over a local GGUF.
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

    private static final String PROVIDER = "jinfer";
    private static final ChatModelObservationConvention DEFAULT_CONVENTION =
            new DefaultChatModelObservationConvention();

    /**
     * Core's thought-lane convention: {@link MessageAggregator} accumulates these on {@code
     * thoughts} metadata instead of conflating them with content.
     */
    static final String IS_THOUGHT_KEY = "isThought";

    final ChatEngine engine;
    final JinferChatOptions defaultOptions;
    final ObservationRegistry observationRegistry;
    final ChatModelObservationConvention observationConvention; // null = the default convention
    // cached-prompt view state: EMPTY for the base model. Converted to jinfer types ONCE at view
    // creation (media decoded once, not per request); a view's conversations all start with this
    // prefix, its KV restored from the engine's block tree instead of re-prefilled.
    final com.qxotic.jinfer.media.VideoSampler videoSampler;
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
        if (b.defaultOptions != null && hasKnobs(b)) {
            throw new IllegalArgumentException(
                    "defaultOptions and individual knobs are mutually exclusive");
        }
        this.engine =
                b.loaded == null
                        ? new ChatEngine(
                                b.modelPath,
                                b.companionPaths,
                                cacheOptions(b.cachedPrompts, b.cachedSessions, b.contextLength))
                        : new ChatEngine(
                                b.loaded,
                                b.modelName == null
                                        ? b.loaded.model().getClass().getSimpleName()
                                        : b.modelName,
                                cacheOptions(b.cachedPrompts, b.cachedSessions, b.contextLength));
        this.videoSampler = b.videoSampler;
        this.prefix = CachedPrompt.NONE;
        this.observationRegistry =
                b.observationRegistry == null ? ObservationRegistry.NOOP : b.observationRegistry;
        this.observationConvention = b.observationConvention;
        // precedence: request > builder > the container's recommendation (general.sampling.*)
        // > port author recommendation > the engine baseline (SamplingDefaults.DEFAULT_*)
        var recommended = engine.loaded().samplingDefaults();
        JinferChatOptions knobs =
                JinferChatOptions.builder()
                        .model(engine.modelName())
                        .temperature(
                                b.temperature != null
                                        ? b.temperature
                                        : recommended.temperature() == null
                                                ? null
                                                : recommended.temperature().doubleValue())
                        .topP(
                                b.topP != null
                                        ? b.topP
                                        : recommended.topP() == null
                                                ? null
                                                : recommended.topP().doubleValue())
                        .topK(b.topK != null ? b.topK : recommended.topK())
                        .minP(
                                b.minP != null
                                        ? b.minP
                                        : recommended.minP() == null
                                                ? null
                                                : recommended.minP().doubleValue())
                        .maxTokens(b.maxTokens)
                        .seed(b.seed)
                        .thinking(b.thinking)
                        .timeout(b.timeout)
                        .build();
        if (b.defaultOptions != null) {
            validate(b.defaultOptions);
            this.defaultOptions = b.defaultOptions;
        } else {
            this.defaultOptions = knobs;
        }
    }

    private JinferChatModel(JinferChatModel base, CachedPrompt prefix) {
        this.engine = base.engine;
        this.defaultOptions = base.defaultOptions;
        this.observationRegistry = base.observationRegistry;
        this.observationConvention = base.observationConvention;
        this.videoSampler = base.videoSampler;
        this.prefix = prefix;
    }

    /**
     * A model view whose conversations all start with {@code prefixMessages} (+ welded {@code
     * tools}) - prefilled ONCE into the engine's block tree, restored (not recomputed) on every
     * call. Composable: calling this on a view branches on its prefix. Immutable, shares the base
     * engine; a view's prefix is pinned intent, where the base model's traffic is cached
     * best-effort.
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
            List<org.springframework.ai.chat.messages.Message> prefixMessages,
            List<ToolCallback> tools) {
        return withPrefix(
                prefix.merge(
                        JinferMappings.toMessages(prefixMessages, videoSampler),
                        tools == null ? List.of() : JinferMappings.toTools(tools)));
    }

    private JinferChatModel withPrefix(CachedPrompt merged) {
        engine.define(merged.conversation(defaultOptions.getThinking() != Boolean.FALSE));
        return new JinferChatModel(this, merged);
    }

    /** Freezes every prompt defined so far (plus any mounted base) into one artifact. */
    public void saveCachedPrompts(Path out) {
        engine.freezePrompts(out);
    }

    private static boolean hasKnobs(Builder b) {
        return b.temperature != null
                || b.topP != null
                || b.maxTokens != null
                || b.seed != null
                || b.thinking != null
                || b.timeout != null;
    }

    @Override
    public JinferChatOptions getOptions() {
        return defaultOptions;
    }

    /** Everything the blocking and streaming paths share, computed once per request. */
    /** Framework types mapped away; every policy below this line lives in {@link ChatEngine}. */
    private ChatEngine.Prepared prepare(Prompt prompt) {
        JinferChatOptions options = resolveOptions(prompt.getOptions());
        boolean cached = !prefix.isEmpty();
        List<ToolCallback> callbacks = options.getToolCallbacks();
        if (cached && callbacks != null && !callbacks.isEmpty()) {
            throw new IllegalArgumentException(
                    "a cached-prompt view welds its tools into the cached prefix; per-request"
                            + " toolCallbacks would silently forfeit the cache - put tools on"
                            + " withCachedPrompt(...) instead");
        }
        List<Message> messages = new ArrayList<>(prefix.messages());
        messages.addAll(JinferMappings.toMessages(prompt.getInstructions(), videoSampler));
        List<org.springframework.ai.chat.messages.Message> instructions = prompt.getInstructions();
        ChatEngine.Request lowered =
                new ChatEngine.Request(
                        messages,
                        cached
                                ? prefix.tools()
                                : callbacks == null ? List.of() : JinferMappings.toTools(callbacks),
                        options.getThinking() != Boolean.FALSE,
                        options.getMaxTokens() == null ? -1 : options.getMaxTokens(),
                        null, // Spring AI has no reasoning-budget knob
                        options.getTimeout() == null ? 0 : options.getTimeout().toNanos(),
                        engine.loaded()
                                .samplingDefaults()
                                .resolve(
                                        options.getTemperature() == null
                                                ? null
                                                : options.getTemperature().floatValue(),
                                        options.getTopP() == null
                                                ? null
                                                : options.getTopP().floatValue(),
                                        options.getTopK(),
                                        options.getMinP() == null
                                                ? null
                                                : options.getMinP().floatValue(),
                                        options.getSeed()),
                        grammar(options.getOutputSchema()),
                        null, // Spring AI has no forced-tool-call knob
                        cached,
                        options.getStopSequences(),
                        null); // Spring AI has no chat_template_kwargs equivalent
        return engine.prepare(
                lowered,
                () -> JinferMappings.toMessageMaps(instructions),
                () -> callbacks == null ? List.of() : JinferMappings.toToolMaps(callbacks));
    }

    /**
     * Grammar-constrained output (llama.cpp-style token masking): the schema is compiled to a GBNF
     * grammar whose automaton masks the logits so invalid JSON is unrepresentable, not just
     * unlikely. Compiling it is the framework-shaped half (Spring AI spells a schema as JSON text);
     * the think gating and the dead-end stop token are the engine's. Specs are cached per (schema,
     * vocab), so repeated schemas reuse the compiled masks.
     */
    private Grammar.Spec grammar(String outputSchema) {
        if (outputSchema == null) return null;
        @SuppressWarnings("unchecked")
        Map<String, Object> schemaMap = (Map<String, Object>) JsonCodec.parse(outputSchema);
        return Grammar.fromSchema(schemaMap, engine.loaded().tokenizer());
    }

    @Override
    public ChatResponse call(Prompt prompt) {
        Prompt requestPrompt = buildRequestPrompt(prompt);
        ChatModelObservationContext observationContext =
                ChatModelObservationContext.builder()
                        .prompt(requestPrompt)
                        .provider(PROVIDER)
                        .streaming(false)
                        .build();
        return ChatModelObservationDocumentation.CHAT_MODEL_OPERATION
                .observation(
                        observationConvention,
                        DEFAULT_CONVENTION,
                        () -> observationContext,
                        observationRegistry)
                .observe(
                        () -> {
                            ChatResponse response = doCall(requestPrompt);
                            observationContext.setResponse(response);
                            return response;
                        });
    }

    /**
     * The provider pattern: a request with no options runs on the model's defaults, so the
     * observation span (and any options-reading consumer) sees the effective request, not nulls.
     */
    private Prompt buildRequestPrompt(Prompt prompt) {
        return prompt.getOptions() == null
                ? new Prompt(prompt.getInstructions(), defaultOptions)
                : prompt;
    }

    private ChatResponse doCall(Prompt prompt) {
        ChatEngine.Prepared p = prepare(prompt);
        ChatEngine.Completion done = engine.complete(p, ChatEngine.ReplySink.NONE);
        AssistantMessage ai = JinferMappings.toAssistantMessage(done.reply());
        if (done.stopped()) {
            ai =
                    AssistantMessage.builder()
                            .content(TextStops.apply(ai.getText(), p.stops()).text())
                            .properties(ai.getMetadata())
                            .toolCalls(ai.getToolCalls())
                            .build();
        }
        return response(ai, p, done);
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

    /**
     * Streams delta chunks (text only in each chunk's {@code AssistantMessage}; reasoning deltas
     * are flagged {@code isThought} per core's convention; metadata on the final chunk carries the
     * finish reason, tool calls and usage). Generation runs on the engine's single lazy driver
     * thread; cancellation aborts the pass silently.
     */
    @Override
    public Flux<ChatResponse> stream(Prompt prompt) {
        Prompt requestPrompt = buildRequestPrompt(prompt);
        ChatEngine.Prepared p =
                prepare(requestPrompt); // invalid requests throw here, not on the thread
        // per-subscription observation state: a flux is re-subscribable, and a shared
        // Observation would race on start()/setResponse across subscriptions
        return Flux.deferContextual(
                view -> {
                    ChatModelObservationContext observationContext =
                            ChatModelObservationContext.builder()
                                    .prompt(requestPrompt)
                                    .provider(PROVIDER)
                                    .streaming(true)
                                    .build();
                    Observation observation =
                            ChatModelObservationDocumentation.CHAT_MODEL_OPERATION.observation(
                                    observationConvention,
                                    DEFAULT_CONVENTION,
                                    () -> observationContext,
                                    observationRegistry);
                    observation.parentObservation(
                            (Observation)
                                    view.getOrDefault(ObservationThreadLocalAccessor.KEY, null));
                    observation.start();
                    Flux<ChatResponse> events =
                            Flux.create(sink -> engine.stream(() -> streamInto(p, sink)));
                    return new MessageAggregator()
                            .aggregate(events, observationContext::setResponse)
                            .doOnError(observation::error)
                            .doFinally(signal -> observation.stop())
                            .contextWrite(
                                    c -> c.put(ObservationThreadLocalAccessor.KEY, observation));
                });
    }

    private void streamInto(ChatEngine.Prepared p, FluxSink<ChatResponse> sink) {
        AtomicBoolean cancelled = new AtomicBoolean();
        sink.onCancel(() -> cancelled.set(true));
        sink.onDispose(() -> cancelled.set(true));
        try {
            ChatEngine.Completion done =
                    engine.complete(
                            p,
                            new ChatEngine.ReplySink() {
                                @Override
                                public void content(String delta) {
                                    sink.next(chunk(delta, false));
                                }

                                @Override
                                public void thinking(String delta) {
                                    // reasoning streams too, flagged so consumers can keep it off
                                    // the content lane
                                    sink.next(chunk(delta, true));
                                }

                                @Override
                                public boolean cancelled() {
                                    return cancelled.get();
                                }
                            });
            if (done.cancelled()) return; // cancelled subscriptions end silently
            // the final chunk: no text (deltas carried it), but complete tool calls + metadata,
            // from the same parse that streamed
            AssistantMessage parsed = JinferMappings.toAssistantMessage(done.reply());
            AssistantMessage ai =
                    AssistantMessage.builder()
                            .content("")
                            .properties(parsed.getMetadata())
                            .toolCalls(parsed.getToolCalls())
                            .build();
            sink.next(response(ai, p, done));
            sink.complete();
        } catch (Throwable e) {
            sink.error(e);
        }
    }

    static ChatResponse chunk(String delta, boolean thought) {
        AssistantMessage.Builder<?> b = AssistantMessage.builder().content(delta);
        if (thought) {
            b.properties(Map.of(IS_THOUGHT_KEY, true));
        }
        return new ChatResponse(List.of(new Generation(b.build())));
    }

    private JinferChatOptions resolveOptions(ChatOptions runtime) {
        if (runtime == null) return defaultOptions;
        // runtime options MERGE over the defaults field-by-field, jinfer-typed or foreign -
        // taking jinfer-typed options as-is silently discarded every builder default (an
        // outputSchema-only request ran with an unlimited completion budget)
        JinferChatOptions resolved = JinferChatOptions.copyOnto(defaultOptions, runtime);
        validate(resolved);
        return resolved;
    }

    // topK is NOT rejected here: it is a supported sampling knob (the builder exposes it, the
    // port's recommendation seeds it, the sampler receives it). A guard here once predated that
    // support and, because runtime options merge over the defaults, it rejected EVERY request on
    // a model whose port recommends a top_k - gemma4 does.
    private void validate(JinferChatOptions o) {
        if (o.getFrequencyPenalty() != null)
            throw new IllegalArgumentException("frequencyPenalty is not supported");
        if (o.getPresencePenalty() != null)
            throw new IllegalArgumentException("presencePenalty is not supported");
        if (o.getModel() != null && !o.getModel().equals(engine.modelName()))
            throw new IllegalArgumentException(
                    "per-request model is not supported: this model IS '"
                            + engine.modelName()
                            + "' (one loaded GGUF per instance)");
        if (o.getTimeout() != null && o.getTimeout().isNegative())
            throw new IllegalArgumentException("timeout must not be negative");
        if (o.getOutputSchema() != null
                && o.getToolCallbacks() != null
                && !o.getToolCallbacks().isEmpty())
            throw new IllegalArgumentException(
                    "tools together with an output schema are not supported:"
                            + " grammar-constrained output cannot admit tool-call syntax");
    }

    private ChatResponse response(
            AssistantMessage ai, ChatEngine.Prepared p, ChatEngine.Completion done) {
        Generator.GenerationResult result = done.result();
        String finishReason =
                done.stopped() // a stop-sequence cut IS a stop, not an abort
                        ? "stop"
                        : toFinishReason(result.finishReason(), ai.hasToolCalls());
        Generation generation =
                new Generation(
                        ai, ChatGenerationMetadata.builder().finishReason(finishReason).build());
        // cacheRead: tokens restored from the block tree (0 = uncached request -> null, the
        // observation convention omits nulls); cacheWrite stays null - the tree is written at
        // withCachedPrompt time, never per request. rateLimit: none exists in-process, but the
        // slot must not be null (framework consumers read it provider-agnostically).
        ChatResponseMetadata metadata =
                ChatResponseMetadata.builder()
                        .id(UUID.randomUUID().toString())
                        .model(engine.modelName())
                        .usage(
                                new DefaultUsage(
                                        p.promptTokens(),
                                        result.completionTokens(),
                                        null,
                                        new JinferUsage(
                                                result.promptNanos(), result.predictedNanos()),
                                        done.restoredTokens() > 0
                                                ? Long.valueOf(done.restoredTokens())
                                                : null,
                                        null))
                        .rateLimit(new EmptyRateLimit())
                        .keyValue("prompt-eval-duration", Duration.ofNanos(result.promptNanos()))
                        .keyValue("eval-duration", Duration.ofNanos(result.predictedNanos()))
                        .build();
        return new ChatResponse(List.of(generation), metadata);
    }

    /** Native usage detail: the exact phase timings of the generation pass. */
    public record JinferUsage(long promptNanos, long predictedNanos) {}

    private static String toFinishReason(String jinferReason, boolean hasToolCalls) {
        if (hasToolCalls) return "tool_calls";
        return switch (jinferReason) {
            case "stop" -> "stop";
            case "length" -> "length";
            default -> "other";
        };
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Object source; // Path | ref/URL String | LoadedModel: the last setter wins
        private Path modelPath; // derived from source at build()
        private LoadedModel<?> loaded; // derived from source at build()
        private java.util.Map<String, Path> companionPaths; // resolved at build()
        private String modelName;
        private final java.util.Map<String, String> companions = new java.util.LinkedHashMap<>();
        private com.qxotic.jinfer.media.VideoSampler videoSampler =
                com.qxotic.jinfer.media.VideoSampler.UNIFORM;
        private Path cachedPrompts;
        private int cachedSessions;
        private int contextLength;
        private Double temperature;
        private Double topP;
        private Integer topK;
        private Double minP;
        private Integer maxTokens;
        private Long seed;
        private Boolean thinking;
        private Duration timeout;
        private JinferChatOptions defaultOptions;
        private ObservationRegistry observationRegistry;
        private ChatModelObservationConvention observationConvention;

        /** The GGUF to load. Required unless {@link #model}. */
        public Builder modelPath(Path modelPath) {
            this.source = modelPath;
            return this;
        }

        /**
         * The model as ONE string: a local GGUF path, a hub ref ({@code hf.co/owner/repo:Q4_K_M})
         * or a pasted browser URL - resolved by {@link #build()} with the rest of the load, so a
         * remote ref downloads there (see the package doc) and the chain never blocks.
         */
        public Builder model(String pathOrRef) {
            this.source = pathOrRef;
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
            this.source = loaded;
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
            this.companions.put(capability, file.toString());
            return this;
        }

        /**
         * As {@link #companion(String, Path)}, taking a path, hub ref or URL like {@link
         * #model(String)} - resolved at {@link #build()}.
         */
        public Builder companion(String capability, String pathOrRef) {
            this.companions.put(capability, pathOrRef);
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
         * Keeps the last {@code n} live conversation states resident, reused append-only when a
         * request's conversation strictly extends one (the multi-turn zero-restore tier). Each kept
         * conversation holds a full context of KV.
         *
         * <p>0 (default) keeps the model stateless between requests - its state is wiped the moment
         * a reply ends - but the ALLOCATION is still reused: a pipeline allocates its context once
         * and never per request, whatever this is set to. This knob buys warmth, not memory reuse.
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
         * 0.8. Per-request options override; pass 0 for greedy argmax.
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
         * Top-k cutoff (0 disables); default: the model's recommended value, else 40. Per-request
         * options override.
         */
        public Builder topK(Integer topK) {
            this.topK = topK;
            return this;
        }

        /**
         * Min-p cutoff relative to the top token, in [0,1] (0 disables); default: the model's
         * recommended value, else 0.05. Per-request options override.
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
        public Builder maxTokens(Integer maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        /**
         * RNG seed for temperature sampling; default: a fresh random seed per request. Set one to
         * pin sampling; per-request options override. Same seed does NOT guarantee byte-identical
         * replay at temperature &gt; 0: the CPU backend's run-to-run FP jitter flips near-tie
         * samples.
         */
        public Builder seed(Long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * The model's reasoning scaffold toggle (templates without one ignore it). Default on.
         * Completion budgets below 16 tokens disable it per request regardless - the budget cannot
         * fit a think span.
         */
        public Builder thinking(Boolean thinking) {
            this.thinking = thinking;
            return this;
        }

        /** Wall-clock deadline per request; unset = none. Exceeding it finishes with LENGTH. */
        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        /**
         * Default options for requests that carry none. Mutually exclusive with the individual
         * knobs above; unsupported parameters are rejected eagerly at build.
         */
        public Builder defaultOptions(JinferChatOptions defaultOptions) {
            this.defaultOptions = defaultOptions;
            return this;
        }

        /** Metrics/tracing registry; default {@link ObservationRegistry#NOOP} (zero cost). */
        public Builder observationRegistry(ObservationRegistry observationRegistry) {
            this.observationRegistry = observationRegistry;
            return this;
        }

        /** Custom observation convention; default {@link DefaultChatModelObservationConvention}. */
        public Builder observationConvention(ChatModelObservationConvention observationConvention) {
            this.observationConvention = observationConvention;
            return this;
        }

        public JinferChatModel build() {
            modelPath = null;
            loaded = null;
            if (source == null)
                throw new IllegalArgumentException(
                        "a model is required: model(\"hf.co/owner/repo:Q4_K_M\"),"
                                + " modelPath(...) or model(LoadedModel)");
            if (source instanceof LoadedModel<?> l) {
                if (!companions.isEmpty() || contextLength != 0)
                    throw new IllegalArgumentException(
                            "companions/contextLength are load-time settings; apply them when you"
                                    + " build the LoadedModel passed to model(...)");
                loaded = l;
                companionPaths = java.util.Map.of();
                return new JinferChatModel(this);
            }
            // the model (when it is a string) and the companions resolve in ONE batch, so a cold
            // start pays the slowest download, not the sum
            java.util.List<String> wanted = new java.util.ArrayList<>();
            if (source instanceof String ref) wanted.add(ref);
            wanted.addAll(companions.values());
            java.util.List<Path> resolved = com.qxotic.jinfer.hub.ModelStore.resolveAll(wanted);
            int at = 0;
            modelPath = source instanceof Path path ? path : resolved.get(at++);
            var resolvedCompanions = new java.util.LinkedHashMap<String, Path>();
            for (String capability : companions.keySet()) {
                resolvedCompanions.put(capability, resolved.get(at++));
            }
            companionPaths = java.util.Collections.unmodifiableMap(resolvedCompanions);
            return new JinferChatModel(this);
        }
    }
}
