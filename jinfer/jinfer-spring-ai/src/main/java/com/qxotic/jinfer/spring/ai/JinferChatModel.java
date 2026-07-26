package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.ReplyLanes;
import com.qxotic.jinfer.chat.RequestPolicy;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.Sampler;
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
 * fairly on it. For parallel pipelines call {@code copy()}: siblings share the already-loaded model
 * (zero reload, no extra weight memory) and each owns its own serial pipeline. Footprint: an
 * instance holds its weights (shared across copies), at most {@code cachedSessions} full-context
 * states (recycled - extended on a prefix hit, reset on a miss, never re-allocated per request),
 * plus one KV block set per defined cached prompt (explicit and deliberately paid for).
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
    final CachedPrompt prefix;

    /** A view's prefix in jinfer types; {@link #EMPTY} for the base model. */
    record CachedPrompt(List<Message> messages, List<Tool> tools) {
        static final CachedPrompt EMPTY = new CachedPrompt(List.of(), List.of());

        boolean isEmpty() {
            return messages.isEmpty() && tools.isEmpty();
        }
    }

    private JinferChatModel(Builder b) {
        if (b.defaultOptions != null && hasKnobs(b)) {
            throw new IllegalArgumentException(
                    "defaultOptions and individual knobs are mutually exclusive");
        }
        this.engine =
                new ChatEngine(
                        b.modelPath,
                        b.mediaProjector,
                        b.contextLength,
                        b.cachedPrompts,
                        b.cachedSessions);
        this.prefix = CachedPrompt.EMPTY;
        this.observationRegistry =
                b.observationRegistry == null ? ObservationRegistry.NOOP : b.observationRegistry;
        this.observationConvention = b.observationConvention;
        JinferChatOptions knobs =
                JinferChatOptions.builder()
                        .model(engine.modelName())
                        .temperature(b.temperature)
                        .topP(b.topP)
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

    private JinferChatModel(JinferChatModel base, ChatEngine fork) {
        this.engine = fork;
        this.defaultOptions = base.defaultOptions;
        this.observationRegistry = base.observationRegistry;
        this.observationConvention = base.observationConvention;
        this.prefix = CachedPrompt.EMPTY;
    }

    private JinferChatModel(JinferChatModel base, CachedPrompt prefix) {
        this.engine = base.engine;
        this.defaultOptions = base.defaultOptions;
        this.observationRegistry = base.observationRegistry;
        this.observationConvention = base.observationConvention;
        this.prefix = prefix;
    }

    /**
     * A model view whose conversations all start with {@code prefixMessages} (+ welded {@code
     * tools}) - prefilled ONCE into the engine's block tree, restored (not recomputed) on every
     * call. Composable: calling this on a view branches on its prefix. Immutable, shares the base
     * engine; the base model itself never touches the tree.
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
            List<org.springframework.ai.chat.messages.Message> prefixMessages,
            List<ToolCallback> tools) {
        List<Message> messages = new ArrayList<>(prefix.messages());
        messages.addAll(
                JinferMappings.toMessages(prefixMessages)); // converted ONCE, media decoded here
        List<Tool> welded = new ArrayList<>(prefix.tools());
        if (tools != null) {
            welded.addAll(JinferMappings.toTools(tools));
        }
        CachedPrompt merged = new CachedPrompt(List.copyOf(messages), List.copyOf(welded));
        engine.define(
                new Conversation(
                        merged.messages(),
                        merged.tools(),
                        defaultOptions.getThinking() != Boolean.FALSE,
                        ""));
        return new JinferChatModel(this, merged);
    }

    /**
     * An independent sibling of this model: shares the loaded weights/tokenizer/template (nothing
     * reloads, no extra weight memory) but owns its OWN serial inference pipeline - lock, caches,
     * stream driver. THE way to run several pipelines of one model in parallel. A copy's lifecycle
     * is independent: closing it never affects the base or other copies (views, by contrast, share
     * their creator's engine - closing any of them closes all). Cached-prompt definitions are not
     * carried over, but a MOUNTED artifact is (the frozen tier is immutable, shared safely);
     * options and conventions are. A copy of a VIEW re-defines its prefix on the fresh pipeline
     * (one prefill).
     */
    public JinferChatModel copy() {
        JinferChatModel base = new JinferChatModel(this, engine.fork());
        return prefix.isEmpty() ? base : base.withPrefix(prefix);
    }

    private JinferChatModel withPrefix(CachedPrompt merged) {
        engine.define(
                new Conversation(
                        merged.messages(),
                        merged.tools(),
                        defaultOptions.getThinking() != Boolean.FALSE,
                        ""));
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
    private record Prepared(
            ChatEngine.Encoded encoded,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            int promptTokens,
            List<String> stops,
            boolean cached,
            int[] parserSeed) {}

    /** All request-shape validation happens here, synchronously (before any thread starts). */
    private Prepared prepare(Prompt prompt) {
        JinferChatOptions options = resolveOptions(prompt.getOptions());
        boolean cached = !prefix.isEmpty();
        List<ToolCallback> callbacks = options.getToolCallbacks();
        if (cached && callbacks != null && !callbacks.isEmpty()) {
            throw new IllegalArgumentException(
                    "a cached-prompt view welds its tools into the cached prefix; per-request"
                            + " toolCallbacks would silently forfeit the cache - put tools on"
                            + " withCachedPrompt(...) instead");
        }
        int maxTokens = options.getMaxTokens() == null ? -1 : options.getMaxTokens();
        // a think span cannot fit a tiny completion budget: below the floor, reasoning is
        // disabled outright (scaffold and sampler both) so the budget buys VISIBLE text
        boolean think =
                options.getThinking() != Boolean.FALSE
                        && (maxTokens < 0 || maxTokens >= RequestPolicy.THINK_FLOOR);
        List<Tool> tools =
                cached
                        ? prefix.tools()
                        : callbacks == null ? List.of() : JinferMappings.toTools(callbacks);
        List<Message> messages = new ArrayList<>(prefix.messages());
        messages.addAll(JinferMappings.toMessages(prompt.getInstructions()));
        Conversation conversation = new Conversation(messages, tools, think, "");
        // cached views are native-only (define enforced it); the base keeps the Jinja fallback
        List<org.springframework.ai.chat.messages.Message> instructions = prompt.getInstructions();
        ChatEngine.Encoded encoded =
                cached
                        ? engine.encodeNative(conversation)
                        : engine.encode(
                                conversation,
                                () -> JinferMappings.toMessageMaps(instructions),
                                () ->
                                        callbacks == null
                                                ? List.of()
                                                : JinferMappings.toToolMaps(callbacks));

        Sampler sampler =
                RequestPolicy.sampler(
                        engine.loaded(),
                        options.getTemperature() == null
                                ? 0.0f
                                : options.getTemperature().floatValue(),
                        options.getTopP() == null ? 0.95f : options.getTopP().floatValue(),
                        options.getSeed() == null ? 42 : options.getSeed(),
                        think,
                        maxTokens,
                        null);
        if (options.getOutputSchema() != null) {
            sampler = withSchemaGrammar(sampler, think, options.getOutputSchema());
        }

        int promptTokens = encoded.prompt().stream().mapToInt(Batch::count).sum();
        long timeoutNanos = options.getTimeout() == null ? 0 : options.getTimeout().toNanos();
        // the generation prompt's reply-grammar tail (a prompt-opened think span): the parser
        // must start in the span state the prompt left the model in, or reasoning routes to
        // the content lane (Qwen3.5, MiniCPM5, Nemotron, SmolLM3 open the span in the prompt)
        int[] parserSeed = encoded.template().map(t -> t.replySeed(think)).orElse(new int[0]);
        return new Prepared(
                encoded,
                sampler,
                maxTokens,
                timeoutNanos,
                promptTokens,
                options.getStopSequences() == null ? List.of() : options.getStopSequences(),
                cached,
                parserSeed);
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
        Prepared p = prepare(prompt);
        ReplyLanes feed =
                new ReplyLanes(p.encoded().template(), engine.loaded().tokenizer(), p.parserSeed());
        List<String> stops = p.stops();
        TextStops.Holdback watch =
                stops.isEmpty() ? null : new TextStops.Holdback(stops, ignored -> {});
        Generator.TokenSink sink =
                token -> {
                    String fragment = feed.feed(token);
                    if (watch != null && !feed.reasoning() && !fragment.isEmpty()) {
                        watch.accept(fragment); // stop strings match the content lane only
                    }
                    return watch == null || !watch.stopped();
                };
        ChatEngine.Outcome outcome =
                engine.generate(
                        p.encoded().prompt(),
                        p.sampler(),
                        p.maxTokens(),
                        p.timeoutNanos(),
                        sink,
                        p.cached());

        // the same parse that fed the stop watch finishes the message - no second decode pass
        Message reply = feed.finish();
        AssistantMessage ai = JinferMappings.toAssistantMessage(reply);
        boolean stopHit = watch != null && watch.stopped();
        if (stopHit) {
            ai =
                    AssistantMessage.builder()
                            .content(TextStops.apply(ai.getText(), stops).text())
                            .properties(ai.getMetadata())
                            .toolCalls(ai.getToolCalls())
                            .build();
        }
        return response(ai, p.promptTokens(), outcome, stopHit);
    }

    /**
     * Closes the shared engine, freeing the prompt tree's native arenas; every view shares it, so
     * closing any model closes them all. Idempotent; later requests fail with {@link
     * IllegalStateException}.
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
        Prepared p = prepare(requestPrompt); // invalid requests throw here, not on the thread
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

    private void streamInto(Prepared p, FluxSink<ChatResponse> sink) {
        AtomicBoolean cancelled = new AtomicBoolean();
        sink.onCancel(() -> cancelled.set(true));
        sink.onDispose(() -> cancelled.set(true));
        ReplyLanes feed =
                new ReplyLanes(p.encoded().template(), engine.loaded().tokenizer(), p.parserSeed());
        List<String> stops = p.stops();
        // the holdback keeps any could-still-be-a-stop suffix unemitted; safe chars flow through
        TextStops.Holdback watch =
                new TextStops.Holdback(stops, out -> sink.next(chunk(out, false)));
        Generator.TokenSink tokenSink =
                token -> {
                    if (cancelled.get()) return false;
                    String fragment = feed.feed(token);
                    if (fragment.isEmpty()) return true;
                    // reasoning streams too, flagged so consumers can keep it off the content
                    // lane; stop sequences stay armed on content only
                    boolean thought = feed.reasoning();
                    if (thought) sink.next(chunk(fragment, true));
                    else watch.accept(fragment);
                    return thought || !watch.stopped();
                };
        try {
            ChatEngine.Outcome outcome =
                    engine.generate(
                            p.encoded().prompt(),
                            p.sampler(),
                            p.maxTokens(),
                            p.timeoutNanos(),
                            tokenSink,
                            p.cached());
            if (cancelled.get()) return; // cancelled subscriptions end silently
            watch.flush(); // release held-back chars (a stopped watch emits nothing past the cut)
            // the final chunk: no text (deltas carried it), but complete tool calls + metadata,
            // from the same parse that streamed
            Message reply = feed.finish();
            AssistantMessage parsed = JinferMappings.toAssistantMessage(reply);
            boolean stopHit = watch.stopped();
            AssistantMessage ai =
                    AssistantMessage.builder()
                            .content("")
                            .properties(parsed.getMetadata())
                            .toolCalls(parsed.getToolCalls())
                            .build();
            sink.next(response(ai, p.promptTokens(), outcome, stopHit));
            sink.complete();
        } catch (Throwable e) {
            sink.error(e);
        }
    }

    /** One delta chunk: text only, no metadata (that lives on the final chunk). */
    static ChatResponse chunk(String delta, boolean thought) {
        AssistantMessage.Builder<?> b = AssistantMessage.builder().content(delta);
        if (thought) {
            b.properties(Map.of(IS_THOUGHT_KEY, true));
        }
        return new ChatResponse(List.of(new Generation(b.build())));
    }

    /**
     * Grammar-constrained output (llama.cpp-style token masking): the schema is compiled to a GBNF
     * grammar whose automaton masks the logits so invalid JSON is unrepresentable, not just
     * unlikely. For a reasoning request the grammar stays dormant until {@code </think>} so the
     * constraint never suppresses the think span, and the boilerplate newline after it passes
     * through unconsumed. The forced token on grammar dead-ends is one of the model's real stop
     * tokens. Specs are cached per (schema, vocab), so repeated schemas reuse the compiled masks.
     */
    private Sampler withSchemaGrammar(Sampler sampler, boolean think, String outputSchema) {
        @SuppressWarnings("unchecked")
        Map<String, Object> schemaMap = (Map<String, Object>) JsonCodec.parse(outputSchema);
        Grammar.Spec spec = Grammar.fromSchema(schemaMap, engine.loaded().tokenizer());
        return RequestPolicy.constrained(engine.loaded(), sampler, spec.cursor(), think);
    }

    private JinferChatOptions resolveOptions(ChatOptions runtime) {
        if (runtime == null) return defaultOptions;
        JinferChatOptions resolved =
                runtime instanceof JinferChatOptions j
                        ? j
                        // foreign options carry only Spring fields: copy them over the defaults
                        : JinferChatOptions.copyOnto(defaultOptions, runtime);
        validate(resolved);
        return resolved;
    }

    private void validate(JinferChatOptions o) {
        if (o.getTopK() != null) throw new IllegalArgumentException("topK is not supported");
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
            AssistantMessage ai,
            int promptTokens,
            ChatEngine.Outcome outcome,
            boolean stoppedBySequence) {
        Generator.GenerationResult result = outcome.result();
        String finishReason =
                stoppedBySequence // a stop-sequence cut IS a stop, not an abort
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
                                        promptTokens,
                                        result.completionTokens(),
                                        null,
                                        new JinferUsage(
                                                result.promptNanos(), result.predictedNanos()),
                                        outcome.restoredTokens() > 0
                                                ? Long.valueOf(outcome.restoredTokens())
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
        private Path modelPath;
        private Path mediaProjector;
        private Path cachedPrompts;
        private int cachedSessions;
        private int contextLength;
        private Double temperature;
        private Double topP;
        private Integer maxTokens;
        private Long seed;
        private Boolean thinking;
        private Duration timeout;
        private JinferChatOptions defaultOptions;
        private ObservationRegistry observationRegistry;
        private ChatModelObservationConvention observationConvention;

        /** The GGUF to load. Required. */
        public Builder modelPath(Path modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        /** The media sidecar (mmproj GGUF: vision/audio encoders) for multimodal models. */
        public Builder mediaProjector(Path mediaProjector) {
            this.mediaProjector = mediaProjector;
            return this;
        }

        /** Mounts a cached-prompt artifact ({@link #saveCachedPrompts}); model-seed-checked. */
        public Builder loadCachedPrompts(Path cachedPrompts) {
            this.cachedPrompts = cachedPrompts;
            return this;
        }

        /**
         * Keeps the last {@code n} live conversation states resident, reused append-only when a
         * request's conversation strictly extends one (the multi-turn zero-restore tier). 0
         * (default) disables the pool. Each kept state holds a full context of KV; on a miss the
         * evictee's allocation is recycled, never re-allocated.
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

        /** Sampling temperature; default 0 (greedy argmax). Per-request options override. */
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
        public Builder maxTokens(Integer maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        /**
         * RNG seed for temperature sampling; default 42. Per-request options override. Same seed
         * does NOT guarantee byte-identical replay at temperature &gt; 0: the CPU backend's
         * run-to-run FP jitter flips near-tie samples.
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
            if (modelPath == null) throw new IllegalArgumentException("modelPath is required");
            return new JinferChatModel(this);
        }
    }
}
