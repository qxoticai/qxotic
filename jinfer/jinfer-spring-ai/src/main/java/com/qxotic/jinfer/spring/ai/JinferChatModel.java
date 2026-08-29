package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Channel;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.TextStops;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.codecs.VideoSampler;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Grammar;
import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationRegistry;
import io.micrometer.observation.contextpropagation.ObservationThreadLocalAccessor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
import org.springframework.ai.model.tool.ToolCallingChatOptions;
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
 * fairly on it. For a second pipeline, load the weights once into YOUR arena ({@code
 * Models.load(path, arena)}), build with {@code model(loaded)}, and {@code fork()} pipelines for
 * the price of a context each. Footprint: an instance holds its weights, up to the configured
 * number of retained full-context states, plus the block layer's KV (every served conversation,
 * best-effort within a 2 GiB LRU-evicted RAM-only budget; defined prompts are pinned intent within
 * it).
 *
 * <p>Three cache controls with distinct jobs: {@code withCachedPrompt} defines a shared prefix
 * (prefilled once, restored per request - the system-prompt/tools/few-shot case); {@code
 * Builder.retainSessions} keeps finished CONVERSATION states warm for append-only multi-turn reuse
 * (in-RAM, gone at close); {@code saveCachedPrompts}/{@code Builder.promptCache} persist the
 * defined prompts as an immutable artifact that mounts zero-prefill in later processes. None
 * changes output - byte-identity to a cold run is the law.
 *
 * <p>Run with jinfer's JVM flags: {@code --add-modules jdk.incubator.vector
 * --enable-native-access=ALL-UNNAMED}.
 */
public final class JinferChatModel implements ChatModel, AutoCloseable {

    private static final String PROVIDER = "jinfer";
    private static final Logger LOG = LoggerFactory.getLogger(JinferChatModel.class);
    private static final ChatModelObservationConvention DEFAULT_CONVENTION =
            new DefaultChatModelObservationConvention();

    /**
     * Core's thought-lane convention: {@link MessageAggregator} accumulates these on {@code
     * thoughts} metadata instead of conflating them with content.
     */
    static final String IS_THOUGHT_KEY = "isThought";

    final ChatEngine engine;
    final JinferChatOptions options;
    final ObservationRegistry observationRegistry;
    final ChatModelObservationConvention observationConvention; // null = the default convention
    // cached-prompt view state: EMPTY for the base model. Converted to jinfer types ONCE at view
    // creation (media decoded once, not per request); a view's conversations all start with this
    // prefix, its KV restored from the engine's block tree instead of re-prefilled.
    final VideoSampler videoSampler;
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
            if (b.speculationDepth != null) {
                engine.speculationDepth(b.speculationDepth);
            }
            this.videoSampler = b.videoSampler;
            this.prefix = CachedPrompt.NONE;
            this.observationRegistry =
                    b.observationRegistry == null
                            ? ObservationRegistry.NOOP
                            : b.observationRegistry;
            this.observationConvention = b.observationConvention;
            // precedence: request > options > the container's recommendation (general.sampling.*)
            // > port author recommendation > the engine baseline (SamplingDefaults.DEFAULT_*)
            this.options =
                    resolveDefaults(
                            engine.modelName(), engine.loaded().samplingDefaults(), b.options);
            validate(this.options);
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

    static JinferChatOptions resolveDefaults(
            String model, LoadedModel.SamplingDefaults recommended, JinferChatOptions configured) {
        JinferChatOptions.Builder effective =
                JinferChatOptions.builder()
                        .model(model)
                        .temperature(toDouble(recommended.temperature()))
                        .topP(toDouble(recommended.topP()))
                        .topK(recommended.topK())
                        .minP(toDouble(recommended.minP()));
        if (configured != null) effective.combineWith(configured.mutate());
        return effective.build();
    }

    private static Double toDouble(Float value) {
        return value == null ? null : value.doubleValue();
    }

    private JinferChatModel(JinferChatModel base, CachedPrompt prefix) {
        this.engine = base.engine;
        this.options = base.options;
        this.observationRegistry = base.observationRegistry;
        this.observationConvention = base.observationConvention;
        this.videoSampler = base.videoSampler;
        this.cacheOptions = base.cacheOptions;
        this.ownsWeights = base.ownsWeights;
        this.prefix = prefix;
    }

    /** The fork constructor: a fresh engine over the same borrowed weights, every knob carried. */
    private JinferChatModel(JinferChatModel base, ChatEngine engine) {
        this.engine = engine;
        this.options = base.options;
        this.observationRegistry = base.observationRegistry;
        this.observationConvention = base.observationConvention;
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
        forked.engine.speculationDepth(engine.speculationDepth());
        if (prefix.isEmpty()) return forked;
        try {
            return forked.withPrefix(prefix);
        } catch (RuntimeException | Error e) {
            close(forked::close, e); // a failed re-define must not leak the fork's engine
            throw e;
        }
    }

    /**
     * A model view whose conversations all start with {@code prefixMessages}, offering {@code
     * tools} as the view's DEFAULT tool set - both prefilled ONCE into the engine's block tree,
     * restored (not recomputed) on every call. Composable: calling this on a view branches on its
     * prefix. Immutable, shares the base engine; a view's prefix is pinned intent, where the base
     * model's traffic is cached best-effort.
     *
     * <p>Tools follow the standard parameter precedence, request over defaults: a request that
     * states none offers the welded set (a ChatClient re-stating the SAME set lands on the cache
     * too); a request that states a different set is served with ITS tools, byte-identical to the
     * base model - a cache changes latency, never behavior - but forfeits the prepaid prefill for
     * that call. The usage's cache-read count on every response tells which happened; the first
     * override also warns once on stderr.
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
            List<org.springframework.ai.chat.messages.Message> prefixMessages,
            List<ToolCallback> tools) {
        Objects.requireNonNull(prefixMessages, "prefixMessages");
        Objects.requireNonNull(tools, "tools");
        return withPrefix(
                prefix.merge(
                        JinferMappings.toMessages(prefixMessages, videoSampler),
                        JinferMappings.toTools(tools)));
    }

    private JinferChatModel withPrefix(CachedPrompt merged) {
        engine.definePrompt(merged.conversation(options.getThinking() != Boolean.FALSE));
        return new JinferChatModel(this, merged);
    }

    /** Freezes every prompt defined so far (plus any mounted base) into one artifact. */
    public void saveCachedPrompts(Path out) {
        engine.freezePrompts(out);
    }

    @Override
    public JinferChatOptions getOptions() {
        return options;
    }

    /** Framework types mapped away; every policy below this line lives in {@link ChatEngine}. */
    private ChatEngine.Prepared prepare(Prompt prompt) {
        JinferChatOptions options = JinferChatOptions.from(prompt.getOptions());
        validate(options);
        List<ToolCallback> callbacks = options.getToolCallbacks();
        // request > view default (CachedPrompt.resolveTools, THE precedence rule): an override
        // is served correctly (byte-identical to the base model) at full prefill - a cache
        // changes latency, never behavior - and warns once so a wiring bug stays discoverable
        List<Tool> tools =
                prefix.resolveTools(
                        callbacks == null || callbacks.isEmpty()
                                ? null
                                : JinferMappings.toTools(callbacks));
        boolean cached = prefix.serves(tools);
        if (!cached && !prefix.isEmpty() && warnedToolsOverride.compareAndSet(false, true)) {
            LOG.warn(prefix.toolsOverrideWarning(tools));
        }
        List<Message> messages = new ArrayList<>(prefix.messages());
        messages.addAll(JinferMappings.toMessages(prompt.getInstructions(), videoSampler));
        ChatEngine.Request lowered =
                new ChatEngine.Request(
                        messages,
                        tools,
                        options.getThinking() != Boolean.FALSE,
                        options.getMaxTokens() == null ? -1 : options.getMaxTokens(),
                        null, // Spring AI has no reasoning-budget knob
                        null, // nor a reasoning-message one
                        options.getTimeout() == null ? Duration.ZERO : options.getTimeout(),
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
                        contentGbnf(options.getOutputSchema(), !tools.isEmpty()),
                        ChatEngine.ForcedTool.NONE, // Spring AI has no forced-tool-call knob
                        options.getStopSequences(),
                        null); // Spring AI has no chat_template_kwargs equivalent
        return engine.prepare(lowered);
    }

    /**
     * Grammar-constrained output (llama.cpp-style token masking): the schema is compiled to a GBNF
     * grammar whose automaton masks the logits so invalid JSON is unrepresentable, not just
     * unlikely. Compiling it is the framework-shaped half (Spring AI spells a schema as JSON text);
     * the think gating and the dead-end stop token are the engine's. Specs are cached per (schema,
     * vocab), so repeated schemas reuse the compiled masks.
     *
     * <p>Grammar ONLY: neither adapter restates the schema in the prompt. Spring AI's own {@code
     * BeanOutputConverter} appends it as format instructions on the paths that use it, and where it
     * does not ({@code useProviderStructuredOutput}, or a caller setting {@code outputSchema}),
     * saying it is the caller's to do - see the package docs.
     *
     * <p>The output schema as GBNF source - the engine compiles the family's constrained selection.
     */
    private static String contentGbnf(String outputSchema, boolean toolsOffered) {
        if (outputSchema == null) return null;
        Map<String, Object> schema = JinferMappings.jsonMap(outputSchema);
        return toolsOffered ? Grammar.schemaHoleGbnf(schema) : Grammar.schemaGbnf(schema);
    }

    @Override
    public ChatResponse call(Prompt prompt) {
        Prompt effective = effectivePrompt(prompt, options);
        ChatModelObservationContext observationContext =
                ChatModelObservationContext.builder()
                        .prompt(effective)
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
                            ChatResponse response = doCall(effective);
                            observationContext.setResponse(response);
                            return response;
                        });
    }

    /**
     * The complete request seen by observation, validation and execution: request values over the
     * model's defaults. Tools follow Spring AI's own rule ({@link
     * ToolCallingChatOptions#mergeToolCallbacks}): a request that names tools replaces the
     * defaults' list, one that names none inherits it. {@code combineWith} concatenates the two
     * lists instead, which declared every default tool twice (ChatClient already folds the defaults
     * into the request) and let no request drop them.
     */
    static Prompt effectivePrompt(Prompt requestedPrompt, JinferChatOptions defaults) {
        ChatOptions requested = requestedPrompt.getOptions();
        if (requested == null) return new Prompt(requestedPrompt.getInstructions(), defaults);
        JinferChatOptions request = JinferChatOptions.from(requested);
        JinferChatOptions effective =
                defaults.mutate()
                        .combineWith(request.mutate())
                        .toolCallbacks(
                                ToolCallingChatOptions.mergeToolCallbacks(
                                        request.getToolCallbacks(), defaults.getToolCallbacks()))
                        .toolContext(
                                ToolCallingChatOptions.mergeToolContext(
                                        request.getToolContext(), defaults.getToolContext()))
                        .build();
        return new Prompt(requestedPrompt.getInstructions(), effective);
    }

    private ChatResponse doCall(Prompt prompt) {
        try (ChatEngine.Prepared p = prepare(prompt)) {
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
            return response(ai, done);
        }
    }

    /**
     * Blocking, idempotent: waits out any in-flight request (including a live stream), then frees
     * the pooled session states' arenas and the cached-prompt blobs deterministically; later use of
     * this model (or any view sharing its engine) fails with IllegalStateException.
     *
     * <p>Weights are freed too, LAST and only if this model loaded them: mapped tensor pages are
     * kernel-reclaimable, but load-time conversions and repacks are anonymous memory that a
     * GC-managed arena would free only at a GC a native-heavy JVM never runs. A model built with
     * {@code model(...)} borrows its weights instead - close YOUR arena after it, never before.
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
        Prompt effective = effectivePrompt(prompt, options);
        // invalid requests throw here, not on the thread: prepare once eagerly (a Prepared is
        // single-use in this engine, so each subscription re-prepares below)
        try (ChatEngine.Prepared validated = prepare(effective)) {
            // validation only
        }
        // per-subscription observation state: a flux is re-subscribable, and a shared
        // Observation would race on start()/setResponse across subscriptions
        return Flux.deferContextual(
                view -> {
                    ChatModelObservationContext observationContext =
                            ChatModelObservationContext.builder()
                                    .prompt(effective)
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
                            Flux.create(
                                    sink ->
                                            engine.stream(
                                                    () -> {
                                                        // prepare can fail on the driver thread
                                                        // (the engine closed while this was
                                                        // queued): the sink must hear it, or the
                                                        // subscriber waits forever
                                                        try (ChatEngine.Prepared p =
                                                                prepare(effective)) {
                                                            streamInto(p, sink);
                                                        } catch (Throwable t) {
                                                            sink.error(t);
                                                        }
                                                    }));
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
                                public void on(ChatEngine.Delta delta) {
                                    if (delta.channel() == Channel.CONTENT) {
                                        sink.next(chunk(delta.text(), false));
                                    } else if (delta.channel() == Channel.REASONING) {
                                        // reasoning streams too, flagged so consumers can keep it
                                        // off the content lane
                                        sink.next(chunk(delta.text(), true));
                                    }
                                    // other channels are structural: a claimed call span surfaces
                                    // nothing until the finished reply's parsed calls
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
            sink.next(response(ai, done));
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
        // tools together with an output schema COMPOSE: the schema rides the family's reply
        // language (calls stay the family's own syntax, visible text can only be the schema);
        // a family without a reply language rejects at prepare, loudly
    }

    private ChatResponse response(AssistantMessage ai, ChatEngine.Completion done) {
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
                                        done.promptTokens(),
                                        result.completionTokens(),
                                        null,
                                        new JinferUsage(
                                                result.promptTime().toNanos(),
                                                result.decodeTime().toNanos(),
                                                done.tier(),
                                                done.speculated()
                                                        .map(s -> (long) s.drafted())
                                                        .orElse(null),
                                                done.speculated()
                                                        .map(s -> (long) s.accepted())
                                                        .orElse(null),
                                                done.speculated()
                                                        .map(s -> (long) s.forwards())
                                                        .orElse(null)),
                                        done.restoredTokens() > 0
                                                ? Long.valueOf(done.restoredTokens())
                                                : null,
                                        null))
                        .rateLimit(new EmptyRateLimit())
                        .keyValue("prompt-eval-duration", result.promptTime())
                        .keyValue("eval-duration", result.decodeTime())
                        .build();
        return new ChatResponse(List.of(generation), metadata);
    }

    /**
     * Native usage detail: the exact phase timings of the generation pass, which cache tier served
     * the prompt ({@code FRESH} = nothing matched; note a partial restore still reports {@code
     * BLOCKS}, so the usage's cache-read COUNT is the ground truth for how much was saved), and -
     * when the pass ran self-speculation - its acceptance counters ({@code drafted} tokens proposed
     * by the draft head, {@code accepted} of them verified, {@code forwards} target model passes;
     * all null on a plain decode).
     */
    public record JinferUsage(
            long promptNanos,
            long predictedNanos,
            PromptCache.Tier servedFrom,
            Long speculatedDrafted,
            Long speculatedAccepted,
            Long speculatedForwards) {}

    private static String toFinishReason(
            Generator.FinishReason jinferReason, boolean hasToolCalls) {
        if (hasToolCalls) return "tool_calls";
        return switch (jinferReason) {
            case STOP -> "stop";
            case LENGTH -> "length";
            default -> "other";
        };
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
        private Integer
                contextLength; // null = unset -> min(4096, model); the loaded path rejects sets
        private JinferChatOptions options;
        private Integer speculationDepth;
        private ObservationRegistry observationRegistry;
        private ChatModelObservationConvention observationConvention;

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
         * model("unsloth/gemma-4-E2B-it-GGUF:Q8_0");
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
         * Reported as the response's model name; defaults to the model class. {@link #model} only.
         */
        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        /**
         * How video content becomes frames - default {@link
         * com.qxotic.jinfer.codecs.VideoSampler#UNIFORM} (the reference policy: 32 frames uniform
         * across the whole duration). Any policy composes: {@code v -> VideoCodec.ffmpeg().span(v,
         * 8)}, a window of a long source, caller-curated timestamps.
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
         * Mounts one existing cached-prompt artifact read-only; missing or incompatible artifacts
         * fail the build loudly.
         */
        public Builder promptCache(Path promptCache) {
            this.promptCache = Objects.requireNonNull(promptCache, "promptCache");
            return this;
        }

        /**
         * Keeps the last {@code n} live conversation states resident, reused append-only when a
         * request's conversation strictly extends one (the multi-turn zero-restore tier). Each kept
         * conversation holds a full context of KV.
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
         * Model-wide generation defaults. Requests override them through Spring AI's standard
         * {@link ChatOptions} precedence; unset sampling fields use the GGUF/port defaults.
         */
        public Builder options(JinferChatOptions options) {
            this.options = Objects.requireNonNull(options, "options");
            return this;
        }

        /**
         * Draft tokens per verify block for self-speculative decoding, 0..8 (0 disables). Inert
         * unless the model carries a draft head (e.g. Gemma 4's MTP sidecar, attached with {@code
         * companion("speculation", ...)}); unset leaves the engine's default (4). Output is
         * byte-identical to plain greedy decode - speculation changes speed, never content; the
         * acceptance counters ride {@link JinferUsage}.
         */
        public Builder speculationDepth(Integer speculationDepth) {
            this.speculationDepth = speculationDepth;
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
                        "a model is required: model(\"owner/repo:Q4_K_M\"),"
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
    }
}
