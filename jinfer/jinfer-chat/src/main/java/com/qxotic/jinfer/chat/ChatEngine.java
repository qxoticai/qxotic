package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.LeakWatch;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.llm.SpeculativeDecoding;
import com.qxotic.jinfer.llm.TextStops;
import com.qxotic.jinfer.telemetry.InferenceEvent;
import com.qxotic.jinfer.telemetry.Telemetry;
import com.qxotic.toknroll.IntSequence;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

/**
 * The framework-neutral provider runtime shared by the langchain4j and Spring AI integrations: one
 * loaded model, the two-tier template stack (native codec first, hardened Jinja whole-render
 * fallback), the single-stream generation lock (a jinfer model runs one generation at a time;
 * concurrent calls queue), and the one {@link PromptCache} - hot sessions, block tree and catalog
 * behind withCachedPrompt / cachedSessions(n) / save / load. Integrations adapt only what is
 * genuinely theirs: message/tool mapping into {@link Conversation}s and framework exception types.
 *
 * <p>Everything here speaks jinfer types - no framework classes, no fingerprint/cache internals
 * (the cache package's content addressing stays its own law).
 */
public final class ChatEngine {

    private final LoadedModel<?> loaded;
    private final String modelName;
    private final JinjaChatTemplate jinja;
    private final ReentrantLock lock = new ReentrantLock(true);
    // THE cache: hot sessions + block tree + optional catalog, one front door (all access
    // under the generation lock - the facade is single-threaded by design, like the tree)
    private final PromptCache<?> cache;
    // the streaming driver: at most ONE lazy platform thread, reused while streams keep coming,
    // gone after an idle minute. One is enough - generations serialize on the engine lock anyway,
    // and a fresh thread per request would just park extras on that lock
    private final ThreadPoolExecutor streamDriver =
            new ThreadPoolExecutor(
                    0,
                    1,
                    60,
                    TimeUnit.SECONDS,
                    new LinkedBlockingQueue<>(),
                    r -> new Thread(r, "jinfer-stream"));
    private final AtomicReference<Thread> streamThread = new AtomicReference<>();
    private volatile boolean closed;
    // owned: freed at close(), never shared. null when the CALLER loaded the model and keeps the
    // arena - then close() quiesces and frees this engine's own memory, and nothing else
    private final Arena weights;
    private final Runnable leakWatch; // -Djinfer.leakDetection: reports a GC'd unclosed engine
    // held STRONGLY here on purpose: the telemetry registry keeps only a weak reference, so this
    // field is what keeps the gauge alive exactly as long as the engine it samples
    private final Telemetry.CacheGauge cacheGauge;
    // Published by the generation thread UNDER THE LOCK and read by the JFR sampler. PromptCache
    // is single-threaded by design, so the sampler must never touch it: it reads this immutable
    // snapshot instead. Stale while idle, which is exactly right for a gauge - an idle cache is
    // not changing.
    private volatile PromptCache.Sample cacheSnapshot;

    /** A loaded model with the arena this engine owns - {@code weights} null = the caller's. */
    private record Owned(LoadedModel<?> loaded, Arena weights) {}

    /**
     * Loads {@code modelPath} into an arena the engine will own. The engine OWNS that arena and
     * nothing outside it holds a reference (views share this very engine - "closing any closes
     * all"), so close() can free the weights deterministically after quiescence - mmap pages were
     * always kernel-reclaimable, but load-time conversions/repacks are anonymous memory that a
     * GC-managed arena would only free at a GC that a native-heavy JVM never runs.
     */
    private static Owned load(Path modelPath, Map<String, Path> companions) {
        Arena weights = Arena.ofShared();
        try {
            return new Owned(
                    companions.isEmpty()
                            ? Models.load(modelPath, weights)
                            : Models.load(modelPath, weights, companions),
                    weights);
        } catch (IOException e) {
            weights.close(); // a leaked ofShared arena has no Cleaner: free before failing
            throw new UncheckedIOException("failed to load " + modelPath, e);
        } catch (RuntimeException | Error e) {
            weights.close();
            throw e;
        }
    }

    public ChatEngine(
            Path modelPath, Map<String, Path> companions, PromptCache.Options cacheOptions) {
        this(
                // null = none, as everywhere else companions are accepted: "no companions" and
                // "an empty map" must not be two states, and this threw NullPointerException
                load(modelPath, companions == null ? Map.of() : Map.copyOf(companions)),
                modelPath.getFileName().toString(),
                cacheOptions);
    }

    /**
     * Over a model the CALLER loaded: the seam for a hand-built {@link LoadedModel}, e.g. one
     * carrying a different tokenizer via {@link LoadedModel#withTokenizer}, or a model whose
     * weights are shared with something else in the process.
     *
     * <p>The caller owns the weights arena. {@link #close()} frees this engine's states and blobs
     * and is still the quiescence certificate, but it does NOT free weights it did not allocate -
     * close your arena after this engine, never before.
     *
     * <p>The cache is {@link PromptCache.Options}, which says everything there is to say about it:
     * how many live conversations stay resident, the block layer's RAM budget (0 = blocks off), the
     * catalog file (null = RAM only) and whether that file is served or also written. Three of
     * those used to come from system properties read inside this constructor, which meant an
     * embedder could not set them at all and two engines in one process could not differ.
     */
    public ChatEngine(LoadedModel<?> loaded, String modelName, PromptCache.Options cacheOptions) {
        this(new Owned(loaded, null), modelName, cacheOptions);
    }

    /**
     * As {@link #ChatEngine(LoadedModel, String, PromptCache.Options)} with the SPECULATION depth
     * for a model carrying a draft head (inert otherwise): drafts per verify, the caller's
     * performance policy - it never changes what a pass produces, only how fast (verification keeps
     * the sampler's own distribution).
     */
    public ChatEngine(
            LoadedModel<?> loaded,
            String modelName,
            PromptCache.Options cacheOptions,
            int speculationDepth) {
        this(new Owned(loaded, null), modelName, cacheOptions, checkedDepth(speculationDepth));
    }

    /**
     * Range-checked as a delegation ARGUMENT, so an out-of-range depth throws before anything is
     * built: a check after {@code this(...)} would abandon a fully built engine (pooled state
     * arenas, registered telemetry gauge, armed LeakWatch) with no close() - the exact defect class
     * the providers' build guards exist for.
     */
    private static int checkedDepth(int speculationDepth) {
        if (speculationDepth < 1 || speculationDepth > 8) {
            throw new IllegalArgumentException("speculationDepth " + speculationDepth);
        }
        return speculationDepth;
    }

    private final int speculationDepth; // drafts per verify when the model has a draft head

    private ChatEngine(Owned owned, String modelName, PromptCache.Options cacheOptions) {
        this(owned, modelName, cacheOptions, 4);
    }

    private ChatEngine(
            Owned owned, String modelName, PromptCache.Options cacheOptions, int speculationDepth) {
        this.speculationDepth = speculationDepth;
        if (owned.loaded() == null) throw new IllegalArgumentException("null model");
        if (modelName == null) throw new IllegalArgumentException("null modelName");
        this.weights = owned.weights();
        this.loaded = owned.loaded();
        this.modelName = modelName;
        if (cacheOptions == null) throw new IllegalArgumentException("null cache options");
        // contextCapacity 0 = "the model's maximum", resolvable only now that the model is
        // loaded; explicit values clamp to it (a state larger than the model serves is waste)
        int max = loaded.model().config().contextLength();
        int capacity = cacheOptions.contextCapacity();
        cacheOptions =
                cacheOptions.withContextCapacity(capacity == 0 ? max : Math.min(capacity, max));
        PromptCache<?> built = null;
        try {
            // PromptCache.of reads the model's capabilities itself (codec-less = hot-only,
            // coarse = define-only writes); a zero budget is the block layer's off-switch, and an
            // explicit catalog still mounts - the caller pointed at an artifact on purpose
            built = PromptCache.of(loaded.model(), loaded.seed(), cacheOptions);
            this.cache = built;
            // inside the try: a malformed chat template in the GGUF throws at compile, and an
            // OWNED weights arena must not outlive a constructor that never returns
            this.jinja = new JinjaChatTemplate(loaded.tokenizer(), loaded.chatTemplateSource());
        } catch (RuntimeException | Error e) {
            if (built != null) built.close(); // a failed ctor must not read as a store leak
            freeOwnedWeights();
            throw e;
        }
        // the first reading exists from construction, so /props and the gauge never report a
        // null "no data yet" state distinct from an empty cache (single-threaded here: nothing
        // else can touch the just-built cache)
        this.cacheSnapshot = cache.sample();
        // registered after the cache exists, and after every throwing step: publishing `this`
        // to a registry from a constructor that may still fail would hand out a half-built engine.
        // Only block-caching engines register: a hot-only engine has no tree to sample.
        this.cacheGauge = new Telemetry.CacheGauge(modelName, () -> cacheSnapshot);
        if (cache.blockCaching()) Telemetry.register(cacheGauge);
        // armed last: a ctor throw already cleaned up above and must not read as a leak
        this.leakWatch =
                LeakWatch.arm(
                        this,
                        weights == null
                                ? "ChatEngine (borrowed weights)"
                                : "ChatEngine (owns the weights arena)");
    }

    /** Frees the weights arena iff this engine allocated it. */
    private void freeOwnedWeights() {
        if (weights == null) return; // the caller's arena, and the caller's to close
        try {
            weights.close();
        } catch (UnsupportedOperationException ignored) {
            // a non-closeable arena manages itself; nothing to free eagerly
        }
    }

    public LoadedModel<?> loaded() {
        return loaded;
    }

    public String modelName() {
        return modelName;
    }

    /**
     * Idempotent, blocking: waits out any in-flight generation (the lock) and the stream driver,
     * closes every pooled state (each frees its owned arena NOW - deterministic, not GC-eventual),
     * and frees the tree's blobs; later use fails loudly. Returning is the quiescence certificate:
     * no kernel of this engine touches state memory afterwards.
     */
    public void close() {
        if (Thread.currentThread() == streamThread.get()) {
            throw new IllegalStateException(
                    "cannot close the model from its streaming callback; cancel the stream and"
                            + " close after it ends");
        }
        // the BLOCKING path's twin of the guard above: a ReplySink callback runs on the caller's
        // own thread while it holds this reentrant lock - close() would proceed mid-decode and
        // free state memory under the suspended loop
        if (lock.isHeldByCurrentThread()) {
            throw new IllegalStateException(
                    "cannot close the model from inside its own generation; return from the"
                            + " callback and close after the call ends");
        }
        lock.lock();
        try {
            if (closed) return; // idempotent: the JDK arena close below is one-shot
            closed = true;
            leakWatch.run(); // disarm: this engine was closed properly
            Telemetry.unregister(cacheGauge); // stop sampling a cache that is about to be freed
            cache.close(); // every hot state, the spare, and the block blobs - NOW
        } finally {
            lock.unlock();
        }
        // no interrupt: an in-flight generation finishes; queued streams fail loudly at checkOpen
        streamDriver.shutdown();
        try {
            // await the driver: a live streaming generation may still be reading state memory
            streamDriver.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        // provably quiescent (lock held once, driver drained): the weights can die now
        freeOwnedWeights();
    }

    /** Runs a streaming generation on the engine's single lazy driver thread. */
    public void stream(Runnable generation) {
        try {
            streamDriver.execute(
                    () -> {
                        streamThread.set(Thread.currentThread());
                        try {
                            generation.run();
                        } finally {
                            streamThread.compareAndSet(Thread.currentThread(), null);
                        }
                    });
        } catch (RejectedExecutionException closed) {
            // the unbounded queue never rejects; rejection means the driver was shut down
            throw new IllegalStateException("the model is closed");
        }
    }

    private void checkOpen() {
        if (closed) throw new IllegalStateException("the model is closed");
    }

    /**
     * The encoded prompt, the reply parser source when the model has a native codec, and the
     * prompt's reply seed - the trailing ids grammatically part of the reply, co-produced with the
     * prompt ({@link ChatTemplate.Prompt}) so parser state can never disagree with the tail.
     */
    public record Encoded(List<Batch> prompt, Optional<ChatTemplate> template, int[] replySeed) {}

    /**
     * One request in jinfer terms - what both integrations mean once their own option types are
     * mapped away. The framework-specific parts stay in the adapters: validating their own knobs,
     * compiling a {@code grammar} from their schema type, and resolving their defaults into these
     * fields.
     *
     * @param thinking the caller's intent; {@link #prepare} still applies the completion-budget
     *     floor and a forced call's override, so a request cannot ask for a think span it cannot
     *     afford
     * @param maxTokens completion budget, -1 = bounded only by the context
     * @param reasoningMaxTokens think-span cap override: null = the default policy (half of {@code
     *     maxTokens}), -1 = uncapped, else the cap
     * @param grammar constrains decoding (JSON schema, raw GBNF, ...); null = free
     * @param forcedTool seed the family's call marker so the reply IS a tool call: null = no
     *     forcing, "" = any offered tool, a name = that tool alone (its name is prefix-pinned while
     *     every offered tool stays framed in the prompt)
     * @param cachedView this request runs on a cached-prompt view: native codec only, since the
     *     Jinja whole-render makes no prefix-stability promise
     * @param templateKwargs extra variables for the Jinja whole-render (chat_template_kwargs);
     *     {@link #encode} skips the native codec when any key it does not understand is present
     */
    public record Request(
            List<Message> messages,
            List<Tool> tools,
            boolean thinking,
            int maxTokens,
            Integer reasoningMaxTokens,
            long timeoutNanos,
            Sampling sampling,
            Grammar.Spec grammar,
            String forcedTool,
            boolean cachedView,
            List<String> stops,
            Map<String, Object> templateKwargs) {

        // ranges, not taste: this is a positional record, so a transposed pair of same-typed
        // knobs would otherwise run silently. The four sampling knobs used to sit here loose and
        // adjacent; Sampling groups them and validates its own ranges
        public Request {
            if (messages == null || messages.isEmpty())
                throw new IllegalArgumentException("a request needs at least one message");
            if (sampling == null) throw new IllegalArgumentException("sampling is required");
            if (maxTokens < -1) throw new IllegalArgumentException("maxTokens " + maxTokens);
            if (reasoningMaxTokens != null && reasoningMaxTokens < -1)
                throw new IllegalArgumentException("reasoningMaxTokens " + reasoningMaxTokens);
            if (timeoutNanos < 0) throw new IllegalArgumentException("timeout " + timeoutNanos);
            messages = List.copyOf(messages);
            tools = tools == null ? List.of() : List.copyOf(tools);
            stops = stops == null ? List.of() : List.copyOf(stops);
            if (forcedTool != null) {
                if (tools.isEmpty())
                    throw new IllegalArgumentException("forcing a tool call needs offered tools");
                if (!forcedTool.isEmpty() && namedTool(tools, forcedTool) == null)
                    throw new IllegalArgumentException(
                            "forced tool \"" + forcedTool + "\" is not among the offered tools");
            }
            templateKwargs = templateKwargs == null ? null : Map.copyOf(templateKwargs);
            // a view is native-only and the native codec never sees kwargs - the two are
            // contradictory, so reject the request rather than silently drop the kwargs
            if (cachedView && templateKwargs != null)
                throw new IllegalArgumentException(
                        "templateKwargs need the Jinja whole-render, which a cached view (native"
                                + " codec only) cannot use");
        }
    }

    private static Tool namedTool(List<Tool> tools, String name) {
        for (Tool tool : tools) {
            if (tool.name().equals(name)) return tool;
        }
        return null;
    }

    /** A request lowered to everything a generation pass needs; see {@link #prepare}. */
    public record Prepared(
            Encoded encoded,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            int promptTokens,
            int[] parserSeed,
            List<String> stops,
            boolean claimToolCalls) {

        /**
         * A pre-encoded prompt (raw completions: the caller already tokenized) lowered directly -
         * the one place the no-template sentinels are spelled: no reply parser and no seed (nothing
         * scaffolded the prompt), no view, and call syntax stays visible text because a raw prompt
         * offers no tools.
         */
        public static Prepared raw(
                int[] promptTokens,
                Sampler sampler,
                int maxTokens,
                long timeoutNanos,
                List<String> stops) {
            return new Prepared(
                    new Encoded(List.of(Batch.prefill(promptTokens)), Optional.empty(), new int[0]),
                    sampler,
                    maxTokens,
                    timeoutNanos,
                    promptTokens.length,
                    new int[0],
                    stops,
                    false);
        }
    }

    /**
     * Lowers a request to a prompt, a sampler and a parser seed - the policy both integrations were
     * duplicating:
     *
     * <ul>
     *   <li>the THINK FLOOR: a think span cannot fit a tiny completion budget, so below it (or on a
     *       forced call, whose reply is seeded into the call block) reasoning is disabled in the
     *       scaffold AND the sampler, and the budget buys visible text
     *   <li>encoding: the native codec, falling back to the hardened Jinja whole-render - except on
     *       a view, which is native-only
     *   <li>the sampling stack, with the request's grammar layered on under the same think gating
     *   <li>the parser seed: the generation prompt's reply-grammar tail (a prompt-opened think
     *       span), or a forced call's own seed - the parser must start in the span state the prompt
     *       left the model in, or reasoning routes to the content lane
     *   <li>a forced call's unsplittable recipe: marker seeded into the prompt, names
     *       prefix-pinned, parser pre-fed
     * </ul>
     *
     * {@code messageMaps}/{@code toolMaps} supply the OpenAI-shaped maps the Jinja fallback renders
     * (framework-shaped, hence suppliers - they are never called on the native path).
     */
    public Prepared prepare(
            Request request, Supplier<List<Object>> messageMaps, Supplier<List<Object>> toolMaps) {
        boolean think =
                request.thinking()
                        && request.forcedTool() == null
                        && (request.maxTokens() < 0
                                || request.maxTokens() >= RequestPolicy.THINK_FLOOR);
        Conversation conversation =
                new Conversation(request.messages(), request.tools(), think, "");
        Encoded encoded =
                request.cachedView()
                        ? encodeNative(conversation)
                        : encode(conversation, messageMaps, toolMaps, request.templateKwargs());
        int[] parserSeed = encoded.replySeed();
        Sampler sampler =
                RequestPolicy.sampler(
                        loaded,
                        request.sampling(),
                        think,
                        request.maxTokens(),
                        request.reasoningMaxTokens(),
                        parserSeed);
        if (request.grammar() != null) {
            sampler =
                    RequestPolicy.constrained(
                            loaded, sampler, request.grammar().cursor(), parserSeed);
        }
        if (request.forcedTool() != null) {
            // a named choice pins that tool alone; the prompt still frames every offered tool
            List<Tool> pinned =
                    request.forcedTool().isEmpty()
                            ? conversation.tools()
                            : List.of(namedTool(conversation.tools(), request.forcedTool()));
            RequestPolicy.ForcedCall forced =
                    RequestPolicy.forceCall(loaded, pinned, sampler, parserSeed)
                            .orElseThrow(
                                    () ->
                                            new UnsupportedOperationException(
                                                    "forcing a tool call is not supported by this"
                                                            + " model: it seeds the reply with the"
                                                            + " family's call marker, which needs a"
                                                            + " native codec that declares one"));
            List<Batch> prompt = new ArrayList<>(encoded.prompt());
            prompt.add(forced.seed());
            encoded = new Encoded(List.copyOf(prompt), encoded.template(), encoded.replySeed());
            sampler = forced.sampler();
            parserSeed = forced.parserSeed();
        }
        return new Prepared(
                encoded,
                sampler,
                request.maxTokens(),
                request.timeoutNanos(),
                Batch.positions(encoded.prompt()),
                parserSeed,
                request.stops(),
                !request.tools().isEmpty());
    }

    /**
     * Native-first encode: the model's own codec when it can frame the conversation byte-exactly,
     * else the scrubbed Jinja whole-render over the caller's lazily-built framework maps (only
     * built when the punt actually happens). Media never reaches the text-only fallback - it fails
     * loudly ({@link UnsupportedOperationException}) instead of being silently dropped;
     * integrations map that to their framework's exception type.
     */
    public Encoded encode(
            Conversation conversation,
            Supplier<List<Object>> messageMaps,
            Supplier<List<Object>> toolMaps,
            Map<String, Object> templateKwargs) {
        Optional<ChatTemplate> template = loaded.template();
        UnsupportedConversation punted = null;
        // kwargs the codec has no equivalent for force the whole-render - taking the native path
        // would silently drop them. enable_thinking is the one key lowered separately (it is
        // Conversation.thinking by the time encoding happens), so it alone does not punt.
        if (template.isPresent() && !unknownKwargs(templateKwargs)) {
            try {
                ChatTemplate.Prompt p = template.get().encodePrompt(conversation);
                return new Encoded(p.batches(), template, p.replySeed());
            } catch (UnsupportedConversation punt) {
                punted = punt; // fall through; the parser (same reply grammar) stays usable
            }
        }
        if (hasMedia(conversation)) {
            throw new UnsupportedOperationException(
                    "image/audio/video input is not supported by this model"
                            + (punted != null ? ": " + punted.getMessage() : "")
                            + " (for Gemma 4, attach the media companion: companion(\"media\","
                            + " mmproj))");
        }
        List<Object> tools = toolMaps.get();
        IntSequence ids =
                jinja.render(
                        messageMaps.get(),
                        tools == null || tools.isEmpty() ? null : tools,
                        true,
                        conversation.thinking(),
                        templateKwargs);
        // whole-render fallback: the template's STATIC seed is the only tail knowledge we
        // have for an arbitrary rendered prompt (conversation-shaped tails need the codec path)
        return new Encoded(
                List.of(Batch.prefill(ids.toArray())),
                template,
                template.map(t -> t.replySeed(conversation.thinking())).orElse(new int[0]));
    }

    /** Any key the native path has no equivalent for; template-encoding must punt on these. */
    private static boolean unknownKwargs(Map<String, Object> templateKwargs) {
        if (templateKwargs == null) return false;
        for (String key : templateKwargs.keySet()) {
            if (!"enable_thinking".equals(key)) return true;
        }
        return false;
    }

    private static boolean hasMedia(Conversation conversation) {
        return conversation.messages().stream()
                .flatMap(m -> m.content().stream())
                .anyMatch(p -> p instanceof Part.Blob);
    }

    /**
     * {@code tier} says WHICH source served the prompt, which {@code restoredTokens} alone cannot:
     * a session hit and a block restore can reuse the same count at very different cost (one
     * restores nothing at all). It is the difference worth tuning jinfer.sessions on.
     */
    public record Outcome(
            Generator.GenerationResult result, int restoredTokens, PromptCache.Tier tier) {}

    /**
     * Where a running generation's deltas go. A blocking caller passes {@link #NONE} and reads the
     * finished {@link Completion}; a streaming one emits each delta and answers {@link #cancelled}.
     * The two lanes are separate because reasoning is not content: consumers show it differently,
     * and stop sequences arm on content only.
     */
    public interface ReplySink {

        ReplySink NONE = new ReplySink() {};

        /** A content delta, already past the stop-sequence holdback (safe to show). */
        default void content(String delta) {}

        /** A reasoning delta, when the model has a think span and it is open. */
        default void thinking(String delta) {}

        /** Checked before every token: true ends the pass, and the caller gets no reply. */
        default boolean cancelled() {
            return false;
        }
    }

    /**
     * A finished generation in jinfer terms. {@code reply} is null exactly when {@code cancelled} -
     * a cancelled pass has nothing to report. {@code stopped} means a stop sequence cut the content
     * lane: the reply still carries the full text (with its verbatim token ids intact), and the
     * caller truncates its own message with {@link TextStops#apply}.
     */
    public record Completion(
            Message reply,
            Generator.GenerationResult result,
            boolean stopped,
            boolean cancelled,
            int restoredTokens,
            PromptCache.Tier tier) {}

    /**
     * Runs a prepared request and parses the reply - the loop both integrations wrote twice each
     * (blocking and streaming): reply lanes seeded from the prompt's own grammar tail, the stop
     * holdback that keeps a could-still-be-a-stop suffix unemitted, cancellation checked per token,
     * and ONE parse that both streams the deltas and finishes the message (no second decode pass).
     *
     * <p>Blocking is streaming with a sink that discards: {@code complete(p, ReplySink.NONE)}.
     */
    public Completion complete(Prepared prepared, ReplySink out) {
        InferenceEvent event =
                InferenceEvent.started(modelName, InferenceEvent.CHAT, InferenceEvent.TEXT);
        try {
            Completion completion = complete0(prepared, out);
            record(event, prepared, completion);
            return completion;
        } catch (RuntimeException | Error failure) {
            // the failures worth seeing are exactly the ones that would otherwise emit nothing
            event.errorType = failure.getClass().getSimpleName();
            throw failure;
        } finally {
            event.end();
            event.commit();
        }
    }

    /** Fills the telemetry event from a finished pass; a cancelled pass reports no reply. */
    private void record(InferenceEvent event, Prepared prepared, Completion completion) {
        // outputType stays TEXT until Prepared carries the grammar; JSON lands with that accessor
        event.inputTokens = prepared.promptTokens();
        event.cachedTokens = completion.restoredTokens();
        event.cacheTier = completion.tier().name().toLowerCase(Locale.ROOT);
        event.queueTime = Telemetry.takeQueueWait(); // 0 unless something queued this thread
        Generator.GenerationResult result = completion.result();
        if (result != null) {
            event.outputTokens = result.completionTokens();
            event.prefillTime = result.promptNanos();
            event.decodeTime = result.predictedNanos();
            event.finishReason = result.finishReason();
        }
        if (completion.cancelled()) event.finishReason = "cancelled";
        if (completion.reply() != null) event.reasoningTokens = reasoningTokens(completion.reply());
    }

    /** Reasoning tokens ride the parsed parts as verbatim ids, so counting them is free. */
    private static int reasoningTokens(Message reply) {
        int total = 0;
        for (Part part : reply.content()) {
            if (part instanceof Part.Reasoning reasoning && reasoning.verbatim() != null) {
                total += reasoning.verbatim().length();
            }
        }
        return total;
    }

    private Completion complete0(Prepared prepared, ReplySink out) {
        ReplyLanes lanes =
                new ReplyLanes(
                        prepared.encoded().template(),
                        loaded.tokenizer(),
                        prepared.parserSeed(),
                        prepared.claimToolCalls());
        // over an empty stop list the holdback is a transparent pass-through, so there is no
        // "no stops" special case to carry
        TextStops.Holdback watch = new TextStops.Holdback(prepared.stops(), out::content);
        Generator.TokenSink sink =
                token -> {
                    if (out.cancelled()) return false;
                    String fragment = lanes.feed(token);
                    if (!fragment.isEmpty()) {
                        if (lanes.reasoning()) {
                            out.thinking(fragment);
                        } else {
                            watch.accept(fragment); // stop strings match the content lane only
                        }
                    }
                    // an ENDED reply grammar is the model's own end of turn: every further
                    // token is inert to the parse (fed off-language, or the language accepted
                    // whole), so generating on only burns budget in silence - observed as
                    // whole completion halves lost to LENGTH after a stray control token
                    return !out.cancelled() && !watch.stopped() && !lanes.ended();
                };
        Outcome outcome =
                generate(
                        prepared.encoded().prompt(),
                        prepared.sampler(),
                        prepared.maxTokens(),
                        prepared.timeoutNanos(),
                        sink);
        if (out.cancelled()) {
            // a cancelled pass ends silently: no reply, no completion callback upstream
            return new Completion(
                    null, outcome.result(), false, true, outcome.restoredTokens(), outcome.tier());
        }
        watch.flush(); // release held-back chars (a stopped watch emits nothing past the cut)
        Generator.GenerationResult result = outcome.result();
        if (lanes.ended() && "abort".equals(result.finishReason())) {
            // the sink aborting FOR the ended grammar is a stop, not a client abort: the
            // family's reply language ended the turn exactly like a stop token would
            result =
                    new Generator.GenerationResult(
                            result.tokens(),
                            result.stopToken(),
                            "stop",
                            result.promptNanos(),
                            result.predictedNanos());
        }
        return new Completion(
                lanes.finish(),
                result,
                watch.stopped(),
                false,
                outcome.restoredTokens(),
                outcome.tier());
    }

    /**
     * One generation pass under the engine lock: {@link PromptCache#serve} picks the cheapest
     * source ({@link PromptCache.Tier}) and {@link #run} wires the pass.
     */
    public Outcome generate(
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Generator.TokenSink sink) {
        lock.lock();
        try {
            checkOpen();
            Outcome outcome = run(prompt, sampler, maxTokens, timeoutNanos, sink);
            // sampled here, on the owning thread, while the lock still excludes other generations
            cacheSnapshot = cache.sample();
            return outcome;
        } finally {
            lock.unlock();
        }
    }

    /**
     * One generation pass, every model kind, one path: the cache serves the prompt from the hottest
     * thing that matches (a live session it strictly extends, else the longest block prefix, else
     * fresh compute on a recycled state) and the pass only generates - each decode token joins the
     * cache step-time through the {@code tail} hook, so the reply stays per-position resumable and
     * the hot stream stays in lockstep. Which layers exist (hot-only, define-only coarse, full) was
     * decided once, at construction, by what the model supports.
     */
    @SuppressWarnings("unchecked")
    private <S extends RuntimeState> Outcome run(
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Generator.TokenSink sink) {
        LanguageModel<?, ?, S> model = (LanguageModel<?, ?, S>) loaded.model();
        PromptCache<S> c = (PromptCache<S>) cache;
        // the facade validates the prompt (non-empty, fits the context) before any ingest
        return c.serve(
                prompt,
                (state, serving) -> {
                    if (model instanceof SpeculativeDecoding<?> capable
                            && capable.speculationReady()) {
                        // the model decodes with its own draft-and-verify; verified tokens land
                        // on the state in batches, so the cache adopts them in bulk instead of
                        // the per-token tail
                        SpeculativeDecoding<S> speculative = (SpeculativeDecoding<S>) capable;
                        long start = System.nanoTime();
                        SpeculativeDecoding.Speculation spec =
                                speculative.speculate(
                                        state,
                                        maxTokens,
                                        timeoutNanos,
                                        loaded.stopTokens(),
                                        sampler,
                                        speculationDepth,
                                        sink::onToken);
                        serving.adopt(spec.committed().toArray());
                        return new Outcome(
                                new Generator.GenerationResult(
                                        spec.emitted(),
                                        spec.stopToken(),
                                        spec.stopToken() >= 0 ? "stop" : "length",
                                        0 /* the cache served the prompt */,
                                        System.nanoTime() - start),
                                serving.restored(),
                                serving.tier());
                    }
                    return new Outcome(
                            Generator.generate(
                                    model,
                                    state,
                                    List.of(),
                                    sampler,
                                    maxTokens,
                                    timeoutNanos,
                                    loaded.stopTokens(),
                                    sink,
                                    serving::tail),
                            serving.restored(),
                            serving.tier());
                });
    }

    // ---- cached prompts: define / export / save on the one PromptCache ----

    /**
     * Encode via the native codec only - cached prompts are a prefix-stability bet the Jinja
     * whole-render cannot honor. Throws {@link UnsupportedOperationException} when the model has no
     * native codec; integrations map it.
     */
    public Encoded encodeNative(Conversation conversation) {
        ChatTemplate template =
                loaded.template()
                        .orElseThrow(
                                () ->
                                        new UnsupportedOperationException(
                                                "cached prompts need a native chat-template codec;"
                                                    + " this model only has the Jinja whole-render"
                                                    + " (no prefix-stability guarantee)"));
        ChatTemplate.Prompt p = template.encodePrompt(conversation);
        return new Encoded(p.batches(), Optional.of(template), p.replySeed());
    }

    /**
     * Defines (prefills) a cached prompt: dedups against the tree, commits one block per encoded
     * batch (turn boundaries) - or ONE block for the whole prompt when the model's codec has a
     * coarse residue ({@link StateCodec#coarseBlocks()}) - then discards the working state: the
     * blocks hold the KV.
     */
    public void define(Conversation prefix) {
        List<Batch> prompt = encodeNative(prefix).prompt();
        lock.lock();
        try {
            checkOpen();
            cache.define(prompt);
            cacheSnapshot = cache.sample(); // a define changes blocks/bytes like a pass does
        } finally {
            lock.unlock();
        }
    }

    /**
     * A NEW artifact from the whole block layer (mounted base + everything computed) at {@code out}
     * - refuses this engine's own catalog ({@link #savePrompts} is that write-back).
     */
    public void freezePrompts(Path out) {
        lock.lock();
        try {
            checkOpen();
            cache.export(out);
        } finally {
            lock.unlock();
        }
    }

    /**
     * The accumulating write-back: appends every block computed since boot to this engine's own
     * catalog (append-only, safe against the mounted mapping). A no-op without a read-write
     * catalog. Holds the generation lock across the file IO - generations queue behind it, which is
     * why the server saves at shutdown.
     */
    public void savePrompts() {
        lock.lock();
        try {
            checkOpen();
            cache.save();
        } finally {
            lock.unlock();
        }
    }

    /** Test seam: the block layer's stats line (see {@link PromptCache#treeStats}). */
    public String promptStats() {
        lock.lock();
        try {
            return cache.treeStats();
        } finally {
            lock.unlock();
        }
    }

    /**
     * Test seam: hot-layer occupancy, prefix-hit count, and how many contexts this engine ever
     * allocated - the cache is the allocator, so a steady pipeline stays at {@code
     * allocations=max(1, jinfer.sessions)}.
     */
    public String sessionStats() {
        lock.lock();
        try {
            PromptCache.Sample s = cache.sample();
            return "sessions="
                    + s.hotSessions()
                    + " hits="
                    + s.hotHits()
                    + " allocations="
                    + s.statesAllocated();
        } finally {
            lock.unlock();
        }
    }

    /**
     * The block tree's latest health reading, or null when there is no tree (codec-less model, or
     * {@code -Djinfer.promptCache=false}). Safe from any thread: the tree itself is confined to the
     * generation lock, so this returns the snapshot published after each pass - what the JFR gauge
     * samples and what a server's /props reports.
     */
    public PromptCache.Sample cacheSample() {
        return cacheSnapshot;
    }

    /** Whether the block layer exists for this model (codec present, blocks not disabled). */
    public boolean blockCaching() {
        return cache.blockCaching();
    }
}
