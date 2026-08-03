package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.LeakWatch;
import com.qxotic.jinfer.RuntimeFlags;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.cache.CacheStore;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.FrozenBlocks;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.cache.SessionPool;
import com.qxotic.jinfer.cache.StateCodec;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.TextStops;
import com.qxotic.jinfer.telemetry.InferenceEvent;
import com.qxotic.jinfer.telemetry.Telemetry;
import com.qxotic.toknroll.IntSequence;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

/**
 * The framework-neutral provider runtime shared by the langchain4j and Spring AI integrations: one
 * loaded model, the two-tier template stack (native codec first, hardened Jinja whole-render
 * fallback), the single-stream generation lock (a jinfer model runs one generation at a time;
 * concurrent calls queue), the cached-prompt block tree behind withCachedPrompt/save/load, and the
 * live-session pool behind cachedSessions(n). Integrations adapt only what is genuinely theirs:
 * message/tool mapping into {@link Conversation}s and framework exception types.
 *
 * <p>Everything here speaks jinfer types - no framework classes, no fingerprint/cache internals
 * (the cache package's content addressing stays its own law).
 */
public final class ChatEngine {

    private final LoadedModel<?> loaded;
    private final String modelName;
    private final JinjaChatTemplate jinja;
    private final ReentrantLock lock = new ReentrantLock(true);
    private final CacheStore promptStore; // owned: closed (freed) with the engine
    private final FrozenBlocks mounted; // the read-only mounted artifact (null = none)
    private final PromptCache<?> prompts; // the cached-prompt block tree (null = unsupported)
    // cachedSessions(n): the last n live conversation states, reused append-only when a new
    // prompt's batch stream strictly extends one (all access under the generation lock)
    private final SessionPool<?> pool;
    // codec-less models cannot pool sessions, but the allocation is still recycled: the one bare
    // context, wiped after each pass (all access under the generation lock)
    private RuntimeState bare;
    // the streaming driver: at most ONE lazy platform thread, reused while streams keep coming,
    // gone after an idle minute. One is enough - generations serialize on the engine lock anyway,
    // and a fresh thread per request would just park extras on that lock
    private final java.util.concurrent.ThreadPoolExecutor streamDriver =
            new java.util.concurrent.ThreadPoolExecutor(
                    0,
                    1,
                    60,
                    java.util.concurrent.TimeUnit.SECONDS,
                    new java.util.concurrent.LinkedBlockingQueue<>(),
                    r -> new Thread(r, "jinfer-stream"));
    private final AtomicReference<Thread> streamThread = new AtomicReference<>();
    private int sessionHits;
    private int statesAllocated; // contexts this engine has had to allocate (steady state: 1)
    private volatile boolean closed;
    // owned: freed at close(), never shared. null when the CALLER loaded the model and keeps the
    // arena - then close() quiesces and frees this engine's own memory, and nothing else
    private final java.lang.foreign.Arena weights;
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
    private record Owned(LoadedModel<?> loaded, java.lang.foreign.Arena weights) {}

    /**
     * Loads {@code modelPath} into an arena the engine will own. The engine OWNS that arena and
     * nothing outside it holds a reference (views share this very engine - "closing any closes
     * all"), so close() can free the weights deterministically after quiescence - mmap pages were
     * always kernel-reclaimable, but load-time conversions/repacks are anonymous memory that a
     * GC-managed arena would only free at a GC that a native-heavy JVM never runs.
     */
    private static Owned load(Path modelPath, Path mediaProjector, int contextLength) {
        java.lang.foreign.Arena weights = java.lang.foreign.Arena.ofShared();
        // the integrations' builder contract is "0 = the model's own maximum"; Models.load spells
        // that -1 - without this both integrations crashed in the port's tensor sizing on 0
        int ctx = contextLength <= 0 ? -1 : contextLength;
        try {
            return new Owned(
                    mediaProjector == null
                            ? Models.load(modelPath, ctx, weights)
                            : Models.load(modelPath, mediaProjector, ctx, weights),
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
            Path modelPath,
            Path mediaProjector,
            int contextLength,
            Path cachedPrompts,
            int cachedSessions) {
        this(
                load(modelPath, mediaProjector, contextLength),
                modelPath.getFileName().toString(),
                cachedPrompts,
                cachedSessions);
    }

    /**
     * Over a model the CALLER loaded: the seam for a hand-built {@link LoadedModel}, e.g. one
     * carrying a different tokenizer via {@link LoadedModel#withTokenizer}, or a model whose
     * weights are shared with something else in the process.
     *
     * <p>The caller owns the weights arena. {@link #close()} frees this engine's states and blobs
     * and is still the quiescence certificate, but it does NOT free weights it did not allocate -
     * close your arena after this engine, never before.
     */
    public ChatEngine(
            LoadedModel<?> loaded, String modelName, Path cachedPrompts, int cachedSessions) {
        this(new Owned(loaded, null), modelName, cachedPrompts, cachedSessions);
    }

    private ChatEngine(Owned owned, String modelName, Path cachedPrompts, int cachedSessions) {
        if (owned.loaded() == null) throw new IllegalArgumentException("null model");
        if (modelName == null) throw new IllegalArgumentException("null modelName");
        this.pool = new SessionPool<>(cachedSessions);
        this.weights = owned.weights();
        this.loaded = owned.loaded();
        this.modelName = modelName;
        try {
            this.promptStore = CacheStore.inMemory();
            // built only when the model can support it AND jinfer.promptCache allows it - the
            // engine owns the cache now, so it owns the off-switch too. An explicitly mounted
            // artifact overrides the flag: the caller pointed at frozen blocks on purpose. A
            // codec-less model must still load and chat - the codec throw belongs to the first
            // CACHED-feature use, not to plain construction.
            this.mounted =
                    cachedPrompts == null ? null : FrozenBlocks.open(cachedPrompts, loaded.seed());
            if (mounted != null) {
                this.prompts = tree(loaded, promptStore, mounted);
            } else if (RuntimeFlags.PROMPT_CACHE && loaded.model().stateCodec().isPresent()) {
                this.prompts = tree(loaded, promptStore, null);
            } else {
                this.prompts = null;
            }
            // inside the try: a malformed chat template in the GGUF throws at compile, and an
            // OWNED weights arena must not outlive a constructor that never returns
            this.jinja = new JinjaChatTemplate(loaded.tokenizer(), loaded.chatTemplateSource());
        } catch (IOException e) {
            freeOwnedWeights();
            throw new UncheckedIOException("failed to prepare " + modelName, e);
        } catch (RuntimeException | Error e) {
            freeOwnedWeights();
            throw e;
        }
        // the first reading exists from construction, so /props and the gauge never report a
        // null "no data yet" state distinct from an empty cache (single-threaded here: the tree
        // was just built and nothing else can touch it)
        if (prompts != null) this.cacheSnapshot = prompts.sample();
        // registered after the cache exists, and after every throwing step: publishing `this`
        // to a registry from a constructor that may still fail would hand out a half-built engine
        this.cacheGauge = new Telemetry.CacheGauge(modelName, () -> cacheSnapshot);
        Telemetry.register(cacheGauge);
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

    private static <S extends RuntimeState> PromptCache<S> tree(
            LoadedModel<S> loaded, CacheStore store, FrozenBlocks base) {
        // Bounded, because this tree no longer holds only declared prompts: since it resumes
        // EVERY prompt under jinfer.promptCache, an unbounded budget would grow without limit
        // across arbitrary conversations. jinfer.promptCacheMB caps it, LRU-leaf eviction trims it.
        //
        // Consequence worth knowing: a prompt defined via withCachedPrompt is now cached
        // best-effort, not pinned - an idle one can be evicted and re-prefilled on next use.
        // Permanent residency is what freezePrompts + a mounted artifact is for; frozen blocks
        // never count against this budget and are never evicted.
        return new PromptCache<>(
                loaded.codec(), store, RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES, loaded.seed(), base);
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
        lock.lock();
        try {
            if (closed) return; // idempotent: the JDK arena close below is one-shot
            closed = true;
            leakWatch.run(); // disarm: this engine was closed properly
            Telemetry.unregister(cacheGauge); // stop sampling a cache that is about to be freed
            pool.close();
            if (bare instanceof com.qxotic.jinfer.BaseState base) base.close();
            bare = null;
            promptStore.close();
        } finally {
            lock.unlock();
        }
        // no interrupt: an in-flight generation finishes; queued streams fail loudly at checkOpen
        streamDriver.shutdown();
        try {
            // await the driver: a live streaming generation may still be reading state memory
            streamDriver.awaitTermination(
                    Long.MAX_VALUE, java.util.concurrent.TimeUnit.NANOSECONDS);
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
        } catch (java.util.concurrent.RejectedExecutionException closed) {
            // the unbounded queue never rejects; rejection means the driver was shut down
            throw new IllegalStateException("the model is closed");
        }
    }

    private void checkOpen() {
        if (closed) throw new IllegalStateException("the model is closed");
    }

    /** The encoded prompt plus the reply parser when the model has a native codec. */
    public record Encoded(List<Batch> prompt, Optional<ChatTemplate> template) {}

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
            float temperature,
            float topP,
            long seed,
            Grammar.Spec grammar,
            String forcedTool,
            boolean cachedView,
            List<String> stops,
            java.util.Map<String, Object> templateKwargs) {

        // ranges, not taste: this is a positional record with adjacent same-typed knobs, so a
        // transposed temperature/topP would otherwise sample differently and silently
        public Request {
            if (messages == null || messages.isEmpty())
                throw new IllegalArgumentException("a request needs at least one message");
            if (temperature < 0) throw new IllegalArgumentException("temperature " + temperature);
            if (topP <= 0 || topP > 1) throw new IllegalArgumentException("topP " + topP);
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
            templateKwargs = templateKwargs == null ? null : java.util.Map.copyOf(templateKwargs);
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
            boolean cachedView,
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
                    new Encoded(List.of(Batch.prefill(promptTokens)), Optional.empty()),
                    sampler,
                    maxTokens,
                    timeoutNanos,
                    promptTokens.length,
                    new int[0],
                    stops,
                    false,
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
        Sampler sampler =
                RequestPolicy.sampler(
                        loaded,
                        request.temperature(),
                        request.topP(),
                        request.seed(),
                        think,
                        request.maxTokens(),
                        request.reasoningMaxTokens());
        if (request.grammar() != null) {
            sampler = RequestPolicy.constrained(loaded, sampler, request.grammar().cursor(), think);
        }
        int[] parserSeed = encoded.template().map(t -> t.replySeed(think)).orElse(new int[0]);
        if (request.forcedTool() != null) {
            // a named choice pins that tool alone; the prompt still frames every offered tool
            List<Tool> pinned =
                    request.forcedTool().isEmpty()
                            ? conversation.tools()
                            : List.of(namedTool(conversation.tools(), request.forcedTool()));
            RequestPolicy.ForcedCall forced =
                    RequestPolicy.forceCall(loaded, pinned, sampler)
                            .orElseThrow(
                                    () ->
                                            new UnsupportedOperationException(
                                                    "forcing a tool call is not supported by this"
                                                            + " model: it seeds the reply with the"
                                                            + " family's call marker, which needs a"
                                                            + " native codec that declares one"));
            List<Batch> prompt = new ArrayList<>(encoded.prompt());
            prompt.add(forced.seed());
            encoded = new Encoded(List.copyOf(prompt), encoded.template());
            sampler = forced.sampler();
            parserSeed = forced.parserSeed();
        }
        return new Prepared(
                encoded,
                sampler,
                request.maxTokens(),
                request.timeoutNanos(),
                positions(encoded.prompt()),
                parserSeed,
                request.stops(),
                request.cachedView(),
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
            java.util.Map<String, Object> templateKwargs) {
        Optional<ChatTemplate> template = loaded.template();
        UnsupportedConversation punted = null;
        // kwargs the codec has no equivalent for force the whole-render - taking the native path
        // would silently drop them. enable_thinking is the one key lowered separately (it is
        // Conversation.thinking by the time encoding happens), so it alone does not punt.
        if (template.isPresent() && !unknownKwargs(templateKwargs)) {
            try {
                return new Encoded(template.get().encode(conversation), template);
            } catch (UnsupportedConversation punt) {
                punted = punt; // fall through; the parser (same reply grammar) stays usable
            }
        }
        if (hasMedia(conversation)) {
            throw new UnsupportedOperationException(
                    "image/audio/video input is not supported by this model"
                            + (punted != null ? ": " + punted.getMessage() : "")
                            + " (for Gemma 4, pass the mmproj GGUF via mediaProjector(...))");
        }
        List<Object> tools = toolMaps.get();
        IntSequence ids =
                jinja.render(
                        messageMaps.get(),
                        tools == null || tools.isEmpty() ? null : tools,
                        true,
                        conversation.thinking(),
                        templateKwargs);
        return new Encoded(List.of(Batch.prefill(ids.toArray())), template);
    }

    /** Any key the native path has no equivalent for; template-encoding must punt on these. */
    private static boolean unknownKwargs(java.util.Map<String, Object> templateKwargs) {
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

    /** Which source served a generation's prompt - see {@link Outcome#tier}. */
    public enum Tier {
        /** A pooled live session the prompt strictly extends: zero restore, only the delta. */
        SESSION,
        /** The block tree: the longest cached prefix restored into a fresh state. */
        BLOCKS,
        /** Nothing reusable: a fresh state prefilled the whole prompt. */
        FRESH
    }

    /**
     * {@code tier} says WHICH source served the prompt, which {@code restoredTokens} alone cannot:
     * a session hit and a block restore can reuse the same count at very different cost (one
     * restores nothing at all). It is the difference worth tuning jinfer.sessions on.
     */
    public record Outcome(Generator.GenerationResult result, int restoredTokens, Tier tier) {}

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
            Tier tier) {}

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
        event.cacheTier = completion.tier().name().toLowerCase(java.util.Locale.ROOT);
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
                    return !out.cancelled() && !watch.stopped();
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
        return new Completion(
                lanes.finish(),
                outcome.result(),
                watch.stopped(),
                false,
                outcome.restoredTokens(),
                outcome.tier());
    }

    /**
     * One generation pass under the engine lock, cheapest source first: a pooled live session the
     * prompt strictly extends, else the block tree's longest cached prefix, else a fresh prefill -
     * see {@link Tier}. Every prompt on a codec model is served (and committed) through the tree;
     * {@code -Djinfer.promptCache=false} turns all of that off at construction.
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
            Outcome outcome = run(loaded.model(), prompt, sampler, maxTokens, timeoutNanos, sink);
            // sampled here, on the owning thread, while the lock still excludes other generations
            if (prompts != null) cacheSnapshot = prompts.sample();
            return outcome;
        } finally {
            lock.unlock();
        }
    }

    /**
     * One generation pass, cheapest source first: a pooled live session when the prompt strictly
     * extends one (zero restore, only the delta prefills), else the prompt tree for cached views
     * (block restore, resume capped one short so the final block re-ingests and leaves fresh
     * logits), else a fresh state. On success the finished state (prompt + reply KV) returns to the
     * pool for the conversation's next turn.
     */
    private <S extends RuntimeState> Outcome run(
            LanguageModel<?, ?, S> model,
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Generator.TokenSink sink) {
        int total = positions(prompt);
        if (prompts == null) {
            // no state codec: no tiers, every prompt prefills - but the allocation is recycled
            S state = bareState(model);
            Generator.GenerationResult result =
                    generate(model, state, prompt, sampler, maxTokens, timeoutNanos, sink);
            // wiped BEFORE parking, so nothing of the conversation lingers between requests;
            // on a throw generate() already closed the state and bare stays empty
            state.reset();
            bare = state;
            return new Outcome(result, 0, Tier.FRESH);
        }
        if (loaded.model().stateCodec().map(StateCodec::coarseBlocks).orElse(false)) {
            // A coarse residue costs MBs PER BLOCK (NemotronH: ~90MB at 30B dims), so the tree is
            // written by define() alone - committing a residue per served turn would grow the
            // store by a block's full weight on every request. Serving RESTORES the longest
            // defined prefix (capped one short so the final token re-ingests and leaves fresh
            // logits) and commits nothing; sessions are not pooled either, since pooled sessions
            // commit as they go.
            S state = bareState(model);
            int restored;
            List<Batch> rest;
            try {
                restored = CachedSession.resume(model, tree(), state, prompt, total - 1).position();
                rest = remainder(prompt, restored);
            } catch (RuntimeException | Error e) {
                ((com.qxotic.jinfer.BaseState) state).close();
                throw e;
            }
            Generator.GenerationResult result =
                    generate(model, state, rest, sampler, maxTokens, timeoutNanos, sink);
            state.reset();
            bare = state;
            return new Outcome(result, restored, restored > 0 ? Tier.BLOCKS : Tier.FRESH);
        }
        // ONE group per batch, so the codec's turn boundaries ARE the block boundaries: a
        // follow-up that diverges after turn k still reuses turns 0..k-1. A single giant block
        // would be unusable the moment the conversation grows by a token.
        List<List<Batch>> groups = new ArrayList<>(prompt.size());
        for (Batch b : prompt) groups.add(List.of(b));
        @SuppressWarnings("unchecked")
        SessionPool<S> sessionPool = (SessionPool<S>) pool;
        return sessionPool.withSession(
                model,
                tree(),
                () -> obtainState(model),
                groups,
                (session, tier1) -> {
                    int restored = session.position();
                    if (tier1) sessionHits++;
                    // commits what was not restored, block per group
                    session.ingestGroups(groups);
                    @SuppressWarnings("unchecked")
                    S state = (S) session.state();
                    Generator.GenerationResult result =
                            generate(
                                    model,
                                    state,
                                    List.of(),
                                    sampler,
                                    maxTokens,
                                    timeoutNanos,
                                    sink);
                    // bring the decode back into the session: the reply extends the stream and
                    // commits as a block, so the next turn's echo continues append-only
                    int ingested = state.position() - total;
                    if (ingested > 0) {
                        session.adopt(result.tokens().subSequence(0, ingested).toArray());
                    }
                    return new Outcome(result, restored, tier1 ? Tier.SESSION : Tier.BLOCKS);
                });
    }

    /** One generation pass; a torn state is freed here rather than left to the Cleaner. */
    private <S extends RuntimeState> Generator.GenerationResult generate(
            LanguageModel<?, ?, S> model,
            S state,
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Generator.TokenSink sink) {
        try {
            return Generator.generate(
                    model,
                    state,
                    Batch.prepare(prompt, state.batchCapacity()),
                    sampler,
                    maxTokens,
                    timeoutNanos,
                    loaded.stopTokens(),
                    sink);
        } catch (RuntimeException | Error e) {
            // a server hammered by failing requests would otherwise stack full-context states.
            // On the pooled path SessionPool.withSession closes the torn state too - close is
            // idempotent, and this catch is the bare path's only protection
            ((com.qxotic.jinfer.BaseState) state).close();
            throw e;
        }
    }

    private static int positions(List<Batch> prompt) {
        int total = 0;
        for (Batch b : prompt) total += b.count();
        return total;
    }

    /** The prompt past its first {@code skip} positions - what a restored state still ingests. */
    private static List<Batch> remainder(List<Batch> prompt, int skip) {
        if (skip == 0) return prompt;
        List<Batch> rest = new ArrayList<>();
        int at = 0;
        for (Batch b : prompt) {
            int end = at + b.count();
            if (end > skip) {
                if (at >= skip) {
                    rest.add(b);
                } else {
                    // a restore boundary inside a batch can only land in a token span - block
                    // boundaries are group (batch-list) boundaries, and media commits whole
                    int[] ids = ((Batch.Input.Tokens) b.input()).ids();
                    rest.add(
                            Batch.prefill(
                                    java.util.Arrays.copyOfRange(ids, skip - at, ids.length)));
                }
            }
            at = end;
        }
        return rest;
    }

    /**
     * A fresh context for a pass nothing recycled could serve. Allocated at FULL batch capacity,
     * not prompt-sized: it serves every later request too, and one born from a 20-token prompt
     * would chunk a 2000-token prefill into ~125 forwards.
     *
     * <p>{@link SessionPool} owns these once handed over - it recycles the least-recent allocation
     * at capacity (and keeps one wiped spare when pooling is off), so a steady pipeline holds at
     * most max(1, jinfer.sessions) contexts and allocates no more.
     */
    private <S extends RuntimeState> S obtainState(LanguageModel<?, ?, S> model) {
        statesAllocated++;
        return model.newState(model.config().contextLength(), RuntimeFlags.BATCH_CAPACITY);
    }

    /** The codec-less path's one recycled context - the pool's spare, without the pool. */
    @SuppressWarnings("unchecked")
    private <S extends RuntimeState> S bareState(LanguageModel<?, ?, S> model) {
        if (bare != null) {
            S state = (S) bare;
            bare = null;
            return state;
        }
        return obtainState(model);
    }

    // ---- cached prompts: the block tree behind withCachedPrompt / save / load ----

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
        return new Encoded(template.encode(conversation), Optional.of(template));
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
            defineOn(loaded.model(), tree(), prompt);
            cacheSnapshot = prompts.sample(); // a define changes blocks/bytes like a pass does
        } finally {
            lock.unlock();
        }
    }

    private <S extends RuntimeState> void defineOn(
            LanguageModel<?, ?, S> model, PromptCache<S> cache, List<Batch> prompt) {
        // coarse-residue codecs (NemotronH: MBs per block) commit ONE block over everything but
        // the trailing scaffold batch: one chunk (batch capacity = prompt length), one residue
        // per prompt. The scaffold is request-shaped (it re-encodes after the user's turn), so
        // a block containing it would never match - yet would still pay the residue.
        boolean coarse = model.stateCodec().map(StateCodec::coarseBlocks).orElse(false);
        S state =
                coarse
                        ? model.newState(
                                model.config().contextLength(), Math.max(positions(prompt), 16))
                        : Generator.stateFor(model, positions(prompt));
        try {
            CachedSession<S> s = CachedSession.resume(model, cache, state, prompt);
            s.ingestGroups(
                    coarse
                            ? List.of(prompt.subList(0, Math.max(1, prompt.size() - 1)))
                            : prompt.stream().map(List::of).toList());
        } finally {
            ((com.qxotic.jinfer.BaseState) state).close(); // define-time scratch: free now
        }
    }

    /** Freezes the whole tree (mounted base + everything defined) into one artifact. */
    public void freezePrompts(Path out) {
        lock.lock();
        try {
            checkOpen();
            tree().freeze(out);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to save cached prompts to " + out, e);
        } finally {
            lock.unlock();
        }
    }

    /** Test seam: the tree's stats line ("blocks=.. hits=.." - see PromptCache.stats). */
    public String promptStats() {
        return tree().stats();
    }

    /**
     * Test seam: pool occupancy, prefix-hit count, and how many contexts this engine ever had to
     * allocate - the pool is the allocator, so a steady pipeline stays at {@code allocations=1}.
     */
    public String sessionStats() {
        lock.lock();
        try {
            return "sessions="
                    + pool.size()
                    + " hits="
                    + sessionHits
                    + " allocations="
                    + statesAllocated;
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

    @SuppressWarnings("unchecked")
    private <S extends RuntimeState> PromptCache<S> tree() {
        if (prompts == null) {
            throw new IllegalStateException(
                    loaded.model().stateCodec().isPresent()
                            ? "block caching is disabled (-Djinfer.promptCache=false)"
                            : loaded.model().getClass().getSimpleName()
                                    + " does not support block caching (no state codec)");
        }
        return (PromptCache<S>) prompts;
    }
}
