package com.qxotic.jinfer.x.chat;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Arenas;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.LanguageModel;
import com.qxotic.jinfer.x.boundary.LeakWatch;
import com.qxotic.jinfer.x.boundary.RuntimeState;
import com.qxotic.jinfer.x.cache.PromptCache;
import com.qxotic.jinfer.x.llm.Generator;
import com.qxotic.jinfer.x.llm.Sampler;
import com.qxotic.jinfer.x.llm.Sampling;
import com.qxotic.jinfer.x.llm.SpecialTokens;
import com.qxotic.jinfer.x.telemetry.CacheSample;
import com.qxotic.jinfer.x.telemetry.InferenceEvent;
import com.qxotic.jinfer.x.telemetry.Telemetry;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

/**
 * The framework-neutral provider runtime: one loaded model, the two-tier template stack (native
 * codec first, hardened Jinja whole-render fallback), the single-stream generation lock (a jinfer
 * model runs one generation at a time; concurrent calls queue), and the one {@link PromptCache} -
 * hot sessions, block tree and catalog behind definePrompt / freezePrompts / savePrompts.
 * Integrations adapt only what is genuinely theirs: message/tool mapping into {@link Conversation}s
 * and framework exception types.
 *
 * <p>Everything here speaks jinfer types - no framework classes, no cache internals (the cache's
 * content addressing stays its own law).
 */
public final class ChatEngine {

    /**
     * The smallest completion budget a think span fits: below it, thinking is disabled per request
     * regardless of the model default - silently spending a tiny budget on reasoning scaffold would
     * return empty visible text.
     */
    public static final int THINK_FLOOR = 16;

    // prompt-encode chunk width; Generator re-chunks to the STATE's batchCapacity before ingest,
    // so this only decides how many Batch objects a prompt arrives as (the cache's block
    // boundaries are the codec's turn boundaries, one per batch)
    private static final int ENCODE_BATCH = Integer.getInteger("jinfer.batchCapacity", 512);

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
     * nothing outside it holds a reference, so close() can free the weights deterministically after
     * quiescence - mmap pages were always kernel-reclaimable, but load-time conversions/repacks are
     * anonymous memory that a GC-managed arena would only free at a GC that a native-heavy JVM
     * never runs.
     */
    private static Owned load(Path modelPath, Map<String, Path> companions) {
        Arena weights = Arenas.newCrossThread();
        try {
            return new Owned(
                    companions.isEmpty()
                            ? Models.load(modelPath, weights)
                            : Models.load(modelPath, weights, companions),
                    weights);
        } catch (IOException e) {
            weights.close(); // a leaked shared arena has no Cleaner: free before failing
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
     * Over a model the CALLER loaded: the seam for a hand-built {@link LoadedModel}, or a model
     * whose weights are shared with something else in the process.
     *
     * <p>The caller owns the weights arena. {@link #close()} frees this engine's states and blobs
     * and is still the quiescence certificate, but it does NOT free weights it did not allocate -
     * close your arena after this engine, never before.
     *
     * <p>The cache is {@link PromptCache.Options}, which says everything there is to say about it:
     * how many live conversations stay resident, the block layer's RAM budget (0 = blocks off), the
     * catalog file (null = RAM only) and whether that file is served or also written.
     */
    public ChatEngine(LoadedModel<?> loaded, String modelName, PromptCache.Options cacheOptions) {
        this(new Owned(loaded, null), modelName, cacheOptions);
    }

    private ChatEngine(Owned owned, String modelName, PromptCache.Options cacheOptions) {
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
        // the first reading exists from construction, so the gauge never reports a null "no data
        // yet" state distinct from an empty cache (single-threaded here: nothing else can touch
        // the just-built cache)
        this.cacheSnapshot = cache.sample();
        // registered after the cache exists, and after every throwing step: publishing `this`
        // to a registry from a constructor that may still fail would hand out a half-built engine.
        // Only block-caching engines register: a hot-only engine has no tree to sample.
        this.cacheGauge = new Telemetry.CacheGauge(modelName, () -> sample(cacheSnapshot));
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

    /** Telemetry's vocabulary for the cache's own sample - mapped here, at the only seam. */
    private static CacheSample sample(PromptCache.Sample s) {
        return new CacheSample(
                s.blocks(), s.bytes(), s.budgetBytes(), s.hits(), s.misses(), s.evictions());
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
     * The encoded prompt, the reply parser seeded with the prompt-owned reply prefix (null = raw
     * completions: no scaffold exists, so call {@link #prepare}'s raw lane), and that prefix itself
     * - the trailing ids grammatically part of the reply, co-produced with the prompt so parser
     * state can never disagree with the tail. {@code replyPrefix} is informational: it already
     * rides the prompt's last batch and the parser already consumed it.
     */
    public record Encoded(List<Batch> prompt, ReplyParser parser, IntSequence replyPrefix) {}

    /** Whether a request forces the reply to be a tool call, and to which offered tool. */
    public sealed interface ForcedTool {

        /** No forcing; the reply is whatever the model writes. */
        enum Simple implements ForcedTool {
            NONE,
            /** Any offered tool; the model picks which. */
            ANY
        }

        ForcedTool NONE = Simple.NONE;
        ForcedTool ANY = Simple.ANY;

        /** Exactly this tool - its name is prefix-pinned while every offered tool stays framed. */
        record Named(String name) implements ForcedTool {
            public Named {
                if (name == null || name.isEmpty()) {
                    throw new IllegalArgumentException("a named forced tool needs a name");
                }
            }
        }
    }

    /**
     * One request in jinfer terms - what every integration means once its own option types are
     * mapped away. The framework-specific parts stay in the adapters: validating their own knobs,
     * compiling a {@code contentGbnf} from their schema type, and resolving their defaults into
     * these fields.
     *
     * @param thinking the caller's intent; {@link #prepare} still applies the {@link #THINK_FLOOR}
     *     and a forced call's override, so a request cannot ask for a think span it cannot afford
     * @param maxTokens completion budget, {@link Generator.Constraints#UNLIMITED} = bounded only by
     *     the context
     * @param reasoningMaxTokens think-span cap override: null = the default policy (half of {@code
     *     maxTokens}), -1 = uncapped, else the cap
     * @param timeout wall-clock budget for the whole pass (prefill AND decode); {@link
     *     Duration#ZERO} = none
     * @param contentGbnf constrains decoding to a GBNF grammar (JSON schema, ...); null = free
     * @param forcedTool seed the family's call marker so the reply IS a tool call
     * @param templateKwargs extra variables for the Jinja whole-render (chat_template_kwargs);
     *     {@link #encode} skips the native codec when any key it does not understand is present
     */
    public record Request(
            List<Message> messages,
            List<Tool> tools,
            boolean thinking,
            int maxTokens,
            Integer reasoningMaxTokens,
            Duration timeout,
            Sampling sampling,
            String contentGbnf,
            ForcedTool forcedTool,
            List<String> stops,
            Map<String, Object> templateKwargs) {

        // ranges, not taste: this is a positional record, so a transposed pair of same-typed
        // knobs would otherwise run silently
        public Request {
            if (messages == null || messages.isEmpty()) {
                throw new IllegalArgumentException("a request needs at least one message");
            }
            if (sampling == null) throw new IllegalArgumentException("sampling is required");
            if (maxTokens < Generator.Constraints.UNLIMITED) {
                throw new IllegalArgumentException("maxTokens " + maxTokens);
            }
            if (reasoningMaxTokens != null && reasoningMaxTokens < -1) {
                throw new IllegalArgumentException("reasoningMaxTokens " + reasoningMaxTokens);
            }
            if (timeout == null || timeout.isNegative()) {
                throw new IllegalArgumentException("timeout " + timeout);
            }
            messages = List.copyOf(messages);
            tools = tools == null ? List.of() : List.copyOf(tools);
            stops = stops == null ? List.of() : List.copyOf(stops);
            forcedTool = forcedTool == null ? ForcedTool.NONE : forcedTool;
            if (forcedTool != ForcedTool.NONE) {
                if (tools.isEmpty()) {
                    throw new IllegalArgumentException("forcing a tool call needs offered tools");
                }
                if (forcedTool instanceof ForcedTool.Named named
                        && namedTool(tools, named.name()) == null) {
                    throw new IllegalArgumentException(
                            "forced tool \"" + named.name() + "\" is not among the offered tools");
                }
            }
            templateKwargs = templateKwargs == null ? null : Map.copyOf(templateKwargs);
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
            Duration timeout,
            int promptTokens,
            List<String> stops) {

        /**
         * A pre-encoded prompt (raw completions: the caller already tokenized) lowered directly -
         * the one place the no-template sentinels are spelled: no reply parser (nothing scaffolded
         * the prompt), so the reply streams and finishes as plain text.
         */
        public static Prepared raw(
                int[] promptTokens,
                Sampler sampler,
                int maxTokens,
                Duration timeout,
                List<String> stops) {
            return new Prepared(
                    new Encoded(List.of(Batch.prefill(promptTokens)), null, IntSequence.empty()),
                    sampler,
                    maxTokens,
                    timeout,
                    promptTokens.length,
                    stops);
        }
    }

    /**
     * Lowers a request to a prompt, a sampler and a seeded parser - the policy every integration
     * would otherwise duplicate:
     *
     * <ul>
     *   <li>the THINK FLOOR: a think span cannot fit a tiny completion budget, so below it (or on a
     *       forced call, whose reply is seeded into the call block) reasoning is disabled in the
     *       scaffold AND the sampler, and the budget buys visible text
     *   <li>encoding: the native codec, falling back to the hardened Jinja whole-render
     *   <li>the sampling stack, with the request's grammar layered on under the same think gating
     *   <li>a forced call's unsplittable recipe: marker seeded into the prompt, names
     *       prefix-pinned, parser pre-fed
     * </ul>
     */
    public Prepared prepare(Request request) {
        boolean think =
                request.thinking()
                        && request.forcedTool() == ForcedTool.NONE
                        && (request.maxTokens() < 0 || request.maxTokens() >= THINK_FLOOR);
        Conversation conversation =
                new Conversation(request.messages(), request.tools(), think, "");
        Encoded encoded = encode(conversation, request.templateKwargs());
        Sampler sampler =
                sampler(
                        request.sampling(),
                        think,
                        request.maxTokens(),
                        request.reasoningMaxTokens(),
                        encoded.replyPrefix());
        if (request.contentGbnf() != null) {
            // ONE selection constrains every chat decode: content can only be the grammar,
            // thinking stays free, and with tools offered the family's own calls stay legal
            sampler =
                    constrained(
                            request.contentGbnf(), request.tools(), sampler, encoded.replyPrefix());
        }
        if (request.forcedTool() != ForcedTool.NONE) {
            // a named choice pins that tool alone; the prompt still frames every offered tool
            List<Tool> pinned =
                    request.forcedTool() == ForcedTool.ANY
                            ? conversation.tools()
                            : List.of(
                                    namedTool(
                                            conversation.tools(),
                                            ((ForcedTool.Named) request.forcedTool()).name()));
            ReplyLanguage.Selection selection =
                    loaded.template()
                            .flatMap(t -> t.forcedCall(pinned))
                            .orElseThrow(
                                    () ->
                                            new UnsupportedOperationException(
                                                    "forcing a tool call is not supported by this"
                                                            + " model: it seeds the reply with the"
                                                            + " family's call marker, which needs a"
                                                            + " native codec that declares one"));
            // ONE walk constrains the whole call - header, an OFFERED name, the family's
            // arguments - leaving no free region to derail in
            int[] seed = selection.forcedPrefix();
            ReplyLanguage.Walk walk = selection.walk();
            walk.seed(IntSequence.of(seed));
            List<Batch> prompt = new ArrayList<>(encoded.prompt());
            prompt.add(Batch.prefill(seed));
            // the parser must start in the span state the prompt leaves the model in, or the
            // seeded scaffold parses as visible text
            encoded.parser().seed(IntSequence.of(seed));
            encoded = new Encoded(List.copyOf(prompt), encoded.parser(), encoded.replyPrefix());
            sampler = walk.sampler(sampler, endTurn());
        }
        return new Prepared(
                encoded,
                sampler,
                request.maxTokens(),
                request.timeout(),
                Batch.positions(encoded.prompt()),
                request.stops());
    }

    /**
     * The id to EMIT when a decode must be ended from outside (a grammar's dead end, a forced
     * call's terminator): the stop set's FIRST element - the model's own end-of-turn, an order the
     * family establishes and {@link LoadedModel} preserves.
     */
    private int endTurn() {
        return loaded.stopTokens().iterator().next();
    }

    /**
     * The standard jinfer sampling stack: a resolved {@link Sampling} plus the reasoning policy -
     * thinking on caps the think span so it cannot starve the visible answer ({@code
     * reasoningOverride}: null = half of {@code maxTokens}, -1 = uncapped); thinking off masks the
     * think markers outright.
     */
    private Sampler sampler(
            Sampling sampling,
            boolean think,
            int maxTokens,
            Integer reasoningOverride,
            IntSequence replyPrefix) {
        Sampler sampler = sampling.sampler(loaded.model().config().vocabularySize());
        if (!think) {
            return Thinking.banMarkers(sampler, loaded.tokenizer());
        }
        int budget =
                reasoningOverride != null
                        ? reasoningOverride
                        : maxTokens >= 0 ? Math.max(1, maxTokens / 2) : -1;
        // prompt-opened spans (replyPrefix carries the open id): the cap must start ARMED - the
        // open token never passes through the sampler on those families
        boolean startInThink = false;
        OptionalInt open = SpecialTokens.find(loaded.tokenizer(), Thinking.OPEN);
        if (open.isPresent()) {
            int openId = open.getAsInt();
            for (int i = 0; i < replyPrefix.length(); i++) {
                if (replyPrefix.intAt(i) == openId) {
                    startInThink = true;
                    break;
                }
            }
        }
        return Thinking.capBudget(sampler, loaded.tokenizer(), budget, startInThink);
    }

    /**
     * Every constrained CHAT decode is ONE selection: visible content can only be {@code
     * contentGbnf}, thinking stays free, and offered tools state the reply's rights (the family's
     * calls stay legal and the answer is optional). Native families derive it from their own reply
     * language; a model without one gets the generic think-aware shape - unless tools are offered,
     * which needs the family's call syntax and rejects loudly without it.
     */
    private Sampler constrained(
            String contentGbnf, List<Tool> tools, Sampler base, IntSequence replyPrefix) {
        Optional<ReplyLanguage.Selection> family =
                loaded.template().flatMap(t -> t.constrainedReply(contentGbnf, tools));
        if (family.isEmpty() && !tools.isEmpty()) {
            throw new UnsupportedOperationException(
                    "tools together with a JSON response format need a family reply language;"
                            + " this model's template declares none");
        }
        ReplyLanguage.Selection selection =
                family.orElseGet(
                        () ->
                                ReplyLanguage.Selection.of(
                                        ReplyLanguage.seq(
                                                ReplyLanguage.opt(
                                                        ReplyLanguage.think(
                                                                ReplyLanguage.mark(Thinking.OPEN),
                                                                ReplyLanguage.free(),
                                                                ReplyLanguage.mark(
                                                                        Thinking.CLOSE))),
                                                ReplyLanguage.content(
                                                        ReplyLanguage.gbnf(contentGbnf))),
                                        loaded.tokenizer()));
        ReplyLanguage.Walk walk = selection.walk();
        walk.seed(replyPrefix);
        return walk.sampler(base, endTurn());
    }

    /**
     * The sink-side copy the media contract demands: embedder chunks are borrowed views of the
     * per-encode scratch arena, dead the moment {@code Embedder#embed} returns, while prompt
     * batches are ingested after {@code ChatTemplate#encode} returns. Non-media batches pass
     * through.
     */
    private static Batch own(Batch batch) {
        if (!(batch.input() instanceof Batch.Input.Embeddings e)) return batch;
        MemoryView<MemorySegment> rows = Views.castToSegmentBacked(e.rows(), "embedding rows");
        Views.requireDense(rows, DataType.FP32, "embedding rows");
        MemoryView<MemorySegment> owned = Views.allocateF32(MEDIA_ROWS, rows.shape().toArray());
        MemorySegment.copy(
                rows.memory().base(),
                rows.byteOffset(),
                owned.memory().base(),
                owned.byteOffset(),
                rows.shape().size() * rows.dataType().byteSize());
        return Batch.embeddings(owned, e.count(), e.bidirectional(), e.contentKey());
    }

    /** Owned arena for the sink-side media copies (GC-managed; the copies outlive the encode). */
    private static final PanamaMemoryArena MEDIA_ROWS = new PanamaMemoryArena(Arena.ofAuto());

    /**
     * Native-first encode: the model's own codec when it can frame the conversation byte-exactly,
     * else the scrubbed Jinja whole-render over the OpenAI-shaped maps this engine derives itself.
     * Media never reaches the text-only fallback - it fails loudly ({@link
     * UnsupportedOperationException}) instead of being silently dropped; integrations map that to
     * their framework's exception type.
     */
    public Encoded encode(Conversation conversation, Map<String, Object> templateKwargs) {
        Optional<ChatTemplate> template = loaded.template();
        UnsupportedConversation punted = null;
        // kwargs the codec has no equivalent for force the whole-render - taking the native path
        // would silently drop them. enable_thinking is the one key lowered separately (it is
        // Conversation.thinking by the time encoding happens), so it alone does not punt.
        if (template.isPresent() && !unknownKwargs(templateKwargs)) {
            try {
                List<Batch> prompt = new ArrayList<>();
                ChatTemplate.ReplyState state =
                        template.get().encode(conversation, ENCODE_BATCH, b -> prompt.add(own(b)));
                return new Encoded(List.copyOf(prompt), state.parser(), state.replyPrefix());
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
        IntSequence ids =
                jinja.render(
                        RenderMaps.messages(conversation),
                        RenderMaps.tools(conversation.tools()),
                        true,
                        conversation.thinking(),
                        templateKwargs);
        // whole-render fallback: no codec co-produced a reply prefix. When the render's own tail
        // opens a think span (the template's scaffold), that tail IS the prefix - without it the
        // parser starts in the wrong span and a constraint walk constrains from token zero,
        // silencing the reasoning it must leave free. The marker is a special token request text
        // can never mint (the render scrubs), so its last occurrence is the scaffold.
        IntSequence replyPrefix = IntSequence.empty();
        if (conversation.thinking()) {
            OptionalInt open = SpecialTokens.find(loaded.tokenizer(), Thinking.OPEN);
            if (open.isPresent()) {
                int[] all = ids.toArray();
                int at = all.length;
                while (--at >= 0 && all[at] != open.getAsInt()) {}
                if (at >= 0) replyPrefix = IntSequence.of(Arrays.copyOfRange(all, at, all.length));
            }
        }
        ReplyParser parser =
                template.map(t -> t.parser(loaded.tokenizer()))
                        .orElseGet(() -> ReplyParser.spans(loaded.tokenizer()));
        parser.seed(replyPrefix);
        return new Encoded(List.of(Batch.prefill(ids.toArray())), parser, replyPrefix);
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
                .anyMatch(p -> p instanceof Content.Media);
    }

    /**
     * {@code tier} says WHICH source served the prompt, which {@code restoredTokens} alone cannot:
     * a session hit and a block restore can reuse the same count at very different cost (one
     * restores nothing at all). It is the difference worth tuning the hot-session count on.
     */
    public record Outcome(
            Generator.GenerationResult result, int restoredTokens, PromptCache.Tier tier) {}

    /**
     * One streamed fragment: the channel it belongs to, UTF-8-safe text, and the verbatim tokens
     * that produced it. Concatenating {@code tokens} over a pass re-encodes the finished reply's
     * verbatim ids exactly (a stop-sequence cut excluded - the holdback swallows the tail it never
     * emits).
     */
    public record Delta(Channel channel, String text, IntSequence tokens) {
        public Delta {
            if (channel == null) throw new IllegalArgumentException("null channel");
            if (text == null || text.isEmpty()) throw new IllegalArgumentException("empty delta");
            if (tokens == null) throw new IllegalArgumentException("null tokens");
        }
    }

    /**
     * Where a running generation's deltas go. A blocking caller passes {@link #NONE} and reads the
     * finished {@link Completion}; a streaming one emits each delta and answers {@link #cancelled}.
     * Channels are separate because reasoning is not content: consumers show it differently, and
     * stop sequences arm on {@link Channel#CONTENT} only.
     */
    public interface ReplySink {

        ReplySink NONE = new ReplySink() {};

        /** One fragment, already past the stop-sequence holdback on the content lane. */
        default void on(Delta delta) {}

        /** Checked before every token: true ends the pass, and the caller gets no reply. */
        default boolean cancelled() {
            return false;
        }
    }

    /**
     * A finished generation in jinfer terms. {@link #cancelled} is derived: a cancelled pass has
     * nothing to report, so its {@code reply} is null - there is no second boolean to disagree
     * with. {@code stopped} means a stop sequence cut the content lane: the reply still carries the
     * full text (with its verbatim token ids intact), and the caller truncates its own message with
     * {@link TextStops#apply}.
     */
    public record Completion(
            Message reply,
            Generator.GenerationResult result,
            boolean stopped,
            int restoredTokens,
            PromptCache.Tier tier) {

        /** A cancelled pass has no reply; that is the whole of it. */
        public boolean cancelled() {
            return reply == null;
        }
    }

    /**
     * Runs a prepared request and parses the reply - the loop integrations would each write
     * (blocking and streaming): the parser seeded by encoding itself, the stop holdback that keeps
     * a could-still-be-a-stop suffix unemitted, cancellation checked per token, and ONE parse that
     * both streams the deltas and finishes the message (no second decode pass).
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
        event.inputTokens = prepared.promptTokens();
        event.cachedTokens = completion.restoredTokens();
        event.cacheTier = completion.tier().name().toLowerCase(Locale.ROOT);
        event.queueTime = Telemetry.takeQueueWait(); // 0 unless something queued this thread
        Generator.GenerationResult result = completion.result();
        if (result != null) {
            event.outputTokens = result.completionTokens();
            event.prefillTime = result.promptTime().toNanos();
            event.decodeTime = result.decodeTime().toNanos();
            event.finishReason = result.finishReason().name().toLowerCase(Locale.ROOT);
        }
        if (completion.cancelled()) event.finishReason = "cancelled";
        if (completion.reply() != null) event.reasoningTokens = reasoningTokens(completion.reply());
    }

    /** Reasoning tokens ride the parsed parts as verbatim ids, so counting them is free. */
    private static int reasoningTokens(Message reply) {
        int total = 0;
        for (Content part : reply.content()) {
            if (part instanceof Content.Reasoning reasoning) {
                total += reasoning.verbatim().length();
            }
        }
        return total;
    }

    private Completion complete0(Prepared prepared, ReplySink out) {
        ReplyParser parser = prepared.encoded().parser();
        // the raw lane: no parser, decoded text only (pre-encoded completions)
        PendingUtf8 pending = parser == null ? new PendingUtf8() : null;
        StringBuilder rawText = parser == null ? new StringBuilder() : null;
        // over an empty stop list the holdback is a transparent pass-through, so there is no
        // "no stops" special case to carry. heldIds tracks the buffered fragments' tokens so a
        // flushed delta still re-encodes its text exactly.
        IntSequence.Builder[] heldIds = {IntSequence.newBuilder()};
        TextStops.Holdback watch =
                new TextStops.Holdback(
                        prepared.stops(),
                        text -> {
                            IntSequence tokens = heldIds[0].build();
                            heldIds[0] = IntSequence.newBuilder();
                            out.on(new Delta(Channel.CONTENT, text, tokens));
                        });
        Generator.GenerationListener listener =
                new Generator.GenerationListener() {
                    @Override
                    public boolean onToken(int token) {
                        if (out.cancelled()) return false;
                        if (parser == null) {
                            PendingUtf8.Fragment fragment =
                                    pending.add(
                                            loaded.tokenizer().decodeBytes(new int[] {token}),
                                            token);
                            if (fragment != null && !fragment.text().isEmpty()) {
                                rawText.append(fragment.text());
                                fragment.ids().forEachInt(heldIds[0]::add);
                                watch.accept(fragment.text());
                            }
                        } else {
                            ReplyParser.Fragment fragment = parser.feed(token);
                            if (!fragment.text().isEmpty()) {
                                Channel channel = parser.channel();
                                if (channel == Channel.CONTENT || channel == null) {
                                    fragment.tokens().forEachInt(heldIds[0]::add);
                                    watch.accept(fragment.text());
                                } else {
                                    out.on(new Delta(channel, fragment.text(), fragment.tokens()));
                                }
                            }
                        }
                        // an ENDED reply grammar is the model's own end of turn: every further
                        // token is inert to the parse, so generating on only burns budget in
                        // silence - observed as whole completion halves lost to LENGTH after a
                        // stray control token
                        return !out.cancelled()
                                && !watch.stopped()
                                && (parser == null || !parser.ended());
                    }
                };
        Outcome outcome =
                generate(
                        prepared.encoded().prompt(),
                        prepared.sampler(),
                        prepared.maxTokens(),
                        prepared.timeout(),
                        listener);
        if (out.cancelled()) {
            // a cancelled pass ends silently: no reply, no completion callback upstream
            return new Completion(
                    null, outcome.result(), false, outcome.restoredTokens(), outcome.tier());
        }
        watch.flush(); // release held-back chars (a stopped watch emits nothing past the cut)
        Generator.GenerationResult result = outcome.result();
        if (parser != null
                && parser.ended()
                && result.finishReason() == Generator.FinishReason.ABORT) {
            // the listener aborting FOR the ended grammar is a stop, not a client abort: the
            // family's reply language ended the turn exactly like a stop token would
            result =
                    new Generator.GenerationResult(
                            result.tokens(),
                            result.stopToken(),
                            Generator.FinishReason.STOP,
                            result.promptTime(),
                            result.decodeTime());
        }
        return new Completion(
                finishReply(parser, pending, rawText),
                result,
                watch.stopped(),
                outcome.restoredTokens(),
                outcome.tier());
    }

    /** The finished reply, from the same parse that streamed - or the raw lane's plain text. */
    private Message finishReply(ReplyParser parser, PendingUtf8 pending, StringBuilder rawText) {
        if (parser != null) return parser.finish();
        // a reply that ENDS mid-sequence leaves bytes buffered; without this drain they were
        // simply lost from the finished message (flush decodes permissively, so a genuinely
        // truncated character becomes U+FFFD rather than vanishing)
        PendingUtf8.Fragment tail = pending.flush();
        if (tail != null) rawText.append(tail.text());
        return new Message(Role.ASSISTANT, rawText.toString());
    }

    /**
     * One generation pass under the engine lock: {@link PromptCache#serve} picks the cheapest
     * source ({@link PromptCache.Tier}) and {@link #run} wires the pass. The listener's {@code
     * onIngested} is REPLACED by the cache's per-position tail hook - a listener cannot own it,
     * because the serving only exists for the pass's duration.
     */
    public Outcome generate(
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            Duration timeout,
            Generator.GenerationListener listener) {
        lock.lock();
        try {
            checkOpen();
            Outcome outcome = run(prompt, sampler, maxTokens, timeout, listener);
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
     * cache step-time through the {@code onIngested} hook, so the reply stays per-position
     * resumable and the hot stream stays in lockstep. Which layers exist (hot-only, define-only
     * coarse, full) was decided once, at construction, by what the model supports.
     */
    @SuppressWarnings("unchecked")
    private <S extends RuntimeState> Outcome run(
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            Duration timeout,
            Generator.GenerationListener listener) {
        LanguageModel<?, ?, S> model = (LanguageModel<?, ?, S>) loaded.model();
        PromptCache<S> c = (PromptCache<S>) cache;
        // the facade validates the prompt (non-empty, fits the context) before any ingest
        return c.serve(
                prompt,
                (state, serving) ->
                        new Outcome(
                                Generator.generate(
                                        model,
                                        state,
                                        List.of(),
                                        sampler,
                                        new Generator.Constraints(
                                                maxTokens, timeout, loaded.stopTokens()),
                                        new Generator.GenerationListener() {
                                            @Override
                                            public boolean onToken(int token) {
                                                return listener.onToken(token);
                                            }

                                            @Override
                                            public void onIngested(int token) {
                                                serving.tail(token);
                                            }
                                        }),
                                serving.restored(),
                                serving.tier()));
    }

    // ---- cached prompts: define / freeze / save on the one PromptCache ----

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
        List<Batch> prompt = new ArrayList<>();
        try {
            ChatTemplate.ReplyState state =
                    template.encode(conversation, ENCODE_BATCH, b -> prompt.add(own(b)));
            return new Encoded(List.copyOf(prompt), state.parser(), state.replyPrefix());
        } catch (UnsupportedConversation punt) {
            // a reply-codec-only family frames through the whole-render - no prefix stability
            throw new UnsupportedOperationException(
                    "cached prompts need a native chat-template codec; " + punt.getMessage());
        }
    }

    /**
     * Defines (prefills) a cached prompt: dedups against the tree, commits one block per encoded
     * batch (turn boundaries) - or ONE block for the whole prompt when the model's codec has a
     * coarse residue ({@code StateCodec#coarseCheckpoints}) - then discards the working state: the
     * blocks hold the KV.
     */
    public void definePrompt(Conversation prefix) {
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
     * allocations=max(1, hotSessions)}.
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
     * The block tree's latest health reading, or a zeroed sample when there is no tree (codec-less
     * model, or a zero block budget). Safe from any thread: the tree itself is confined to the
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
