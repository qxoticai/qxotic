package com.qxotic.jinfer.libjinfer;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.json.Json;
import com.qxotic.jam.JAM;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.ContextModel;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.RuntimeFlags;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.LoadedReranker;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.server.ServerConfig;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.nativeimage.c.type.CTypeConversion.CCharPointerHolder;

/**
 * A loaded model of whatever kind the file is: the model selects the face, as the CLI server does.
 * A language model owns a ChatEngine; the other kinds own their weights arena. Every child handle
 * (state, stream) holds a reference, so the weights outlive {@code jinfer_model_free} until the
 * last child is gone.
 */
final class Model {
    enum Kind {
        LANGUAGE,
        EMBEDDING,
        RERANKING,
        SPEECH,
        TRANSCRIPTION
    }

    /**
     * The server's rules for a chat.completions request, with no request deadline: a library call
     * ends when the caller stops stepping, not on a clock.
     */
    static final ServerConfig CONFIG =
            ServerConfig.local(0)
                    .withLimits(
                            new ServerConfig.Limits(
                                    16,
                                    32L << 20,
                                    true,
                                    Duration.ofSeconds(30),
                                    Duration.ZERO,
                                    Duration.ofSeconds(30)));

    final Kind kind;
    final String name;
    final Path path;
    final Map<String, Path> companions;
    final ChatEngine engine; // LANGUAGE
    final LoadedEmbedder<?> embedder; // EMBEDDING
    final LoadedReranker<?> reranker; // RERANKING
    final SpeechSynthesisModel<?, ?, ?> speech; // SPEECH
    final TranscriptionModel<?, ?, ?> transcription; // TRANSCRIPTION
    final int capacity; // tokens of context each state holds (JINFER_CONTEXT_CAPACITY), 0 for the
    // stateless kinds
    private final Arena arena; // owned by the non-language kinds
    private final Map<String, Object> header;
    private final AtomicInteger references = new AtomicInteger(1);
    private ExecutorService driver; // speech and transcription streams, one at a time
    private ContextState shared; // embed and rerank run in this one state, reset per call
    private CCharPointerHolder info; // the borrowed jinfer_model_info string

    /**
     * {@code face} is the loaded object of {@code kind}; the typed field for that kind takes it.
     * {@code contextCapacity} is the option as given; a language engine has resolved its own.
     */
    private Model(
            Kind kind,
            Path path,
            Map<String, Path> companions,
            Object face,
            Arena arena,
            Map<String, Object> header,
            int contextCapacity) {
        this.kind = kind;
        this.path = path;
        this.name = path.getFileName().toString();
        this.companions = companions;
        this.arena = arena;
        this.header = header;
        this.engine = kind == Kind.LANGUAGE ? (ChatEngine) face : null;
        this.embedder = kind == Kind.EMBEDDING ? (LoadedEmbedder<?>) face : null;
        this.reranker = kind == Kind.RERANKING ? (LoadedReranker<?>) face : null;
        this.speech = kind == Kind.SPEECH ? (SpeechSynthesisModel<?, ?, ?>) face : null;
        this.transcription = kind == Kind.TRANSCRIPTION ? (TranscriptionModel<?, ?, ?>) face : null;
        this.capacity =
                switch (kind) {
                    case LANGUAGE -> engine.contextCapacity();
                    case EMBEDDING, RERANKING ->
                            capacity(
                                    contextCapacity,
                                    contextModel().configuration().maxContextLength(),
                                    0); // a retrieval state grows to the text, so the default is
                    // the maximum
                    default -> 0;
                };
    }

    /**
     * The context_capacity option resolved against the model's maximum: 0 is {@code dflt} (0 for
     * the maximum itself), -1 the maximum, a positive value exact and refused above the maximum.
     */
    static int capacity(int requested, int max, int dflt) {
        if (requested == 0) return dflt == 0 ? max : Math.min(dflt, max);
        if (requested == -1) return max;
        if (requested < 0)
            throw Errors.invalid(
                    "context_capacity must be -1 (the model's maximum), 0 (the default) or"
                            + " positive, got "
                            + requested);
        if (requested > max)
            throw Errors.invalid(
                    "context_capacity "
                            + requested
                            + " exceeds the model's maximum context length "
                            + max
                            + "; -1 selects the maximum");
        return requested;
    }

    static Model load(String ref, Options.Values options) throws IOException {
        Log.init();
        Log.debug(
                "jam providers "
                        + JAM.providers().stream().map(JAM.Provider::id).toList()
                        + ", "
                        + RuntimeFlags.THREADS
                        + " threads");
        return load0(ref, options);
    }

    private static Model load0(String ref, Options.Values options) throws IOException {
        List<String> refs = new ArrayList<>();
        refs.add(ref);
        refs.addAll(options.companions().values());
        for (String r : refs) {
            // a string spelled as a path is a file, never a ref: missing, it is ENOTFOUND by name
            // (the hub answers with a lesson on ref syntax, meant for a mistyped ref)
            boolean pathShaped = r.startsWith("/") || r.startsWith(".") || r.startsWith("~");
            if (pathShaped && !Files.exists(Path.of(r)))
                throw new NoSuchFileException(r, null, "no such file");
        }
        List<Path> paths = ModelStore.standard().resolveAll(refs);
        Path path = paths.get(0);
        Map<String, Path> companions = new LinkedHashMap<>();
        int at = 1;
        for (String capability : options.companions().keySet())
            companions.put(capability, paths.get(at++));
        Map<String, Object> header = header(path);
        PromptCache.Options cache =
                PromptCache.Options.DEFAULTS
                        .withRetainedSessions(RuntimeFlags.cacheSessions())
                        .withBlockBudget(RuntimeFlags.cacheBudget());
        if (options.promptCache() != null)
            cache = cache.withCatalog(Path.of(options.promptCache()), false);
        int requested = options.contextCapacity();
        if (requested < -1) capacity(requested, 0, 0); // refused by name before any weights load
        if (requested == -1) cache = cache.withContextCapacity(0); // the cache's 0 is the maximum
        else if (requested > 0)
            cache = cache.withContextCapacity(requested); // above the maximum, the cache refuses
        Model model;
        try {
            ChatEngine engine = new ChatEngine(path, companions, cache);
            model = new Model(Kind.LANGUAGE, path, companions, engine, null, header, requested);
        } catch (ModelProvider.IncompatibleModelException notLanguage) {
            Arena arena = Arenas.newCrossThread();
            try {
                model = loadOther(path, companions, arena, header, notLanguage, requested);
            } catch (Throwable t) {
                Arenas.close(arena); // best-effort: in the image the arena is ofAuto, not closeable
                throw t;
            }
        }
        Log.debug("loaded " + model.kind + " " + model.name);
        return model;
    }

    private static Model loadOther(
            Path path,
            Map<String, Path> companions,
            Arena arena,
            Map<String, Object> header,
            ModelProvider.IncompatibleModelException notLanguage,
            int contextCapacity)
            throws IOException {
        var retrieval = Models.retrieval(path);
        if (retrieval.isPresent()) {
            Kind kind =
                    retrieval.get() == ModelProvider.Retrieval.EMBEDDING
                            ? Kind.EMBEDDING
                            : Kind.RERANKING;
            Object face =
                    kind == Kind.EMBEDDING
                            ? Models.loadEmbedder(path, arena)
                            : Models.loadReranker(path, arena);
            return new Model(kind, path, companions, face, arena, header, contextCapacity);
        }
        try {
            var asr = Models.loadTranscription(path, arena, companions);
            return new Model(Kind.TRANSCRIPTION, path, companions, asr, arena, header, 0);
        } catch (ModelProvider.IncompatibleModelException notTranscription) {
            try {
                var tts = Models.loadSpeech(path, arena, companions);
                return new Model(Kind.SPEECH, path, companions, tts, arena, header, 0);
            } catch (ModelProvider.IncompatibleModelException notSpeech) {
                throw notLanguage; // the language refusal names the architecture and the artifact
            }
        }
    }

    /** The GGUF header's general.* metadata, what jinfer_model_info describes the file with. */
    private static Map<String, Object> header(Path path) {
        Map<String, Object> out = new LinkedHashMap<>();
        try {
            GGUF gguf = GGUF.read(path);
            for (String key : gguf.getMetadataKeys()) {
                if (!key.startsWith("general.") || key.contains("tokens")) continue;
                Object value = gguf.getValue(Object.class, key);
                if (value instanceof String || value instanceof Number || value instanceof Boolean)
                    out.put(key.substring("general.".length()), value);
            }
        } catch (Exception e) {
            Log.debug("header of " + path + " not read: " + e.getMessage());
        }
        return out;
    }

    void retain() {
        references.incrementAndGet();
    }

    void release() {
        if (references.decrementAndGet() != 0) return;
        synchronized (this) {
            if (driver != null) driver.shutdownNow();
            if (info != null) info.close();
            if (shared != null) shared.close();
        }
        if (engine != null) {
            try {
                engine.savePrompts(); // the catalog file, when one was given: what this process
                // computed
            } catch (RuntimeException e) {
                Log.debug("prompt cache not saved: " + e.getMessage());
            }
            engine.close();
        }
        if (arena != null) Arenas.close(arena); // best-effort, see load0
        Log.debug("freed " + name);
    }

    synchronized ExecutorService driver() {
        if (driver == null) {
            driver =
                    Executors.newSingleThreadExecutor(
                            r -> {
                                Thread t = new Thread(r, "jinfer-" + kind.name().toLowerCase());
                                t.setDaemon(true);
                                return t;
                            });
        }
        return driver;
    }

    ChatEngine engine() {
        if (engine == null)
            throw Errors.unsupported(name + " is " + article() + ", not a language model");
        return engine;
    }

    /** The model behind the raw door: every kind but speech and transcription has states. */
    ContextModel<?, ?, ?> contextModel() {
        return switch (kind) {
            case LANGUAGE -> engine.loaded().model();
            case EMBEDDING -> embedder.model();
            case RERANKING -> reranker.model();
            default ->
                    throw Errors.unsupported(name + " is " + article() + ": it has no token state");
        };
    }

    Tokenizer tokenizer() {
        return switch (kind) {
            case LANGUAGE -> engine.loaded().tokenizer();
            case EMBEDDING -> embedder.tokenizer();
            default ->
                    throw Errors.unsupported(
                            name + " is " + article() + ": it exposes no tokenizer");
        };
    }

    String article() {
        return switch (kind) {
            case LANGUAGE -> "a language model";
            case EMBEDDING -> "an embedding model";
            case RERANKING -> "a reranking model";
            case SPEECH -> "a speech synthesis model";
            case TRANSCRIPTION -> "a transcription model";
        };
    }

    /** jinfer_embed: count x dimension, row-major; one pass over a fresh state. */
    synchronized float[] embed(List<String> texts, int dimension) {
        if (embedder == null)
            throw Errors.unsupported(name + " is " + article() + ", not an embedding model");
        if (dimension < 0) throw Errors.invalid("dimension must not be negative; 0 = native");
        int dim = dimension <= 0 ? embedder.dimension() : dimension;
        float[] all = new float[texts.size() * dim];
        int[] at = {0};
        embedder.embedAll(
                sharedState(longest(texts)),
                texts,
                dim,
                vector -> {
                    System.arraycopy(vector, 0, all, at[0], dim);
                    at[0] += dim;
                });
        return all;
    }

    /** jinfer_rerank: one score per document; a null instruction is the model's default. */
    synchronized float[] rerank(String query, List<String> documents, String instruction) {
        if (reranker == null)
            throw Errors.unsupported(name + " is " + article() + ", not a reranking model");
        String task = instruction == null ? reranker.reranker().defaultInstruction() : instruction;
        float[] scores = new float[documents.size()];
        int[] at = {0};
        reranker.scoreAll(
                sharedState(longestBytes(documents) + longestBytes(List.of(query + task)) + 64),
                task,
                query,
                documents,
                score -> scores[at[0]++] = (float) score);
        return scores;
    }

    /**
     * The token count of the longest text plus the special tokens the retrieval face wraps it in,
     * what a bidirectional forward must hold whole.
     */
    private int longest(List<String> texts) {
        int max = 0;
        for (String text : texts) max = Math.max(max, tokenizer().encode(text).length());
        return max + 16;
    }

    /**
     * An upper bound on the token count of the longest text, for the reranker whose tokenizer is
     * private to its recipe: no tokenizer spends fewer than two bytes of UTF-8 per token on text.
     */
    private static int longestBytes(List<String> texts) {
        int max = 0;
        for (String text : texts)
            max = Math.max(max, text.getBytes(java.nio.charset.StandardCharsets.UTF_8).length);
        return max / 2 + 16;
    }

    /**
     * The one state embed and rerank run in, reset before each call (the calls are serialized on
     * this model). A state per call would be simpler, but in a native image a dropped state is
     * freed only by a garbage collection (Arenas.newCrossThread is ofAuto there), ~100 MB each. The
     * state grows to the longest text seen (a bidirectional model forwards a text whole, and resets
     * between groups, so its positions never exceed one batch) and never shrinks: short texts cost
     * the default 512-token state, a long one pays for its length once.
     */
    private ContextState sharedState(int tokens) {
        int batch = Math.min(capacity, Math.max(RuntimeFlags.BATCH_CAPACITY, tokens));
        if (shared == null || shared.batchCapacity() < batch) {
            if (shared != null) shared.close();
            shared = contextModel().newState(batch, batch);
        }
        shared.reset();
        return shared;
    }

    /** jinfer_property keys, as the header numbers them. */
    int property(int key) {
        return switch (key) {
            case 1 ->
                    kind == Kind.LANGUAGE || kind == Kind.EMBEDDING
                            ? tokenizer().vocabulary().size()
                            : 0;
            case 2 -> capacity;
            case 3 -> kind == Kind.EMBEDDING ? embedder.dimension() : 0;
            case 4 ->
                    switch (kind) {
                        case SPEECH -> speech.sampleRate();
                        case TRANSCRIPTION -> transcription.sampleRate();
                        default -> 0;
                    };
            case 5 -> capacity == 0 ? 0 : contextModel().configuration().maxContextLength();
            case 100 -> kind == Kind.LANGUAGE ? 1 : 0;
            case 101 -> kind == Kind.LANGUAGE && companions.containsKey("media") ? 1 : 0;
            case 102 -> kind == Kind.EMBEDDING ? 1 : 0;
            case 103 -> kind == Kind.RERANKING ? 1 : 0;
            case 104 -> kind == Kind.SPEECH ? 1 : 0;
            case 105 -> kind == Kind.TRANSCRIPTION ? 1 : 0;
            default -> throw Errors.invalid("unknown jinfer_property key " + key);
        };
    }

    synchronized CCharPointerHolder info() {
        if (info != null) return info;
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("name", name);
        out.put("path", path.toString());
        out.put("kind", kind.name().toLowerCase());
        List<String> capabilities = new ArrayList<>();
        String[] names = {"chat", "media", "embedding", "rerank", "speech", "transcription"};
        for (int i = 0; i < names.length; i++)
            if (property(100 + i) == 1) capabilities.add(names[i]);
        out.put("capabilities", capabilities);
        out.put("kernels", JAM.providers().stream().map(JAM.Provider::id).toList());
        out.put("threads", RuntimeFlags.THREADS);
        out.put("cache_sessions", RuntimeFlags.cacheSessions());
        out.put("cache_budget", RuntimeFlags.cacheBudget());
        out.put("context_capacity", property(2));
        out.put("max_context_length", property(5));
        out.put("vocab_size", property(1));
        out.put("embedding_dimension", property(3));
        out.put("sample_rate", property(4));
        Map<String, String> files = new LinkedHashMap<>();
        companions.forEach((k, v) -> files.put(k, v.toString()));
        out.put("companions", files);
        if (kind == Kind.LANGUAGE) {
            var d = engine.loaded().samplingDefaults();
            Map<String, Object> sampling = new LinkedHashMap<>();
            sampling.put("temperature", d.temperature() == null ? Json.NULL : d.temperature());
            sampling.put("top_p", d.topP() == null ? Json.NULL : d.topP());
            sampling.put("top_k", d.topK() == null ? Json.NULL : d.topK());
            sampling.put("min_p", d.minP() == null ? Json.NULL : d.minP());
            out.put("sampling", sampling);
        }
        out.put("header", header);
        info = CTypeConversion.toCString(Json.stringify(out));
        return info;
    }
}
