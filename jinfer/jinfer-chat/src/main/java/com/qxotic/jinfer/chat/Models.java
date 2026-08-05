package com.qxotic.jinfer.chat;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.telemetry.ModelLoadEvent;
import com.qxotic.jinfer.telemetry.Telemetry;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.ServiceLoader;

/**
 * Loads any generative model, dispatching on {@code general.architecture} to the matching port via
 * {@link ModelProvider} services - the ports on the classpath define what is loadable. The one
 * "path to model" entry every consumer (server, CLI, benches) shares.
 */
public final class Models {

    private Models() {}

    private static final List<ModelProvider> PROVIDERS =
            ServiceLoader.load(ModelProvider.class).stream()
                    .map(ServiceLoader.Provider::get)
                    .toList();

    /**
     * Loads {@code path} at context size {@code ctx} (-1 = the model's full context). Weights map
     * into {@code arena}: who provides the arena owns the weights' lifetime ({@code ofAuto} =
     * GC-managed, {@code global} = process, a scoped arena = deterministic - it must outlive every
     * model sharing the weights, and closing it while any computation runs is a crash, not an
     * exception).
     */
    public static LoadedModel<?> load(Path path, int ctx, Arena arena) throws IOException {
        return open(path, (fc, gguf) -> provider(gguf).load(fc, gguf, ctx, arena));
    }

    /**
     * The one-argument bliss form: full model context, GC-managed weights ({@code Arena.ofAuto()} -
     * freed when the model becomes unreachable). Reach for the explicit overload when you need a
     * bounded context or deterministic weight lifetime.
     */
    public static LoadedModel<?> load(Path path) throws IOException {
        return load(path, -1, Arena.ofAuto());
    }

    /** As {@link #load(Path)} with a bounded context length. */
    public static LoadedModel<?> load(Path path, int ctx) throws IOException {
        return load(path, ctx, Arena.ofAuto());
    }

    /**
     * Multimodal load: the text model plus its media sidecar (mmproj GGUF with the vision/audio
     * encoders). Throws {@link UnsupportedOperationException} for architectures without one.
     */
    public static LoadedModel<?> load(Path path, Path mediaProjector, int ctx, Arena arena)
            throws IOException {
        return mediaSeeded(
                open(path, (fc, gguf) -> provider(gguf).load(fc, gguf, ctx, mediaProjector, arena)),
                mediaProjector);
    }

    /**
     * Re-roots the cache seed with the ENCODER IDENTITY: the projector file, the image decoder
     * implementation, and the model's preprocessing plan. Media blocks are content-keyed by their
     * SOURCE bytes (see {@code Batch.embeddings}), so everything between those bytes and the stored
     * KV must be part of the key space - a different projector (or decoder, or plan) producing
     * different rows for the same bytes must never serve the old blocks.
     */
    static <S extends com.qxotic.jinfer.RuntimeState> LoadedModel<S> mediaSeeded(
            LoadedModel<S> loaded, Path mediaProjector) {
        try {
            java.security.MessageDigest sha = java.security.MessageDigest.getInstance("SHA-256");
            sha.update(loaded.seed());
            sha.update(com.qxotic.jinfer.cache.PromptCache.modelSeed(mediaProjector));
            sha.update(
                    com.qxotic.jinfer.media.ImageCodec.decoder()
                            .name()
                            .getBytes(java.nio.charset.StandardCharsets.UTF_8));
            if (loaded.model() instanceof com.qxotic.jinfer.MultiModal mm) {
                sha.update(mm.encodePlanId().getBytes(java.nio.charset.StandardCharsets.UTF_8));
            }
            return new LoadedModel<>(
                    loaded.model(),
                    loaded.tokenizer(),
                    loaded.chatTemplateSource(),
                    loaded.stopTokens(),
                    sha.digest(),
                    loaded.template());
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /**
     * As {@link #load(Path, int, Arena)} but reusing an already-parsed {@code gguf} (the header is
     * not re-read) - used by AOT preload. {@code fileChannel} supplies the tensor data to mmap.
     */
    public static LoadedModel<?> load(FileChannel fileChannel, GGUF gguf, int ctx, Arena arena)
            throws IOException {
        return provider(gguf).load(fileChannel, gguf, ctx, arena);
    }

    /**
     * Loads an embedding model (an {@code EmbeddingModel} port, e.g. Qwen3-Embedding) at context
     * size {@code ctx}; same architecture dispatch as {@link #load}. Generative-only architectures
     * fail with a clear {@link UnsupportedOperationException}.
     */
    public static LoadedEmbedder<?> loadEmbedder(Path path, int ctx, Arena arena)
            throws IOException {
        return open(path, (fc, gguf) -> provider(gguf).loadEmbedder(fc, gguf, ctx, path, arena));
    }

    /**
     * Loads a RERANKER (a cross-encoder recipe over a port's backbone, e.g. Qwen3-Reranker) at
     * context size {@code ctx}; same architecture dispatch as {@link #load}. Architectures with no
     * reranker recipe fail with a clear {@link UnsupportedOperationException}. The GGUF must be the
     * reranker of its family - an architecture cannot tell its reranker and embedder apart, so a
     * wrong file scores rather than refuses.
     */
    public static LoadedReranker<?> loadReranker(Path path, int ctx, Arena arena)
            throws IOException {
        return open(path, (fc, gguf) -> provider(gguf).loadReranker(fc, gguf, ctx, path, arena));
    }

    /**
     * Loads a SPEECH model (a {@code SpeechModel} port, e.g. Inflect2); same architecture dispatch
     * as {@link #load}. Non-speech architectures fail with a clear {@link
     * UnsupportedOperationException}.
     *
     * <p>Defaults only. Anything a port lets you tune - a lexicon, a language, a family's own knobs
     * - lives on that port's own loader, typed; this is the entry for a caller that must not name
     * the port.
     */
    public static com.qxotic.jinfer.SpeechModel<?, ?, ?> loadSpeech(Path path, Arena arena)
            throws IOException {
        return open(path, (fc, gguf) -> provider(gguf).loadSpeech(fc, gguf, path, arena));
    }

    private interface Load<T> {
        T apply(FileChannel fc, GGUF gguf) throws IOException;
    }

    /**
     * Opens {@code path}, reads the GGUF header, and hands both to {@code load}. The single
     * chokepoint for every architecture and every model kind, so it is also where {@code
     * jinfer.ModelLoad} is emitted and the sampled events are installed. A port loaded directly
     * (e.g. {@code Gemma4.loadModel}) bypasses this and reports nothing.
     */
    private static <T> T open(Path path, Load<T> load) throws IOException {
        Telemetry.install();
        if (!Files.exists(path)) {
            throw new java.nio.file.NoSuchFileException(
                    path.toString(), null, "model file not found");
        }
        ModelLoadEvent event = new ModelLoadEvent();
        event.begin();
        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            fc.position(0L);
            GGUF gguf;
            try {
                gguf =
                        GGUF.read(
                                Channels.newChannel(
                                        new BufferedInputStream(
                                                Channels.newInputStream(fc), 1 << 20)));
            } catch (com.qxotic.format.gguf.GGUFFormatException e) {
                throw new IllegalArgumentException(
                        path
                                + " is not a GGUF model file ("
                                + e.getMessage()
                                + "). If this is a HuggingFace checkpoint (safetensors/pytorch),"
                                + " convert it with llama.cpp's convert_hf_to_gguf.py",
                        e);
            }
            // a SPLIT part carries only its own slice of the tensors; loading one alone would
            // build a silently WRONG model (missing weights) - refuse with the remedy instead
            long splitCount = metadataLong(gguf, "split.count");
            if (splitCount > 1) {
                throw new UnsupportedOperationException(
                        path.getFileName()
                                + " is part "
                                + (metadataLong(gguf, "split.no") + 1)
                                + " of a "
                                + splitCount
                                + "-file split GGUF - split models are not supported yet; merge the"
                                + " parts first: llama.cpp's llama-gguf-split --merge <part1>"
                                + " <out>");
            }
            T loaded = load.apply(fc, gguf);
            if (event.isEnabled()) {
                String arch = gguf.getString("general.architecture");
                event.model = path.getFileName().toString();
                event.architecture = arch;
                event.contextLength =
                        gguf.getValueOrDefault(int.class, arch + ".context_length", 0);
                event.dimensions = gguf.getValueOrDefault(int.class, arch + ".embedding_length", 0);
                event.weightsBytes = Files.size(path);
                event.mapped = true; // jinfer maps tensor data; it never reads it into the heap
            }
            return loaded;
        } finally {
            event.end();
            event.commit();
        }
    }

    /**
     * The architectures the classpath's ports claim (sorted) - what {@link #load} can dispatch. For
     * tooling and startup banners; a port that does not enumerate (default {@link
     * ModelProvider#architectures()}) dispatches fine but is absent here.
     */
    public static java.util.SortedSet<String> supportedArchitectures() {
        java.util.TreeSet<String> archs = new java.util.TreeSet<>();
        for (ModelProvider p : PROVIDERS) archs.addAll(p.architectures());
        return archs;
    }

    // arch (or prefix, for pattern-matching ports) -> the Maven artifact that provides it.
    // DIAGNOSTICS ONLY - dispatch never reads this; it exists so "unsupported architecture"
    // can name the jar to add. Kept in sync with the in-repo ports by hand; an entry for a
    // port the user has is harmless (dispatch already succeeded).
    private static final java.util.Map<String, String> PORT_ARTIFACTS =
            java.util.Map.of(
                    "gemma4", "com.qxotic:jinfer-gemma4",
                    "gpt-oss", "com.qxotic:jinfer-gptoss",
                    "lfm", "com.qxotic:jinfer-lfm2",
                    "llama", "com.qxotic:jinfer-llama",
                    "minicpm", "com.qxotic:jinfer-llama",
                    "mistral3", "com.qxotic:jinfer-llama",
                    "smollm3", "com.qxotic:jinfer-llama",
                    "nemotron_h", "com.qxotic:jinfer-nemotronh",
                    "qwen35", "com.qxotic:jinfer-qwen35",
                    "inflect", "com.qxotic:jinfer-inflect2");

    /**
     * The diagnostics table's answer for {@code arch}, or null - package-visible for the drift test
     * (every classpath port's architecture must resolve here, so a new port cannot land without its
     * remedy entry).
     */
    static String artifactFor(String arch) {
        for (var e : PORT_ARTIFACTS.entrySet()) {
            if (arch.equals(e.getKey()) || arch.startsWith(e.getKey())) return e.getValue();
        }
        return null;
    }

    /** A numeric metadata value whatever its GGUF width (split.* is UINT16 in the wild). */
    private static long metadataLong(GGUF gguf, String key) {
        if (!gguf.containsKey(key)) return 0;
        Object v = gguf.getValue(Object.class, key);
        return v instanceof Number n ? n.longValue() : 0;
    }

    /** The port claiming the GGUF's architecture; throws a REMEDY-naming error when none does. */
    private static ModelProvider provider(GGUF gguf) {
        String arch = gguf.getString("general.architecture");
        for (ModelProvider p : PROVIDERS) {
            if (p.supports(arch)) return p;
        }
        String artifact = artifactFor(arch);
        if (PROVIDERS.isEmpty()) {
            throw new IllegalArgumentException(
                    "no model providers on the classpath. This GGUF needs architecture '"
                            + arch
                            + "': add "
                            + (artifact != null
                                    ? artifact
                                    : "the com.qxotic:jinfer-* artifact that supports it")
                            + ", or com.qxotic:jinfer-models-all for everything. (Shading jinfer"
                            + " into one jar? merge META-INF/services - Maven Shade's"
                            + " ServicesResourceTransformer - or ServiceLoader finds nothing.)");
        }
        java.util.SortedSet<String> here = supportedArchitectures();
        throw new IllegalArgumentException(
                "no provider for architecture '"
                        + arch
                        + "'"
                        + (artifact != null
                                ? " on the classpath - add " + artifact
                                : " on the classpath, and none ships with this jinfer version"
                                        + " (a newer jinfer, or a third-party ModelProvider,"
                                        + " may support it)")
                        + ". Supported architectures: "
                        + (here.isEmpty()
                                ? PROVIDERS.size() + " provider(s), none enumerated"
                                : here));
    }
}
