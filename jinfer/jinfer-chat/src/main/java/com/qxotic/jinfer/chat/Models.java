package com.qxotic.jinfer.chat;

import com.qxotic.format.gguf.GGUF;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
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
     * Multimodal load: the text model plus its media sidecar (mmproj GGUF with the vision/audio
     * encoders). Throws {@link UnsupportedOperationException} for architectures without one.
     */
    public static LoadedModel<?> load(Path path, Path mediaProjector, int ctx, Arena arena)
            throws IOException {
        return open(path, (fc, gguf) -> provider(gguf).load(fc, gguf, ctx, mediaProjector, arena));
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
        return open(path, (fc, gguf) -> provider(gguf).loadEmbedder(fc, gguf, ctx, arena));
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
        return open(path, (fc, gguf) -> provider(gguf).loadReranker(fc, gguf, ctx, arena));
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

    /** Opens {@code path}, reads the GGUF header, and hands both to {@code load}. */
    private static <T> T open(Path path, Load<T> load) throws IOException {
        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            fc.position(0L);
            GGUF gguf =
                    GGUF.read(
                            Channels.newChannel(
                                    new BufferedInputStream(Channels.newInputStream(fc), 1 << 20)));
            return load.apply(fc, gguf);
        }
    }

    /** The port claiming the GGUF's architecture; throws when no port on the classpath does. */
    private static ModelProvider provider(GGUF gguf) {
        String arch = gguf.getString("general.architecture");
        for (ModelProvider p : PROVIDERS) {
            if (p.supports(arch)) return p;
        }
        throw new IllegalArgumentException(
                "unsupported architecture '"
                        + arch
                        + "' ("
                        + PROVIDERS.size()
                        + " ports on the classpath)");
    }
}
