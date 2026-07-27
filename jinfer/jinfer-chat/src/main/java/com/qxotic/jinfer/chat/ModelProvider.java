package com.qxotic.jinfer.chat;

import com.qxotic.format.gguf.GGUF;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;

/**
 * One port's entry in the architecture dispatch: a {@link java.util.ServiceLoader} service each
 * port module registers (META-INF/services), so {@link Models#load} finds exactly the ports on the
 * classpath - no hand-maintained arch table in every consumer.
 */
public interface ModelProvider {

    /** Whether this port loads GGUFs with the given {@code general.architecture}. */
    boolean supports(String architecture);

    /**
     * Loads the model from an already-parsed GGUF; {@code fileChannel} supplies the tensor data,
     * mapped into {@code arena} (who provides the arena owns the weights' lifetime; it must outlive
     * every model sharing them). {@code contextLength} -1 means the model's full context.
     */
    LoadedModel<?> load(FileChannel fileChannel, GGUF gguf, int contextLength, Arena arena)
            throws IOException;

    /**
     * As {@link #load(FileChannel, GGUF, int)} plus the architecture's media sidecar (llama.cpp's
     * mmproj convention: vision/audio encoders in a separate GGUF). Ports without media support
     * keep this default.
     */
    default LoadedModel<?> load(
            FileChannel fileChannel, GGUF gguf, int contextLength, Path mediaProjector, Arena arena)
            throws IOException {
        throw new UnsupportedOperationException("this architecture has no media sidecar support");
    }

    /**
     * Loads an EMBEDDING model from an already-parsed GGUF ({@link Models#loadEmbedder}). Ports
     * whose architectures are generative-only keep this default.
     */
    default LoadedEmbedder<?> loadEmbedder(
            FileChannel fileChannel, GGUF gguf, int contextLength, Arena arena) throws IOException {
        throw new UnsupportedOperationException(
                "'"
                        + gguf.getString("general.architecture")
                        + "' is not an embedding architecture");
    }
}
