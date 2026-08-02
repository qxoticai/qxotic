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
     * Loads a GENERATIVE model from an already-parsed GGUF; {@code fileChannel} supplies the tensor
     * data, mapped into {@code arena} (who provides the arena owns the weights' lifetime; it must
     * outlive every model sharing them). {@code contextLength} -1 means the model's full context.
     *
     * <p>Every capability here is optional and {@link #supports} is the only requirement: a port
     * overrides the loads its architecture actually has, and a speech-only or embedding-only port
     * keeps the rest of these defaults.
     */
    default LoadedModel<?> load(FileChannel fileChannel, GGUF gguf, int contextLength, Arena arena)
            throws IOException {
        throw new UnsupportedOperationException(
                "'"
                        + gguf.getString("general.architecture")
                        + "' is not a generative architecture");
    }

    /**
     * As {@link #load(FileChannel, GGUF, int, Arena)} plus the architecture's media sidecar
     * (llama.cpp's mmproj convention: vision/audio encoders in a separate GGUF). Ports without
     * media support keep this default.
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

    /**
     * Loads a SPEECH model from an already-parsed GGUF ({@link Models#loadSpeech}) at the port's
     * own defaults. Ports whose architectures do not synthesize speech keep this default.
     *
     * <p>{@code path} is where the GGUF lives, passed because a speech front end has companions the
     * container does not carry - a phoneme port looks for its pronunciation lexicon beside the
     * model before falling back.
     */
    default com.qxotic.jinfer.SpeechModel<?, ?, ?> loadSpeech(
            FileChannel fileChannel, GGUF gguf, Path path, Arena arena) throws IOException {
        throw new UnsupportedOperationException(
                "'" + gguf.getString("general.architecture") + "' is not a speech architecture");
    }

    /**
     * Loads a RERANKER from an already-parsed GGUF ({@link Models#loadReranker}): the backbone plus
     * this family's {@link Reranker} recipe. Ports with no reranker in the family keep this
     * default.
     */
    default LoadedReranker<?> loadReranker(
            FileChannel fileChannel, GGUF gguf, int contextLength, Arena arena) throws IOException {
        throw new UnsupportedOperationException(
                "'" + gguf.getString("general.architecture") + "' is not a reranker architecture");
    }
}
