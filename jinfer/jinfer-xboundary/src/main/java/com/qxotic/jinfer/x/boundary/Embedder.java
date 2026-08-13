package com.qxotic.jinfer.x.boundary;

import com.qxotic.jota.memory.MemoryView;
import java.util.function.Consumer;

/** A model-paired converter from decoded media to model-dimension embedding rows. */
public interface Embedder<R extends Media> {

    /**
     * Streams dense FP32 {@code [rows, modelDim]} views in chunks of at most {@code maxChunkSize}.
     * An encoder whose output is one atomic attention block rejects an insufficient chunk size.
     * Each view is borrowed from embedder-owned storage and is valid only for the duration of the
     * sink invocation: once the callback returns, neither the buffer's liveness nor its contents
     * are guaranteed (the arena may be freed, the next chunk may overwrite the same storage). A
     * sink that needs the rows later must copy them during the call; it must never close the
     * backing memory.
     */
    void embed(R source, int maxChunkSize, Consumer<MemoryView<?>> sink);

    /** Best-effort planned row count without running the encoder. */
    default int positions(R source) {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " does not plan media positions");
    }
}
