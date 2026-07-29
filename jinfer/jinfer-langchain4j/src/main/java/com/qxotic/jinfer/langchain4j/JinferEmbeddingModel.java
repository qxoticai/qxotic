package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.BaseState;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.RuntimeFlags;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.Models;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

/**
 * langchain4j {@link EmbeddingModel} backed by jinfer: in-process CPU embeddings over a local GGUF
 * (the Qwen3-Embedding family; any embedding port on the classpath loads via the same architecture
 * dispatch as the chat models). {@link #embedAll} packs segments into ragged batches bounded by the
 * context - one forward pass embeds many segments under segmented attention - so RAG ingestion of
 * hundreds of chunks costs a handful of prefills, not hundreds.
 *
 * <p>Token counts in the returned usage are exact (the real tokenizer, not an estimate). Run with
 * jinfer's JVM flags: {@code --enable-preview --add-modules jdk.incubator.vector
 * --enable-native-access=ALL-UNNAMED}.
 */
public final class JinferEmbeddingModel implements EmbeddingModel, AutoCloseable {

    private final LoadedEmbedder<?> loaded;
    private final RuntimeState state; // one reusable state; embed() resets it per group
    private final int contextLength;
    private final ReentrantLock lock = new ReentrantLock(true); // single-stream, like ChatEngine

    private JinferEmbeddingModel(Builder b) {
        // ONE arena for weights and state, adopted by the state: state.close() frees everything
        // (idempotent, blocking, Cleaner-backstopped - all BaseState's laws, implemented once)
        Arena arena = Arena.ofShared();
        try {
            try {
                // same contract as the chat builders: <= 0 means the model's own maximum (-1 to the
                // loader); a literal 0 would crash the port's tensor sizing
                this.loaded =
                        Models.loadEmbedder(
                                b.modelPath, b.contextLength <= 0 ? -1 : b.contextLength, arena);
            } catch (IOException e) {
                throw new UncheckedIOException("failed to load " + b.modelPath, e);
            }
            this.contextLength = loaded.model().config().contextLength();
            this.state = newState(loaded, contextLength, arena);
        } catch (RuntimeException | Error e) {
            arena.close(); // a leaked ofShared arena has no Cleaner: free before failing
            throw e;
        }
    }

    private static <S extends RuntimeState> S newState(LoadedEmbedder<S> l, int ctx, Arena arena) {
        return l.model().newState(ctx, RuntimeFlags.BATCH_CAPACITY, arena, true);
    }

    /**
     * Blocking, idempotent: waits out any in-flight embed (the fair lock), then frees the single
     * arena holding weights and state deterministically; later calls fail with
     * IllegalStateException.
     */
    @Override
    public void close() {
        lock.lock();
        try {
            ((BaseState) state).close();
        } finally {
            lock.unlock();
        }
    }

    /**
     * Token counting over THIS model's tokenizer: exact on text - for sizing splitter chunks
     * against {@code contextLength}. The embedder adds {@code sequenceSuffix} tokens per embedded
     * segment (one EOS on Qwen3) on top of the text count.
     */
    public dev.langchain4j.model.TokenCountEstimator tokenCountEstimator() {
        return new Estimators(loaded.tokenizer(), null);
    }

    /** The embedding width - static from the port, never probed with a forward pass. */
    @Override
    public int dimension() {
        return loaded.dimension();
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
        List<String> texts = segments.stream().map(TextSegment::text).toList();
        List<Embedding> out = new ArrayList<>(segments.size());
        int dim = loaded.dimension();
        int total;
        lock.lock();
        try {
            total = loaded.embedAll(state, contextLength, texts, v -> out.add(toEmbedding(v, dim)));
        } finally {
            lock.unlock();
        }
        return Response.from(out, new TokenUsage(total));
    }

    private static Embedding toEmbedding(FloatTensor vector, int dim) {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) v[i] = vector.getFloat(i);
        return Embedding.from(v);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Path modelPath;
        private int contextLength = 2048;

        public Builder modelPath(Path modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        /**
         * The packing window and per-segment ceiling (default 2048): larger packs more segments per
         * forward pass and admits longer segments, at the cost of a bigger resident KV state.
         * {@code <= 0} = the model's own maximum.
         */
        public Builder contextLength(int contextLength) {
            this.contextLength = contextLength;
            return this;
        }

        public JinferEmbeddingModel build() {
            if (modelPath == null) throw new IllegalArgumentException("modelPath is required");
            return new JinferEmbeddingModel(this);
        }
    }
}
