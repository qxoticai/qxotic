package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.FloatTensor;
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
public final class JinferEmbeddingModel implements EmbeddingModel {

    private final LoadedEmbedder<?> loaded;
    private final RuntimeState state; // one reusable state; embed() resets it per group
    private final int contextLength;
    private final ReentrantLock lock = new ReentrantLock(true); // single-stream, like ChatEngine

    private JinferEmbeddingModel(Builder b) {
        try {
            // same contract as the chat builders: <= 0 means the model's own maximum (-1 to the
            // loader); a literal 0 would crash the port's tensor sizing
            this.loaded =
                    Models.loadEmbedder(b.modelPath, b.contextLength <= 0 ? -1 : b.contextLength);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to load " + b.modelPath, e);
        }
        this.contextLength = loaded.model().config().contextLength();
        this.state = newState(loaded, contextLength);
    }

    private static <S extends RuntimeState> S newState(LoadedEmbedder<S> l, int ctx) {
        return l.model().newState(ctx);
    }

    /** The embedding width - static from the port, never probed with a forward pass. */
    @Override
    public int dimension() {
        return loaded.dimension();
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
        if (segments.isEmpty()) {
            return Response.from(List.of(), new TokenUsage(0));
        }
        int[] suffix = loaded.sequenceSuffix();
        int[][] seqs = new int[segments.size()][];
        int total = 0;
        for (int i = 0; i < segments.size(); i++) {
            List<Integer> ids = loaded.tokenizer().encode(segments.get(i).text()).toList();
            int[] seq = new int[ids.size() + suffix.length];
            for (int j = 0; j < ids.size(); j++) seq[j] = ids.get(j);
            System.arraycopy(suffix, 0, seq, ids.size(), suffix.length);
            if (seq.length > contextLength) {
                throw new IllegalArgumentException(
                        "segment "
                                + i
                                + " is "
                                + seq.length
                                + " tokens, over the "
                                + contextLength
                                + "-token context - raise contextLength(...) or chunk smaller");
            }
            seqs[i] = seq;
            total += seq.length;
        }
        List<Embedding> out = new ArrayList<>(segments.size());
        lock.lock();
        try {
            // greedy packing: each group fills the context, one forward pass per group
            int start = 0;
            while (start < seqs.length) {
                int end = start, packed = 0;
                while (end < seqs.length && packed + seqs[end].length <= contextLength) {
                    packed += seqs[end].length;
                    end++;
                }
                embedGroup(loaded, state, seqs, start, end, packed, out);
                start = end;
            }
        } finally {
            lock.unlock();
        }
        return Response.from(out, new TokenUsage(total));
    }

    private static <S extends RuntimeState> void embedGroup(
            LoadedEmbedder<S> l,
            RuntimeState state,
            int[][] seqs,
            int start,
            int end,
            int packed,
            List<Embedding> out) {
        int[] ids = new int[packed];
        int[] len = new int[end - start];
        int at = 0;
        for (int i = start; i < end; i++) {
            System.arraycopy(seqs[i], 0, ids, at, seqs[i].length);
            at += seqs[i].length;
            len[i - start] = seqs[i].length;
        }
        @SuppressWarnings("unchecked")
        S s = (S) state;
        int dim = l.dimension();
        l.model()
                .embed(
                        s,
                        new Batch.Input.Sequences(new Batch.Input.Tokens(ids), len),
                        vector -> out.add(toEmbedding(vector, dim)));
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
