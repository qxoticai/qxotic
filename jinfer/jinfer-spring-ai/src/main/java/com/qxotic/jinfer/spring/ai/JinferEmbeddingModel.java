package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.Models;
import io.micrometer.observation.ObservationRegistry;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;
import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.embedding.EmbeddingResponseMetadata;
import org.springframework.ai.embedding.observation.DefaultEmbeddingModelObservationConvention;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationContext;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationConvention;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationDocumentation;

/**
 * Spring AI {@link EmbeddingModel} backed by jinfer: in-process CPU embeddings over a local GGUF
 * (the Qwen3-Embedding family; any embedding port on the classpath loads via the same architecture
 * dispatch as the chat models). {@link #call} packs inputs into ragged batches bounded by the
 * context - one forward pass embeds many sequences under segmented attention - so RAG ingestion of
 * hundreds of chunks costs a handful of prefills, not hundreds.
 *
 * <p>Token counts in the returned usage are exact (the real tokenizer, not an estimate). Run with
 * jinfer's JVM flags: {@code --enable-preview --add-modules jdk.incubator.vector
 * --enable-native-access=ALL-UNNAMED}.
 */
public final class JinferEmbeddingModel implements EmbeddingModel {

    private static final String PROVIDER = "jinfer";
    private static final EmbeddingModelObservationConvention DEFAULT_CONVENTION =
            new DefaultEmbeddingModelObservationConvention();

    private final LoadedEmbedder<?> loaded;
    final String modelName;
    private final RuntimeState state; // one reusable state; embed() resets it per group
    private final int contextLength;
    private final ObservationRegistry observationRegistry;
    private final EmbeddingModelObservationConvention observationConvention;
    private final ReentrantLock lock = new ReentrantLock(true); // single-stream, like ChatEngine

    private JinferEmbeddingModel(Builder b) {
        try {
            // same contract as the chat builder: <= 0 means the model's own maximum (-1 to the
            // loader); a literal 0 would crash the port's tensor sizing
            this.loaded =
                    Models.loadEmbedder(b.modelPath, b.contextLength <= 0 ? -1 : b.contextLength);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to load " + b.modelPath, e);
        }
        this.modelName = b.modelPath.getFileName().toString();
        this.contextLength = loaded.model().config().contextLength();
        this.state = newState(loaded, contextLength);
        this.observationRegistry =
                b.observationRegistry == null ? ObservationRegistry.NOOP : b.observationRegistry;
        this.observationConvention = b.observationConvention;
    }

    private static <S extends RuntimeState> S newState(LoadedEmbedder<S> l, int ctx) {
        return l.model().newState(ctx);
    }

    /** The embedding width - static from the port, never probed with a forward pass. */
    @Override
    public int dimensions() {
        return loaded.dimension();
    }

    @Override
    public float[] embed(Document document) {
        return embed(document.getText());
    }

    @Override
    public EmbeddingResponse call(EmbeddingRequest request) {
        EmbeddingModelObservationContext observationContext =
                EmbeddingModelObservationContext.builder()
                        .embeddingRequest(request)
                        .provider(PROVIDER)
                        .build();
        return EmbeddingModelObservationDocumentation.EMBEDDING_MODEL_OPERATION
                .observation(
                        observationConvention,
                        DEFAULT_CONVENTION,
                        () -> observationContext,
                        observationRegistry)
                .observe(
                        () -> {
                            EmbeddingResponse response = doCall(request);
                            observationContext.setResponse(response);
                            return response;
                        });
    }

    private EmbeddingResponse doCall(EmbeddingRequest request) {
        EmbeddingOptions options = request.getOptions();
        if (options != null
                && options.getModel() != null
                && !options.getModel().equals(modelName)) {
            throw new IllegalArgumentException(
                    "per-request model is not supported: this model IS '"
                            + modelName
                            + "' (one loaded GGUF per instance)");
        }
        int truncate =
                options != null && options.getDimensions() != null
                        ? options.getDimensions()
                        : dimensions();
        List<String> inputs = request.getInstructions();
        if (inputs.isEmpty()) {
            return new EmbeddingResponse(List.of(), metadata(0));
        }
        int[] suffix = loaded.sequenceSuffix();
        int[][] seqs = new int[inputs.size()][];
        int total = 0;
        for (int i = 0; i < inputs.size(); i++) {
            List<Integer> ids = loaded.tokenizer().encode(inputs.get(i)).toList();
            int[] seq = new int[ids.size() + suffix.length];
            for (int j = 0; j < ids.size(); j++) seq[j] = ids.get(j);
            System.arraycopy(suffix, 0, seq, ids.size(), suffix.length);
            if (seq.length > contextLength) {
                throw new IllegalArgumentException(
                        "input "
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
        List<Embedding> out = new ArrayList<>(inputs.size());
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
                embedGroup(loaded, state, seqs, start, end, packed, truncate, out);
                start = end;
            }
        } finally {
            lock.unlock();
        }
        return new EmbeddingResponse(out, metadata(total));
    }

    private EmbeddingResponseMetadata metadata(int totalTokens) {
        return new EmbeddingResponseMetadata(modelName, new DefaultUsage(totalTokens, 0));
    }

    private static <S extends RuntimeState> void embedGroup(
            LoadedEmbedder<S> l,
            RuntimeState state,
            int[][] seqs,
            int start,
            int end,
            int packed,
            int truncate,
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
        int[] index = {out.size()};
        l.model()
                .embed(
                        s,
                        new Batch.Input.Sequences(new Batch.Input.Tokens(ids), len),
                        vector -> out.add(toEmbedding(vector, dim, truncate, index[0]++)));
    }

    private static Embedding toEmbedding(FloatTensor vector, int dim, int truncate, int index) {
        float[] v = new float[Math.min(dim, truncate)];
        for (int i = 0; i < v.length; i++) v[i] = vector.getFloat(i);
        return new Embedding(v, index);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Path modelPath;
        private int contextLength = 2048;
        private ObservationRegistry observationRegistry;
        private EmbeddingModelObservationConvention observationConvention;

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

        /** Metrics/tracing registry; default {@link ObservationRegistry#NOOP} (zero cost). */
        public Builder observationRegistry(ObservationRegistry observationRegistry) {
            this.observationRegistry = observationRegistry;
            return this;
        }

        /** Custom observation convention; default is Spring AI's. */
        public Builder observationConvention(EmbeddingModelObservationConvention convention) {
            this.observationConvention = convention;
            return this;
        }

        public JinferEmbeddingModel build() {
            if (modelPath == null) throw new IllegalArgumentException("modelPath is required");
            return new JinferEmbeddingModel(this);
        }
    }
}
