package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.BaseState;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.RuntimeFlags;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.telemetry.InferenceEvent;
import io.micrometer.observation.ObservationRegistry;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
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
public final class JinferEmbeddingModel implements EmbeddingModel, AutoCloseable {

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
        // ONE arena for weights and state, adopted by the state: state.close() frees everything
        // (idempotent, blocking, Cleaner-backstopped - all BaseState's laws, implemented once)
        Arena arena = Arena.ofShared();
        try {
            try {
                // same contract as the chat builder: <= 0 means the model's own maximum (-1 to the
                // loader); a literal 0 would crash the port's tensor sizing
                this.loaded =
                        Models.loadEmbedder(
                                b.modelPath, b.contextLength <= 0 ? -1 : b.contextLength, arena);
            } catch (IOException e) {
                throw new UncheckedIOException("failed to load " + b.modelPath, e);
            }
            this.modelName = b.modelPath.getFileName().toString();
            this.contextLength = loaded.model().config().contextLength();
            this.state = newState(loaded, contextLength, arena);
            this.observationRegistry =
                    b.observationRegistry == null
                            ? ObservationRegistry.NOOP
                            : b.observationRegistry;
            this.observationConvention = b.observationConvention;
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
        List<Embedding> out = new ArrayList<>(inputs.size());
        int dim = loaded.dimension();
        int total;
        InferenceEvent event =
                InferenceEvent.started(modelName, InferenceEvent.EMBEDDINGS, InferenceEvent.TEXT);
        long startNanos = System.nanoTime();
        lock.lock();
        try {
            total =
                    loaded.embedAll(
                            state,
                            contextLength,
                            inputs,
                            v -> out.add(toEmbedding(v, dim, truncate, out.size())));
            event.inputTokens = total;
        } catch (RuntimeException | Error failure) {
            event.errorType = failure.getClass().getSimpleName();
            throw failure;
        } finally {
            lock.unlock();
            // an encode has no decode loop, so outputTokens and decodeTime are a true zero here
            event.prefillTime = System.nanoTime() - startNanos;
            event.end();
            event.commit();
        }
        return new EmbeddingResponse(out, metadata(total));
    }

    private EmbeddingResponseMetadata metadata(int totalTokens) {
        return new EmbeddingResponseMetadata(modelName, new DefaultUsage(totalTokens, 0));
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
