package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.PanamaMemoryArena;
import com.qxotic.jinfer.boundary.Arenas;
import com.qxotic.jinfer.boundary.ContextState;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.hub.ModelStore;
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
import org.springframework.ai.embedding.BatchingStrategy;
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
 * (Qwen3-Embedding, LFM2.5-Embedding; any embedding port on the classpath loads via the same
 * architecture dispatch as the chat models). {@link #call} packs inputs into ragged batches bounded
 * by the context - one forward pass embeds many sequences under segmented attention - so RAG
 * ingestion of hundreds of chunks costs a handful of prefills, not hundreds.
 *
 * <p>{@link EmbeddingOptions#getDimensions()} is honored only for Matryoshka-trained ports (Qwen3:
 * 32 through native); truncated vectors are L2-normalized. Fixed-width ports reject it instead of
 * returning an arbitrary prefix.
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
    private final ContextState state; // one reusable state; embed() resets it per group
    private final Arena arena;
    private final boolean ownsWeights; // false = the caller loaded the model and keeps the arena
    private final ObservationRegistry observationRegistry;
    private final EmbeddingModelObservationConvention observationConvention;
    private final ReentrantLock lock = new ReentrantLock(true); // single-stream, like ChatEngine

    private JinferEmbeddingModel(Builder b) {
        // ONE private arena. The state borrows it; close() closes the state first, then the arena.
        // Weights land in it only when THIS instance loads them; caller-loaded weights stay in the
        // caller's arena.
        this.arena = Arenas.newCrossThread();
        try {
            this.ownsWeights = b.loaded == null;
            try {
                this.loaded = b.loaded != null ? b.loaded : Models.loadEmbedder(b.modelPath, arena);
            } catch (IOException e) {
                throw new UncheckedIOException("failed to load " + b.modelPath, e);
            }
            this.modelName = loaded.name();
            int modelContextLength = loaded.model().configuration().contextLength();
            int contextCapacity =
                    b.contextLength == 0
                            ? modelContextLength
                            : Math.min(b.contextLength, modelContextLength);
            this.state = newState(loaded, contextCapacity, arena);
            this.observationRegistry =
                    b.observationRegistry == null
                            ? ObservationRegistry.NOOP
                            : b.observationRegistry;
            this.observationConvention = b.observationConvention;
        } catch (RuntimeException | Error e) {
            Arenas.close(arena); // free before failing (best-effort: ofAuto self-manages)
            throw e;
        }
    }

    /**
     * A parallel pipeline over the same weights: fresh state, own lock, own lifecycle. Only a model
     * whose weights YOU loaded can fork - the weights' lifetime is your arena's, so a fork can
     * never dangle. A model that loaded its own weights refuses: it frees them at {@link #close()},
     * and a fork would outlive them.
     */
    public JinferEmbeddingModel fork() {
        if (ownsWeights) {
            throw new IllegalStateException(
                    "this model owns its weights and frees them at close - a fork would dangle."
                            + " Load once into YOUR arena instead: Models.loadEmbedder(path,"
                            + " arena), build with model(loaded), then fork freely");
        }
        return builder()
                .model(loaded)
                .contextLength(state.contextCapacity())
                .observationRegistry(observationRegistry)
                .observationConvention(observationConvention)
                .build();
    }

    private static <S extends ContextState> S newState(LoadedEmbedder<S> l, int ctx, Arena arena) {
        return l.model().newState(ctx, new PanamaMemoryArena(arena));
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
            state.close();
            Arenas.close(arena);
        } finally {
            lock.unlock();
        }
    }

    /** The embedding width - static from the port, never probed with a forward pass. */
    @Override
    public int dimensions() {
        return loaded.dimension();
    }

    // ---- the retrieval seam: Spring AI has no input-type parameter, but the INTERFACE types
    // state the intent - stores call embed(Document)/embed(List<Document>,...) to ingest and
    // embed(String) to search. Mapping the model card's framing onto those types makes every
    // VectorStore retrieval-correct on BOTH sides through the single bean Spring wires.
    // call(EmbeddingRequest) stays the raw, framing-free door. ----

    /** Ingestion side: the card's document framing is prepended (LFM2.5: {@code "document: "}). */
    @Override
    public float[] embed(Document document) {
        return embedOne(loaded.documentPrefix() + text(document));
    }

    /**
     * Ingestion side, the batching route stores actually use: document framing per document, the
     * strategy's batching preserved.
     */
    @Override
    public List<float[]> embed(
            List<Document> documents, EmbeddingOptions options, BatchingStrategy strategy) {
        List<float[]> all = new ArrayList<>(documents.size());
        for (List<Document> batch : strategy.batch(documents)) {
            List<String> texts =
                    batch.stream().map(d -> loaded.documentPrefix() + text(d)).toList();
            for (Embedding e : call(new EmbeddingRequest(texts, options)).getResults()) {
                all.add(e.getOutput());
            }
        }
        return all;
    }

    /**
     * Query side: in a store-centric API the lone string IS the query ({@code similaritySearch}
     * routes here), so the card's query framing is prepended (LFM2.5: {@code "query: "}; Qwen3: its
     * instructed-query form). For framing-free vectors use {@link #call} directly.
     */
    @Override
    public float[] embed(String text) {
        return embedOne(loaded.queryPrefix() + text);
    }

    /** An embedder embeds TEXT; a media document silently becoming "null" would be a lie. */
    private static String text(Document document) {
        if (!document.isText()) {
            throw new IllegalArgumentException(
                    "document "
                            + document.getId()
                            + " carries media, not text: an embedder embeds text");
        }
        return document.getText();
    }

    private float[] embedOne(String text) {
        return call(new EmbeddingRequest(List.of(text), EmbeddingOptions.builder().build()))
                .getResults()
                .get(0)
                .getOutput();
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
        Integer requestedDimension = options == null ? null : options.getDimensions();
        int outputDimension = requestedDimension == null ? loaded.dimension() : requestedDimension;
        List<String> inputs = request.getInstructions();
        List<Embedding> out = new ArrayList<>(inputs.size());
        int total;
        lock.lock();
        try {
            total =
                    loaded.embedAll(
                            state,
                            inputs,
                            outputDimension,
                            vector -> out.add(new Embedding(vector, out.size())));
        } finally {
            lock.unlock();
        }
        return new EmbeddingResponse(out, metadata(total));
    }

    private EmbeddingResponseMetadata metadata(int totalTokens) {
        return new EmbeddingResponseMetadata(modelName, new DefaultUsage(totalTokens, 0));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Object source; // Path | ref/URL String | LoadedEmbedder: the last setter wins
        private Path modelPath; // derived from source at build()
        private LoadedEmbedder<?> loaded; // derived from source at build()
        private int contextLength = 2048;
        private ObservationRegistry observationRegistry;
        private EmbeddingModelObservationConvention observationConvention;

        public Builder modelPath(Path modelPath) {
            this.source = modelPath;
            return this;
        }

        /**
         * The model as ONE string: a local GGUF path, a hub ref ({@code hf.co/owner/repo:Q4_K_M})
         * or a pasted browser URL - resolved by {@link #build()} with the rest of the load, so a
         * remote ref downloads there (see the package doc) and the chain never blocks.
         */
        public Builder model(String pathOrRef) {
            this.source = pathOrRef;
            return this;
        }

        /**
         * A model you loaded yourself ({@code Models.loadEmbedder(path, arena)}) - the
         * weight-sharing seam: several instances (and their {@link #fork() forks}) over ONE loaded
         * copy are parallel pipelines for the price of one load.
         *
         * <p>You own its weights arena: {@link JinferEmbeddingModel#close()} frees only this
         * instance's state, so close your arena after every instance built on it, never before.
         * Getting the order wrong sequentially is caught fail-fast (a safety canary throws
         * IllegalStateException at the next request); freeing the arena DURING a request is a data
         * race and can still crash the VM.
         */
        public Builder model(LoadedEmbedder<?> loaded) {
            this.source = loaded;
            return this;
        }

        /**
         * Upper bound on the packing window and on each embedded sequence, in tokens. The default
         * is 2048. A larger value admits longer sequences and can pack more sequences into one
         * forward pass, at the cost of a larger resident state. {@code 0} uses the model's declared
         * context length; otherwise the effective capacity is the smaller of this value and that
         * length.
         *
         * @throws IllegalArgumentException if {@code contextLength < 0}
         */
        public Builder contextLength(int contextLength) {
            if (contextLength < 0)
                throw new IllegalArgumentException(
                        "contextLength must be >= 0 (0 uses the model maximum): " + contextLength);
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
            modelPath = null;
            loaded = null;
            switch (source) {
                case String ref -> modelPath = ModelStore.resolve(ref);
                case Path path -> modelPath = path;
                case LoadedEmbedder<?> l -> loaded = l;
                case null, default ->
                        throw new IllegalArgumentException(
                                "a model is required: model(\"hf.co/owner/repo:Q4_K_M\"),"
                                        + " modelPath(...) or model(LoadedEmbedder)");
            }
            return new JinferEmbeddingModel(this);
        }
    }
}
