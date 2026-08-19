package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.PanamaMemoryArena;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.hub.ModelStore;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.TokenCountEstimator;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.listener.EmbeddingModelListener;
import dev.langchain4j.model.embedding.request.EmbeddingParameter;
import dev.langchain4j.model.embedding.request.EmbeddingRequest;
import dev.langchain4j.model.embedding.request.EmbeddingRequestParameters;
import dev.langchain4j.model.embedding.response.EmbeddingResponse;
import dev.langchain4j.model.embedding.response.EmbeddingResponseMetadata;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * langchain4j {@link EmbeddingModel} backed by jinfer: in-process CPU embeddings over a local GGUF
 * (Qwen3-Embedding, LFM2.5-Embedding; any embedding port on the classpath loads via the same
 * architecture dispatch as the chat models). {@link #embedAll} packs segments into ragged batches
 * bounded by the context - one forward pass embeds many segments under segmented attention - so RAG
 * ingestion of hundreds of chunks costs a handful of prefills, not hundreds.
 *
 * <p>The standard request {@code dimensions} parameter is advertised only for Matryoshka-trained
 * ports (Qwen3: 32 through native); truncated vectors are L2-normalized. Fixed-width ports reject
 * it instead of returning an arbitrary prefix.
 *
 * <p>Token counts in the returned usage are exact (the real tokenizer, not an estimate). Run with
 * jinfer's JVM flags: {@code --enable-preview --add-modules jdk.incubator.vector
 * --enable-native-access=ALL-UNNAMED}.
 */
public final class JinferEmbeddingModel implements EmbeddingModel, AutoCloseable {

    private final LoadedEmbedder<?> loaded;
    private final ContextState state; // one reusable state; embed() resets it per group
    private final Arena arena;
    private final boolean ownsWeights; // false = the caller loaded the model and keeps the arena
    private final ReentrantLock lock = new ReentrantLock(true); // single-stream, like ChatEngine
    private static final org.slf4j.Logger LOG =
            org.slf4j.LoggerFactory.getLogger(JinferEmbeddingModel.class);
    private final AtomicBoolean hintedBareUse = new AtomicBoolean();
    private final List<EmbeddingModelListener> listeners;

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
            int modelContextLength = loaded.model().configuration().contextLength();
            int contextCapacity =
                    b.contextLength == 0
                            ? modelContextLength
                            : Math.min(b.contextLength, modelContextLength);
            this.state = newState(loaded, contextCapacity, arena);
            this.listeners = b.listeners == null ? List.of() : List.copyOf(b.listeners);
        } catch (RuntimeException | Error e) {
            Arenas.close(arena); // free before failing (best-effort: ofAuto self-manages)
            throw e;
        }
    }

    private static <S extends ContextState> S newState(LoadedEmbedder<S> l, int ctx, Arena arena) {
        return l.model().newState(ctx, new PanamaMemoryArena(arena));
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
        return builder().model(loaded).contextLength(state.contextCapacity()).build();
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

    /**
     * Token counting over THIS model's tokenizer: exact on text - for sizing splitter chunks
     * against {@code contextLength}. The embedder adds its framing tokens per embedded segment
     * (prefixTokens + suffixTokens: one trailing EOS on Qwen3, one leading BOS on LFM2.5-Embedding)
     * on top of the text count.
     */
    public TokenCountEstimator tokenCountEstimator() {
        // no media plan, no sampler: the estimator refuses media messages before any decode
        return new Estimators(loaded.tokenizer(), null, null);
    }

    /**
     * The GGUF this instance loaded, the same name every response's metadata carries - the
     * interface default answers the literal string "unknown".
     */
    @Override
    public String modelName() {
        return loaded.name();
    }

    /** The embedding width - static from the port, never probed with a forward pass. */
    @Override
    public int dimension() {
        return loaded.dimension();
    }

    /** Retrieval framing always applies; Matryoshka dimensions only when the port declares them. */
    @Override
    public Set<EmbeddingParameter<?>> supportedParameters() {
        return loaded.supportsCustomDimensions()
                ? Set.of(
                        EmbeddingRequestParameters.INPUT_TYPE,
                        EmbeddingRequestParameters.DIMENSIONS)
                : Set.of(EmbeddingRequestParameters.INPUT_TYPE);
    }

    /**
     * The typed door ({@code EmbeddingRequest.inputType()}): {@code QUERY}/{@code DOCUMENT} apply
     * the model card's retrieval framing (the port's {@code queryPrefix}/{@code documentPrefix})
     * before tokenizing - retrieval-tuned embedders are TRAINED with these prefixes, and both
     * {@code EmbeddingStoreIngestor} and {@code EmbeddingStoreContentRetriever} send the type via
     * their {@code embeddingInputType(...)} builder knob. A typeless request embeds raw text as
     * given.
     */
    /**
     * langchain4j listeners; core's default {@code embed(EmbeddingRequest)} dispatches
     * onRequest/onResponse/onError around the call ({@code EmbeddingModelListenerUtils}) as long as
     * this returns them.
     */
    @Override
    public List<EmbeddingModelListener> listeners() {
        return listeners;
    }

    @Override
    public EmbeddingResponse doEmbed(EmbeddingRequest request) {
        int outputDimension =
                request.dimensions() == null ? loaded.dimension() : request.dimensions();
        String prefix =
                switch (request.inputType()) {
                    case QUERY -> loaded.queryPrefix();
                    case DOCUMENT -> loaded.documentPrefix();
                    case null -> hintBareUse();
                };
        List<String> texts = request.inputs().stream().map(in -> prefix + in.text()).toList();
        Response<List<Embedding>> response = embedTexts(texts, outputDimension);
        return EmbeddingResponse.builder()
                .embeddings(response.content())
                .metadata(
                        EmbeddingResponseMetadata.builder()
                                .modelName(loaded.name())
                                .tokenUsage(response.tokenUsage())
                                .build())
                .build();
    }

    private Response<List<Embedding>> embedTexts(List<String> texts, int outputDimension) {
        List<Embedding> out = new ArrayList<>(texts.size());
        int total;
        lock.lock();
        try {
            total =
                    loaded.embedAll(
                            state,
                            texts,
                            outputDimension,
                            vector -> out.add(Embedding.from(vector)));
        } finally {
            lock.unlock();
        }
        return Response.from(out, new TokenUsage(total));
    }

    /**
     * The silent-degradation net: typeless traffic on a prefix-trained model is served raw (as
     * every provider serves it), but says so ONCE - naming the framework's own knob, because the
     * default ingestor/retriever wiring sends no input type at all.
     */
    private String hintBareUse() {
        if (loaded.prefixTrained() && hintedBareUse.compareAndSet(false, true)) {
            LOG.warn(
                    loaded.name()
                            + " is prefix-trained for retrieval, but this request stated no input"
                            + " type, so raw text was embedded as given. For retrieval-quality"
                            + " vectors set .embeddingInputType(QUERY) on"
                            + " EmbeddingStoreContentRetriever and DOCUMENT on"
                            + " EmbeddingStoreIngestor. (noted once per model)");
        }
        return "";
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Object source; // Path | ref/URL String | LoadedEmbedder: the last setter wins
        private Path modelPath; // derived from source at build()
        private LoadedEmbedder<?> loaded; // derived from source at build()
        private int contextLength = 2048;
        private List<EmbeddingModelListener> listeners;

        /** langchain4j listeners; dispatched around every {@code embed(EmbeddingRequest)}. */
        public Builder listeners(List<EmbeddingModelListener> listeners) {
            this.listeners = listeners;
            return this;
        }

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

        public JinferEmbeddingModel build() {
            modelPath = null;
            loaded = null;
            switch (source) {
                case String ref -> modelPath = ModelStore.standard().resolve(ref);
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
