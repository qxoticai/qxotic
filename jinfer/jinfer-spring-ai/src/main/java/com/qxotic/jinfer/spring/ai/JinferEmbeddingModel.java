package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.VideoCodec;
import com.qxotic.jinfer.codecs.VideoSampler;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.Multimodal;
import com.qxotic.jota.memory.MemoryAllocators;
import io.micrometer.observation.ObservationRegistry;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;
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
 * <p>A multimodal embedder (EmbeddingGemma 2) with its projector attached - {@code
 * companion("media", ...)} - also embeds media {@link Document}s (image, audio or video) as
 * documents, in the vector space text queries search. Spring AI's vector stores still refuse
 * non-text documents at {@code add}, so media vectors come from this model directly.
 *
 * <p>Token counts in the returned usage are exact (the real tokenizer, not an estimate). Run with
 * jinfer's JVM flags: {@code --add-modules jdk.incubator.vector
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
    private final Set<String> media; // the MIME types (image, audio, video) the projector takes

    /**
     * EmbeddingGemma 2's sampling, one frame a second and at most 32 (ponytail: the only video
     * embedder, as in the server; a per-model sampler when a second one arrives).
     */
    private static final VideoSampler EMBEDDING_FRAMES =
            video -> VideoCodec.ffmpeg().every(video, Duration.ofSeconds(1), 32);

    private JinferEmbeddingModel(Builder b) {
        // ONE private arena. The state borrows it; close() closes the state first, then the arena.
        // Weights land in it only when THIS instance loads them; caller-loaded weights stay in the
        // caller's arena.
        this.arena = Arenas.newCrossThread();
        try {
            this.ownsWeights = b.loaded == null;
            try {
                this.loaded =
                        b.loaded != null
                                ? b.loaded
                                : Models.loadEmbedder(b.modelPath, arena, b.companionPaths, null);
            } catch (IOException e) {
                throw new UncheckedIOException("failed to load " + b.modelPath, e);
            }
            this.modelName = loaded.name();
            this.media = media(loaded.model());
            int maxContextLength = loaded.model().configuration().maxContextLength();
            int contextCapacity =
                    b.contextCapacity == 0
                            ? maxContextLength
                            : Math.min(b.contextCapacity, maxContextLength);
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
                .contextCapacity(state.contextCapacity())
                .observationRegistry(observationRegistry)
                .observationConvention(observationConvention)
                .build();
    }

    /** What the attached projector ingests; video rides the image projector. */
    private static Set<String> media(Object model) {
        Set<String> kinds = new LinkedHashSet<>();
        if (model instanceof Multimodal multimodal) {
            if (multimodal.projector(Media.Image.class).isPresent()) {
                kinds.add("image");
                kinds.add("video");
            }
            if (multimodal.projector(Media.Audio.class).isPresent()) kinds.add("audio");
        }
        return Collections.unmodifiableSet(kinds);
    }

    private static <S extends ContextState> S newState(LoadedEmbedder<S> l, int ctx, Arena arena) {
        return l.model().newState(ctx, MemoryAllocators.ofArena(arena));
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
            closeStateThenArena(state, arena);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Closes {@code state}, then {@code arena} - the arena even when the state's cleanup throws, so
     * a failing state never strands the weights it borrowed from.
     */
    static void closeStateThenArena(RuntimeState state, Arena arena) {
        try {
            state.close();
        } finally {
            Arenas.close(arena);
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

    /**
     * Ingestion side: the card's document framing is prepended (LFM2.5: {@code "document: "}); a
     * media document embeds its image, audio or video after the same framing.
     */
    @Override
    public float[] embed(Document document) {
        return document.isText()
                ? embedOne(loaded.documentPrefix() + document.getText())
                : embedMedia(document, null);
    }

    /**
     * Ingestion side, the batching route stores actually use: document framing per document, the
     * strategy's batching preserved for text; a media document is one forward pass of its own.
     */
    @Override
    public List<float[]> embed(
            List<Document> documents, EmbeddingOptions options, BatchingStrategy strategy) {
        float[][] all = new float[documents.size()][];
        List<Document> texts = new ArrayList<>(documents.size());
        for (int i = 0; i < documents.size(); i++) {
            Document document = documents.get(i);
            if (document.isText()) texts.add(document);
            else all[i] = embedMedia(document, options);
        }
        int next = 0; // text vectors fill the remaining slots in document order
        for (List<Document> batch : strategy.batch(texts)) {
            List<String> framed =
                    batch.stream().map(d -> loaded.documentPrefix() + d.getText()).toList();
            for (Embedding e : call(new EmbeddingRequest(framed, options)).getResults()) {
                while (all[next] != null) next++;
                all[next++] = e.getOutput();
            }
        }
        return List.of(all);
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

    /** A media document's vector: the document framing, then its image, audio or video. */
    private float[] embedMedia(Document document, EmbeddingOptions options) {
        String kind = document.getMedia().getMimeType().getType();
        if (!media.contains(kind)) {
            throw new IllegalArgumentException(
                    "document "
                            + document.getId()
                            + " carries "
                            + kind
                            + " media, but "
                            + modelName
                            + (media.isEmpty() ? " embeds text only" : " embeds text and " + media)
                            + ": a multimodal embedder takes media with its projector,"
                            + " companion(\"media\", ...) or"
                            + " spring.ai.jinfer.embedding.companions.media");
        }
        int outputDimension = outputDimension(options);
        List<Object> parts = new ArrayList<>(2);
        if (!loaded.documentPrefix().isEmpty()) parts.add(loaded.documentPrefix());
        parts.add(JinferMappings.media(document.getMedia(), EMBEDDING_FRAMES).value());
        EmbeddingResponse response =
                observed(
                        new EmbeddingRequest(List.of(), options),
                        () -> {
                            float[][] vector = new float[1][];
                            int tokens;
                            lock.lock();
                            try {
                                tokens =
                                        loaded.embed(
                                                state, parts, outputDimension, v -> vector[0] = v);
                            } finally {
                                lock.unlock();
                            }
                            return new EmbeddingResponse(
                                    List.of(new Embedding(vector[0], 0)), metadata(tokens));
                        });
        return response.getResult().getOutput();
    }

    private float[] embedOne(String text) {
        return call(new EmbeddingRequest(List.of(text), EmbeddingOptions.builder().build()))
                .getResults()
                .get(0)
                .getOutput();
    }

    @Override
    public EmbeddingResponse call(EmbeddingRequest request) {
        return observed(request, () -> doCall(request));
    }

    private EmbeddingResponse observed(
            EmbeddingRequest request, Supplier<EmbeddingResponse> embedding) {
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
                            EmbeddingResponse response = embedding.get();
                            observationContext.setResponse(response);
                            return response;
                        });
    }

    private EmbeddingResponse doCall(EmbeddingRequest request) {
        int outputDimension = outputDimension(request.getOptions());
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

    /** The requested width, else the native one; a per-request model other than this is refused. */
    private int outputDimension(EmbeddingOptions options) {
        if (options != null
                && options.getModel() != null
                && !options.getModel().equals(modelName)) {
            throw new IllegalArgumentException(
                    "per-request model is not supported: this model IS '"
                            + modelName
                            + "' (one loaded GGUF per instance)");
        }
        Integer requestedDimension = options == null ? null : options.getDimensions();
        return requestedDimension == null ? loaded.dimension() : requestedDimension;
    }

    private EmbeddingResponseMetadata metadata(int totalTokens) {
        return new EmbeddingResponseMetadata(modelName, new DefaultUsage(totalTokens, 0));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Object source; // Path | model-ref String | LoadedEmbedder: the last setter wins
        private Path modelPath; // derived from source at build()
        private LoadedEmbedder<?> loaded; // derived from source at build()
        private Map<String, Path> companionPaths; // resolved at build()
        private final Map<String, String> companionRefs = new LinkedHashMap<>();
        private final Map<String, Path> localCompanions = new LinkedHashMap<>();
        private int contextCapacity = 2048;
        private ObservationRegistry observationRegistry;
        private EmbeddingModelObservationConvention observationConvention;

        public Builder modelPath(Path modelPath) {
            this.source = modelPath;
            return this;
        }

        /**
         * The model as a model ref, resolved - downloading to the local cache on first use - by
         * {@link #build()}.
         *
         * <pre>{@code
         * model("Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0");
         * }</pre>
         *
         * <p>The full grammar - the default quant, pinned revisions, a file inside a repository,
         * ModelScope - is documented once in {@link com.qxotic.jinfer.hub.ModelRef}. For a file
         * already on disk use {@link #modelPath(Path)}. A URL is not a model ref: download it
         * first, then pass the path.
         */
        public Builder model(String modelRef) {
            ModelStore.requireRef(modelRef);
            this.source = modelRef;
            return this;
        }

        /**
         * A model you loaded yourself. Its weights arena stays yours; close the arena only after
         * this model and every fork on it.
         */
        public Builder model(LoadedEmbedder<?> loaded) {
            this.source = loaded;
            return this;
        }

        /**
         * Attaches a local companion file, such as EmbeddingGemma 2's {@code "media"} projector.
         * This method never touches the network.
         */
        public Builder companionPath(String capability, Path companionPath) {
            Objects.requireNonNull(capability, "capability");
            Objects.requireNonNull(companionPath, "companionPath");
            companionRefs.remove(capability);
            localCompanions.put(capability, companionPath);
            return this;
        }

        /**
         * Attaches a companion from a supported model repository. The reference is resolved at
         * {@link #build()}.
         *
         * <pre>{@code
         * companion("media", "unsloth/embeddinggemma-2-GGUF/mmproj-BF16.gguf");
         * }</pre>
         */
        public Builder companion(String capability, String companionRef) {
            Objects.requireNonNull(capability, "capability");
            if (!ModelStore.isRef(companionRef)) {
                throw new IllegalArgumentException(
                        "'"
                                + companionRef
                                + "' is not a companion model ref. Use companionPath(...) for a"
                                + " local file; download plain URLs first.");
            }
            localCompanions.remove(capability);
            companionRefs.put(capability, companionRef);
            return this;
        }

        /**
         * Upper bound on the packing window and on each embedded sequence, in tokens. The default
         * is 2048. A larger value admits longer sequences and can pack more sequences into one
         * forward pass, at the cost of a larger resident state. {@code 0} uses the model's {@code
         * maxContextLength}; otherwise the effective capacity is the smaller of this value and that
         * maximum.
         *
         * @throws IllegalArgumentException if {@code contextCapacity < 0}
         */
        public Builder contextCapacity(int contextCapacity) {
            if (contextCapacity < 0)
                throw new IllegalArgumentException(
                        "contextCapacity must be >= 0 (0 uses the model maximum): "
                                + contextCapacity);
            this.contextCapacity = contextCapacity;
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
            companionPaths = Map.of();
            if (source == null)
                throw new IllegalArgumentException(
                        "a model is required: model(\"owner/repo:Q4_K_M\"),"
                                + " modelPath(...) or model(LoadedEmbedder)");
            if (source instanceof LoadedEmbedder<?> l) {
                if (!companionRefs.isEmpty() || !localCompanions.isEmpty())
                    throw new IllegalArgumentException(
                            "companions are load-time settings; apply them when you load the"
                                    + " LoadedEmbedder passed to model(...)");
                loaded = l;
                return new JinferEmbeddingModel(this);
            }
            // the model (when it is a string) and the companions resolve in ONE batch, so a cold
            // start pays the slowest download, not the sum
            List<String> wanted = new ArrayList<>();
            if (source instanceof String ref) wanted.add(ref);
            wanted.addAll(companionRefs.values());
            List<Path> resolved = ModelStore.standard().resolveAll(wanted);
            int at = 0;
            modelPath = source instanceof Path path ? path : resolved.get(at++);
            var resolvedCompanions = new LinkedHashMap<>(localCompanions);
            for (String capability : companionRefs.keySet()) {
                resolvedCompanions.put(capability, resolved.get(at++));
            }
            companionPaths = Collections.unmodifiableMap(resolvedCompanions);
            return new JinferEmbeddingModel(this);
        }
    }
}
