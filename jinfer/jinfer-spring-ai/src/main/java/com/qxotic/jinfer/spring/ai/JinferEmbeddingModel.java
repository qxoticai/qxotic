package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.cache.PromptCache;
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
import org.springframework.util.Assert;

/**
 * Spring AI {@link EmbeddingModel} backed by jinfer: in-process CPU embeddings over a local GGUF
 * (Qwen3-Embedding, LFM2.5-Embedding; any embedding port on the classpath loads via the same
 * architecture dispatch as the chat models). {@link #call} packs inputs into ragged batches bounded
 * by the context - one forward pass embeds many sequences under segmented attention - so RAG
 * ingestion of hundreds of chunks costs a handful of prefills, not hundreds.
 *
 * <p>{@link EmbeddingOptions#getDimensions()} is honored only for Matryoshka-trained ports (Qwen3:
 * 32 through native, EmbeddingGemma 2: 128 through 768); truncated vectors are L2-normalized.
 * Fixed-width ports reject it instead of returning an arbitrary prefix. {@link
 * Builder#dimensions(int)} sets the width for every call that names none, so a vector store sized
 * from {@link #dimensions()} gets vectors of that width on every route.
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
    private final int dimensions; // the width a call without dimensions gets
    final String modelName;
    private final ContextState state; // one reusable state; embed() resets it per group
    private final Arena arena;
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
            try {
                this.loaded =
                        b.loaded != null
                                ? b.loaded
                                : Models.loadEmbedder(b.modelPath, arena, b.companionPaths, null);
            } catch (IOException e) {
                throw new UncheckedIOException("failed to load model: " + e.getMessage(), e);
            }
            this.modelName = loaded.name();
            this.dimensions = defaultDimensions(loaded, b.dimensions);
            this.media = media(loaded.model());
            int maxContextLength = loaded.model().configuration().maxContextLength();
            int contextCapacity =
                    PromptCache.Options.contextCapacity(
                            b.contextCapacity,
                            maxContextLength,
                            PromptCache.Options.DEFAULT_CONTEXT_CAPACITY);
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

    /** The builder's width: 0 is the model's own, anything else a width the model produces. */
    private static int defaultDimensions(LoadedEmbedder<?> loaded, int dimensions) {
        if (dimensions == 0) return loaded.dimension();
        if (dimensions >= loaded.minimumDimension() && dimensions <= loaded.dimension()) {
            return dimensions;
        }
        throw new IllegalArgumentException(
                loaded.name()
                        + (loaded.supportsCustomDimensions()
                                ? " embeds at "
                                        + loaded.minimumDimension()
                                        + ".."
                                        + loaded.dimension()
                                        + " dimensions"
                                : " embeds at a fixed " + loaded.dimension() + " dimensions")
                        + ", not "
                        + dimensions);
    }

    /**
     * What the model reads: video is no projector's, so the model itself says whether it takes it.
     */
    private static Set<String> media(Object model) {
        Set<String> kinds = new LinkedHashSet<>();
        if (model instanceof Multimodal multimodal) {
            if (multimodal.supports(Media.Image.class)) kinds.add("image");
            if (multimodal.supports(Media.Video.class)) kinds.add("video");
            if (multimodal.supports(Media.Audio.class)) kinds.add("audio");
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

    /**
     * The width a call without dimensions gets: {@link Builder#dimensions(int)}, else the model's
     * own. Static, never probed with a forward pass.
     */
    @Override
    public int dimensions() {
        return dimensions;
    }

    // ---- the retrieval seam: Spring AI has no input-type parameter, but the INTERFACE types
    // state the intent - stores call embed(Document)/embed(List<Document>,...) to ingest and
    // embed(String) to search. Mapping the model card's framing onto those types makes every
    // VectorStore retrieval-correct on BOTH sides through the single bean Spring wires.
    // call(EmbeddingRequest) stays the raw, framing-free door. ----

    /**
     * Ingestion side: the card's document framing is prepended (LFM2.5: {@code "document: "}); a
     * media document embeds its image, audio or video unframed, as the card says ("Prefixes apply
     * to text only. Pass images, video, and audio without any prefix.").
     */
    @Override
    public float[] embed(Document document) {
        return document.isText()
                ? embedOne(getEmbeddingContent(document))
                : embedMedia(document, null);
    }

    /**
     * What a text document embeds as: the card's document framing, then its text. The default
     * batched path and the one below both embed this, so a store gets the same vector either way.
     */
    @Override
    public String getEmbeddingContent(Document document) {
        return loaded.documentPrefix() + document.getText();
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
            List<String> framed = batch.stream().map(this::getEmbeddingContent).toList();
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
        Assert.notNull(text, "Text must not be null");
        return embedOne(loaded.queryPrefix() + text);
    }

    /** A media document's vector: its image, audio or video alone, with no framing. */
    private float[] embedMedia(Document document, EmbeddingOptions options) {
        String kind = JinferMappings.kind(document.getMedia()); // a PDF is refused as such
        if (!media.contains(kind)) {
            throw new IllegalArgumentException(
                    "document "
                            + document.getId()
                            + " carries "
                            + kind
                            + " media, but "
                            + modelName
                            + (media.isEmpty()
                                    ? " embeds text only: a multimodal embedder takes media with"
                                            + " its projector, companion(\"media\", ...) or"
                                            + " spring.ai.jinfer.embedding.companions.media"
                                    : " embeds text and " + media + " only"));
        }
        int outputDimension = outputDimension(options);
        List<Object> parts =
                List.of(JinferMappings.media(document.getMedia(), EMBEDDING_FRAMES).media());
        EmbeddingResponse response =
                observed(
                        new EmbeddingRequest(List.of(), withDefaults(options)),
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
        EmbeddingOptions options = withDefaults(request.getOptions());
        EmbeddingRequest effective =
                options == request.getOptions()
                        ? request
                        : new EmbeddingRequest(request.getInstructions(), options);
        return observed(effective, () -> doCall(effective));
    }

    /**
     * The caller's options with the builder's width filled in when they name none, so the
     * observation records the width embedded. The model's own width is left unstated, as before.
     */
    private EmbeddingOptions withDefaults(EmbeddingOptions options) {
        if (dimensions == loaded.dimension()
                || (options != null && options.getDimensions() != null)) {
            return options;
        }
        EmbeddingOptions.Builder merged = EmbeddingOptions.builder().dimensions(dimensions);
        if (options != null && options.getModel() != null) merged.model(options.getModel());
        return merged.build();
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

    /**
     * The requested width, else the default one; a per-request model other than this is refused.
     */
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
        return requestedDimension == null ? dimensions : requestedDimension;
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
        private int contextCapacity; // 0: the default
        private int dimensions; // 0: the model's own
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
         * every model built on it.
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
            if (companionRef != null && companionRef.contains("://")) {
                ModelStore.requireRef(companionRef); // refused, naming the ref a link spells
            }
            if (!ModelStore.isRef(companionRef)) {
                throw new IllegalArgumentException(
                        "'"
                                + companionRef
                                + "' is not a companion model ref. Use companionPath(...) for a"
                                + " local file.");
            }
            localCompanions.remove(capability);
            companionRefs.put(capability, companionRef);
            return this;
        }

        /**
         * Upper bound on the packing window and on each embedded sequence, in tokens. {@code 0},
         * the default, is {@link PromptCache.Options#DEFAULT_CONTEXT_CAPACITY} (4096) or the
         * model's {@code maxContextLength} when that is smaller; {@link
         * PromptCache.Options#MAXIMUM_CONTEXT} (-1) is that length; any other value is exact, and
         * one past that length is refused at build. A larger value admits longer sequences and can
         * pack more sequences into one forward pass, at the cost of a larger resident state.
         *
         * @throws IllegalArgumentException if {@code contextCapacity < -1}
         */
        public Builder contextCapacity(int contextCapacity) {
            // a negative refused here, the rest at build, against the model's maximum
            PromptCache.Options.contextCapacity(contextCapacity, Integer.MAX_VALUE, 0);
            this.contextCapacity = contextCapacity;
            return this;
        }

        /**
         * The embedding width for every call that names none: {@link #embed(String)}, {@link
         * #embed(Document)}, the batched route vector stores take, {@link #call}. {@code 0}, the
         * default, is the model's own width; a Matryoshka model takes any width in its range
         * (EmbeddingGemma 2: 128 to 768, Qwen3-Embedding: 32 to its own), and a fixed-width model
         * its own width only, refused at build otherwise. {@link #dimensions()} reports it, so a
         * store that sizes its table from the model holds vectors of this width; {@link
         * EmbeddingOptions#getDimensions()} on a call still wins.
         *
         * @throws IllegalArgumentException if {@code dimensions < 0}
         */
        public Builder dimensions(int dimensions) {
            if (dimensions < 0) {
                throw new IllegalArgumentException(
                        "dimensions must be 0 (the model's own width) or positive, not "
                                + dimensions);
            }
            this.dimensions = dimensions;
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
