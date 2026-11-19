package com.qxotic.jinfer.langchain4j;

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
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.ContentType;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.model.TokenCountEstimator;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.listener.EmbeddingModelListener;
import dev.langchain4j.model.embedding.request.EmbeddingInput;
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
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * langchain4j {@link EmbeddingModel} backed by jinfer: in-process CPU embeddings over a local GGUF
 * (Qwen3-Embedding, LFM2.5-Embedding; any embedding port on the classpath loads via the same
 * architecture dispatch as the chat models). {@link #embedAll} packs segments into ragged batches
 * bounded by the context - one forward pass embeds many segments under segmented attention - so RAG
 * ingestion of hundreds of chunks costs a handful of prefills, not hundreds.
 *
 * <p>The standard request {@code dimensions} parameter is advertised only for Matryoshka-trained
 * ports (Qwen3: 32 through native, EmbeddingGemma 2: 128 through 768); truncated vectors are
 * L2-normalized. Fixed-width ports reject it instead of returning an arbitrary prefix. {@link
 * Builder#dimensions(int)} sets the width for every call that names none, so an embedding store
 * sized from {@link #dimension()} gets vectors of that width on every route.
 *
 * <p>A multimodal embedder (EmbeddingGemma 2) with its projector attached - {@code
 * companion("media", ...)} - also embeds image, audio and video {@code EmbeddingRequest} inputs,
 * text and media interleaved in one vector.
 *
 * <p>Token counts in the returned usage are exact (the real tokenizer, not an estimate). Run with
 * jinfer's JVM flags: {@code --add-modules jdk.incubator.vector
 * --enable-native-access=ALL-UNNAMED}.
 */
public final class JinferEmbeddingModel implements EmbeddingModel, AutoCloseable {

    private final LoadedEmbedder<?> loaded;
    private final int dimension; // the width a request without dimensions gets
    private final EmbeddingRequestParameters defaults;
    private final ContextState state; // one reusable state; embed() resets it per group
    private final Arena arena;
    private final ReentrantLock lock = new ReentrantLock(true); // single-stream, like ChatEngine
    private static final Logger LOG = LoggerFactory.getLogger(JinferEmbeddingModel.class);
    private final AtomicBoolean hintedBareUse = new AtomicBoolean();
    private final List<EmbeddingModelListener> listeners;
    private final Set<ContentType> contentTypes;

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
            this.dimension = defaultDimension(loaded, b.dimensions);
            // the native width needs no parameter, and a fixed-width model refuses one
            this.defaults =
                    dimension == loaded.dimension()
                            ? EmbeddingRequestParameters.EMPTY
                            : EmbeddingRequestParameters.builder().dimensions(dimension).build();
            int maxContextLength = loaded.model().configuration().maxContextLength();
            int contextCapacity =
                    PromptCache.Options.contextCapacity(
                            b.contextCapacity,
                            maxContextLength,
                            PromptCache.Options.DEFAULT_CONTEXT_CAPACITY);
            this.state = newState(loaded, contextCapacity, arena);
            this.listeners = b.listeners == null ? List.of() : List.copyOf(b.listeners);
            this.contentTypes = contentTypes(loaded.model());
        } catch (RuntimeException | Error e) {
            Arenas.close(arena); // free before failing (best-effort: ofAuto self-manages)
            throw e;
        }
    }

    /** The builder's width: 0 is the model's own, anything else a width the model produces. */
    private static int defaultDimension(LoadedEmbedder<?> loaded, int dimensions) {
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

    /** Text, plus what the model reads: video is no projector's, so the model itself says. */
    private static Set<ContentType> contentTypes(Object model) {
        Set<ContentType> types = EnumSet.of(ContentType.TEXT);
        if (model instanceof Multimodal media) {
            if (media.supports(Media.Image.class)) types.add(ContentType.IMAGE);
            if (media.supports(Media.Video.class)) types.add(ContentType.VIDEO);
            if (media.supports(Media.Audio.class)) types.add(ContentType.AUDIO);
        }
        return Collections.unmodifiableSet(types);
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
     * Token counting over THIS model's tokenizer: exact on text - for sizing splitter chunks
     * against {@code contextCapacity}. The embedder adds its framing tokens per embedded segment
     * (prefixTokens + suffixTokens: one trailing EOS on Qwen3, one leading BOS on LFM2.5-Embedding)
     * on top of the text count.
     */
    public TokenCountEstimator tokenCountEstimator() {
        // no media plan, no sampler: the estimator refuses media messages before any decode
        return new Estimators(loaded.tokenizer(), null, null, new Replies()); // none to echo
    }

    /**
     * The GGUF this instance loaded, the same name every response's metadata carries - the
     * interface default answers the literal string "unknown".
     */
    @Override
    public String modelName() {
        return loaded.name();
    }

    /**
     * The width a request without {@code dimensions} gets: {@link Builder#dimensions(int)}, else
     * the model's own. Static, never probed with a forward pass.
     */
    @Override
    public int dimension() {
        return dimension;
    }

    /** The builder's width when it is not the model's own; no parameters otherwise. */
    @Override
    public EmbeddingRequestParameters defaultRequestParameters() {
        return defaults;
    }

    /** Text; image, audio and video too when a multimodal embedder has its projector attached. */
    @Override
    public Set<ContentType> supportedContentTypes() {
        return contentTypes;
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
     * langchain4j listeners; core's default {@code embed(EmbeddingRequest)} dispatches
     * onRequest/onResponse/onError around the call ({@code EmbeddingModelListenerUtils}) as long as
     * this returns them.
     */
    @Override
    public List<EmbeddingModelListener> listeners() {
        return listeners;
    }

    /**
     * The typed door ({@code EmbeddingRequest.inputType()}): {@code QUERY}/{@code DOCUMENT} apply
     * the model card's retrieval framing (the port's {@code queryPrefix}/{@code documentPrefix})
     * before tokenizing - retrieval-tuned embedders are TRAINED with these prefixes, and both
     * {@code EmbeddingStoreIngestor} and {@code EmbeddingStoreContentRetriever} send the type via
     * their {@code embeddingInputType(...)} builder knob. A typeless request embeds raw text as
     * given. An input carrying media embeds as one vector of its parts in order, unframed whatever
     * the type: the model card frames text alone ("Prefixes apply to text only. Pass images, video,
     * and audio without any prefix."), as sentence-transformers does.
     */
    @Override
    public EmbeddingResponse doEmbed(EmbeddingRequest request) {
        if (request.inputs() == null) throw new IllegalArgumentException("inputs cannot be null");
        int outputDimension = request.dimensions() == null ? dimension : request.dimensions();
        List<EmbeddingInput> inputs = request.inputs();
        String prefix =
                inputs.stream().noneMatch(JinferEmbeddingModel::textOnly)
                        ? "" // no text-only input to frame
                        : switch (request.inputType()) {
                            case QUERY -> loaded.queryPrefix();
                            case DOCUMENT -> loaded.documentPrefix();
                            case null -> hintBareUse();
                        };
        Response<List<Embedding>> response =
                inputs.stream().allMatch(JinferEmbeddingModel::textOnly)
                        ? embedTexts(
                                inputs.stream().map(in -> prefix + in.text()).toList(),
                                outputDimension)
                        : embedParts(inputs, prefix, outputDimension);
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

    private static boolean textOnly(EmbeddingInput input) {
        return input.contents().stream().allMatch(TextContent.class::isInstance);
    }

    /**
     * Text-only inputs use the same joined text and prefix as the all-text path. Inputs carrying
     * media keep their parts in order, unframed, with media decoded before the lock.
     */
    private Response<List<Embedding>> embedParts(
            List<EmbeddingInput> inputs, String prefix, int outputDimension) {
        List<List<Object>> decoded = new ArrayList<>(inputs.size());
        for (EmbeddingInput input : inputs) {
            if (textOnly(input)) {
                decoded.add(List.of(prefix + input.text()));
                continue;
            }
            List<Object> parts = new ArrayList<>();
            for (var content : input.contents()) {
                parts.add(
                        content instanceof TextContent t
                                ? t.text()
                                : Mappings.media(content, EMBEDDING_FRAMES).media());
            }
            decoded.add(parts);
        }
        List<Embedding> out = new ArrayList<>(inputs.size());
        int total = 0;
        lock.lock();
        try {
            for (List<Object> parts : decoded) {
                total +=
                        loaded.embed(
                                state, parts, outputDimension, v -> out.add(Embedding.from(v)));
            }
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
        private Object source; // Path | model-ref String | LoadedEmbedder: the last setter wins
        private Path modelPath; // derived from source at build()
        private LoadedEmbedder<?> loaded; // derived from source at build()
        private Map<String, Path> companionPaths; // resolved at build()
        private final Map<String, String> companionRefs = new LinkedHashMap<>();
        private final Map<String, Path> localCompanions = new LinkedHashMap<>();
        private int contextCapacity; // 0: the default
        private int dimensions; // 0: the model's own
        private List<EmbeddingModelListener> listeners;

        /** langchain4j listeners; dispatched around every {@code embed(EmbeddingRequest)}. */
        public Builder listeners(List<EmbeddingModelListener> listeners) {
            this.listeners = listeners;
            return this;
        }

        /** The GGUF to load. Required unless {@link #model}. */
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
         * #embedAll}, the ingestor, the retriever. {@code 0}, the default, is the model's own
         * width; a Matryoshka model takes any width in its range (EmbeddingGemma 2: 128 to 768,
         * Qwen3-Embedding: 32 to its own), and a fixed-width model its own width only, refused at
         * build otherwise. {@link #dimension()} reports it, so a store sized from the model holds
         * vectors of this width; a request's own {@code dimensions} still wins.
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
