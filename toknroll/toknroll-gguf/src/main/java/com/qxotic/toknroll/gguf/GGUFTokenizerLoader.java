package com.qxotic.toknroll.gguf;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.TokenizationPipeline;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.TokenizerLoadException;
import com.qxotic.toknroll.impl.ImplAccessor;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * Builds Tok'n'Roll tokenizers from GGUF metadata.
 *
 * <p>When loading from Hugging Face or ModelScope, only the GGUF header and metadata key-value
 * pairs are downloaded and cached — tensors and model weights are never fetched.
 */
public final class GGUFTokenizerLoader {
    private static final String SOURCE_HUGGING_FACE = "huggingface";
    private static final String SOURCE_MODELSCOPE = "modelscope";
    private static final String MODEL_LLAMA = "llama";
    private static final String PRE_DEFAULT = "default";

    private static final class Registries {
        private final Map<String, Function<GGUF, TokenizationModel>> modelFactories;
        private final Map<String, Function<GGUF, Normalizer>> normalizers;
        private final Map<String, Function<GGUF, Splitter>> splitters;
        private final Map<String, String> preFallbackByModel;
        private final Map<String, String> normalizerFallbackByModel;

        private Registries(
                Map<String, Function<GGUF, TokenizationModel>> modelFactories,
                Map<String, Function<GGUF, Normalizer>> normalizers,
                Map<String, Function<GGUF, Splitter>> splitters,
                Map<String, String> preFallbackByModel,
                Map<String, String> normalizerFallbackByModel) {
            this.modelFactories = modelFactories;
            this.normalizers = normalizers;
            this.splitters = splitters;
            this.preFallbackByModel = preFallbackByModel;
            this.normalizerFallbackByModel = normalizerFallbackByModel;
        }
    }

    public static final class Builder {
        private final LinkedHashMap<String, Function<GGUF, TokenizationModel>> modelFactories;
        private final LinkedHashMap<String, Function<GGUF, Normalizer>> normalizers;
        private final LinkedHashMap<String, Function<GGUF, Splitter>> splitters;
        private final LinkedHashMap<String, String> preFallbackByModel;
        private final LinkedHashMap<String, String> normalizerFallbackByModel;

        private Builder() {
            this.modelFactories = new LinkedHashMap<>();
            this.normalizers = new LinkedHashMap<>();
            this.splitters = new LinkedHashMap<>();
            this.preFallbackByModel = new LinkedHashMap<>();
            this.normalizerFallbackByModel = new LinkedHashMap<>();
        }

        /**
         * Registers a model factory for the given GGUF {@code tokenizer.ggml.model} key. The
         * factory receives the parsed GGUF metadata and must produce a {@link TokenizationModel}.
         */
        public Builder registerModelFactory(String key, Function<GGUF, TokenizationModel> factory) {
            modelFactories.put(
                    GGUFMetadataKeys.normalizeKey(key, "key"),
                    Objects.requireNonNull(factory, "factory"));
            return this;
        }

        /**
         * Registers a normalizer factory for the given GGUF pre-tokenizer key. The factory receives
         * the parsed GGUF metadata.
         */
        public Builder registerNormalizer(String key, Function<GGUF, Normalizer> factory) {
            normalizers.put(
                    GGUFMetadataKeys.normalizeKey(key, "key"),
                    Objects.requireNonNull(factory, "factory"));
            return this;
        }

        /**
         * Registers a splitter factory for the given GGUF pre-tokenizer key. The factory receives
         * the parsed GGUF metadata.
         */
        public Builder registerPreTokenizer(String key, Function<GGUF, Splitter> factory) {
            splitters.put(
                    GGUFMetadataKeys.normalizeKey(key, "key"),
                    Objects.requireNonNull(factory, "factory"));
            return this;
        }

        Builder registerPreFallback(String modelKey, String preKey) {
            preFallbackByModel.put(
                    GGUFMetadataKeys.normalizeKey(modelKey, "modelKey"),
                    GGUFMetadataKeys.normalizeKey(preKey, "preKey"));
            return this;
        }

        Builder registerNormalizerFallback(String modelKey, String normalizerKey) {
            normalizerFallbackByModel.put(
                    GGUFMetadataKeys.normalizeKey(modelKey, "modelKey"),
                    GGUFMetadataKeys.normalizeKey(normalizerKey, "normalizerKey"));
            return this;
        }

        /** Finishes configuration and returns a ready-to-use loader. */
        public GGUFTokenizerLoader build() {
            Registries registries =
                    new Registries(
                            Map.copyOf(modelFactories),
                            Map.copyOf(normalizers),
                            Map.copyOf(splitters),
                            Map.copyOf(preFallbackByModel),
                            Map.copyOf(normalizerFallbackByModel));
            return new GGUFTokenizerLoader(registries);
        }
    }

    private final Registries registries;

    private GGUFTokenizerLoader(Registries registries) {
        this.registries = registries;
    }

    /** Creates a builder with no pre-registered factories. */
    public static Builder createEmptyBuilder() {
        return new Builder();
    }

    /**
     * Creates a builder pre-loaded with built-in model factories, normalizers, and splitters for
     * common GGUF models.
     */
    public static Builder createBuilderWithBuiltins() {
        Builder builder = createEmptyBuilder();
        GGUFTokenizerDefaults.applyTo(builder);
        return builder;
    }

    /** Builds a tokenizer from a local GGUF file. */
    public Tokenizer fromLocal(Path ggufFile) {
        Objects.requireNonNull(ggufFile, "ggufFile");
        Path file = ggufFile.toAbsolutePath().normalize();
        if (!Files.exists(file)) {
            throw new IllegalArgumentException("Path does not exist: " + file);
        }
        if (Files.isDirectory(file)) {
            throw new IllegalArgumentException("Expected GGUF file path, got directory: " + file);
        }
        if (!file.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".gguf")) {
            throw new IllegalArgumentException("Expected .gguf file path, got: " + file);
        }

        try {
            GGUF gguf = GGUF.read(file);
            return fromParsedGguf(gguf);
        } catch (IOException e) {
            throw new TokenizerLoadException(
                    "[local] Failed to load GGUF tokenizer from " + file, e);
        }
    }

    /**
     * Fetches a GGUF file from Hugging Face and builds a tokenizer. Uses the default branch,
     * downloads if not cached, and does not force-refresh.
     */
    public Tokenizer fromHuggingFace(String user, String repository, String ggufPath) {
        return fromHuggingFace(user, repository, null, ggufPath, false, false);
    }

    /**
     * Fetches a GGUF file from Hugging Face with full parameter control.
     *
     * @param revision branch/tag/commit, or {@code null} for default
     * @param ggufPath path to the GGUF file within the repository
     * @param useCacheOnly if {@code true}, does not fetch over the network
     * @param forceRefresh if {@code true}, ignores cached data and re-fetches
     */
    public Tokenizer fromHuggingFace(
            String user,
            String repository,
            String revision,
            String ggufPath,
            boolean useCacheOnly,
            boolean forceRefresh) {
        return fromRemote(
                SOURCE_HUGGING_FACE,
                user,
                repository,
                revision,
                ggufPath,
                useCacheOnly,
                forceRefresh);
    }

    /**
     * Fetches a GGUF file from ModelScope and builds a tokenizer. Uses the default branch,
     * downloads if not cached, and does not force-refresh.
     */
    public Tokenizer fromModelScope(String user, String repository, String ggufPath) {
        return fromModelScope(user, repository, null, ggufPath, false, false);
    }

    /**
     * Fetches a GGUF file from ModelScope with full parameter control. See {@link
     * #fromHuggingFace(String, String, String, String, boolean, boolean)} for parameter details.
     */
    public Tokenizer fromModelScope(
            String user,
            String repository,
            String revision,
            String ggufPath,
            boolean useCacheOnly,
            boolean forceRefresh) {
        return fromRemote(
                SOURCE_MODELSCOPE,
                user,
                repository,
                revision,
                ggufPath,
                useCacheOnly,
                forceRefresh);
    }

    private Tokenizer fromRemote(
            String source,
            String user,
            String repository,
            String revision,
            String ggufPath,
            boolean useCacheOnly,
            boolean forceRefresh) {
        GGUFMetadataCache cache = GGUFMetadataCache.create();
        try {
            Path metadataPath;
            if (SOURCE_HUGGING_FACE.equals(source)) {
                metadataPath =
                        cache.fetchHuggingFace(
                                user, repository, revision, ggufPath, useCacheOnly, forceRefresh);
            } else if (SOURCE_MODELSCOPE.equals(source)) {
                metadataPath =
                        cache.fetchModelScope(
                                user, repository, revision, ggufPath, useCacheOnly, forceRefresh);
            } else {
                throw new IllegalArgumentException("Unsupported source: " + source);
            }
            GGUF gguf = GGUF.read(metadataPath);
            return fromParsedGguf(gguf);
        } catch (IOException e) {
            throw new TokenizerLoadException(
                    "["
                            + source
                            + "] Failed to load GGUF tokenizer for "
                            + user
                            + "/"
                            + repository
                            + "@"
                            + String.valueOf(revision)
                            + "#"
                            + ggufPath,
                    e);
        }
    }

    private Tokenizer fromParsedGguf(GGUF gguf) {
        String modelKey = GGUFMetadataKeys.requireKey(gguf, GGUFMetadataKeys.MODEL);
        String preKey = resolvePreTokenizerKey(gguf, modelKey);
        String normalizerKey = resolveNormalizerKey(gguf, modelKey);

        Function<GGUF, TokenizationModel> modelFactory = registries.modelFactories.get(modelKey);
        if (modelFactory == null) {
            throw new IllegalArgumentException(
                    "Unsupported GGUF tokenizer model '"
                            + modelKey
                            + "' (supported: "
                            + String.join(", ", registries.modelFactories.keySet())
                            + ")");
        }

        TokenizationModel model = modelFactory.apply(gguf);

        if (preKey == null) {
            throw new IllegalArgumentException(
                    "No pre-tokenizer key resolved for model '"
                            + modelKey
                            + "' (explicit pre-tokenizer key '"
                            + GGUFMetadataKeys.PRE
                            + "' is absent and no pre-tokenizer fallback is registered)");
        }
        Function<GGUF, Splitter> splitterFactory = registries.splitters.get(preKey);
        if (splitterFactory == null) {
            throw new IllegalArgumentException(
                    "Unsupported GGUF pre-tokenizer key '"
                            + preKey
                            + "' for model '"
                            + modelKey
                            + "'. Register it via registerPreTokenizer(...)");
        }
        Splitter splitter = splitterFactory.apply(gguf);

        if (normalizerKey == null) {
            throw new IllegalArgumentException(
                    "No normalizer key resolved for model '"
                            + modelKey
                            + "' (explicit pre-tokenizer key '"
                            + GGUFMetadataKeys.PRE
                            + "' is absent and no normalizer fallback is registered)");
        }
        Function<GGUF, Normalizer> normalizerFactory = registries.normalizers.get(normalizerKey);
        if (normalizerFactory == null) {
            throw new IllegalArgumentException(
                    "Unsupported GGUF normalizer key '"
                            + normalizerKey
                            + "' for model '"
                            + modelKey
                            + "'. Register it via registerNormalizer(...)");
        }
        Normalizer normalizer = normalizerFactory.apply(gguf);

        Tokenizer tokenizer = new TokenizationPipeline(normalizer, splitter, model);

        // SPM models with metaspace normalization need decode wrapping.
        if (PRE_DEFAULT.equals(preKey) && MODEL_LLAMA.equals(modelKey)) {
            tokenizer = wrapSentencePieceDecode(tokenizer);
        }

        return tokenizer;
    }

    private static Tokenizer wrapSentencePieceDecode(Tokenizer base) {
        return ImplAccessor.sentencePieceDecodeWrapper(base, true);
    }

    private String resolvePreTokenizerKey(GGUF gguf, String modelKey) {
        String explicitPre = GGUFMetadataKeys.key(gguf, GGUFMetadataKeys.PRE);
        if (explicitPre != null) {
            return explicitPre;
        }
        return registries.preFallbackByModel.get(modelKey);
    }

    private String resolveNormalizerKey(GGUF gguf, String modelKey) {
        String explicitPre = GGUFMetadataKeys.key(gguf, GGUFMetadataKeys.PRE);
        if (explicitPre != null) {
            return explicitPre;
        }
        return registries.normalizerFallbackByModel.get(modelKey);
    }
}
