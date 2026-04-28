package com.qxotic.toknroll.gguf;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.TokenizationPipeline;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/** Builds Tok'n'Roll tokenizers from GGUF metadata. */
public final class GGUFTokenizerLoader {
    private static final String SOURCE_HUGGING_FACE = "huggingface";
    private static final String SOURCE_MODELSCOPE = "modelscope";
    private static final String MODEL_LLAMA = "llama";
    private static final String PRE_DEFAULT = "default";
    private static final char METASPACE = '\u2581';

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

        public Builder registerModelFactory(String key, Function<GGUF, TokenizationModel> factory) {
            modelFactories.put(
                    GGUFMetadataKeys.normalizeKey(key, "key"),
                    Objects.requireNonNull(factory, "factory"));
            return this;
        }

        public Builder registerNormalizer(String key, Function<GGUF, Normalizer> factory) {
            normalizers.put(
                    GGUFMetadataKeys.normalizeKey(key, "key"),
                    Objects.requireNonNull(factory, "factory"));
            return this;
        }

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

    public static Builder builderEmpty() {
        return new Builder();
    }

    public static Builder builderDefault() {
        Builder builder = builderEmpty();
        GGUFTokenizerDefaults.applyTo(builder);
        return builder;
    }

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

    public Tokenizer fromHuggingFace(String user, String repository, String ggufPath) {
        return fromHuggingFace(user, repository, null, ggufPath, false, false);
    }

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

    public Tokenizer fromModelScope(String user, String repository, String ggufPath) {
        return fromModelScope(user, repository, null, ggufPath, false, false);
    }

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
                            + "' (metadata key "
                            + GGUFMetadataKeys.PRE
                            + " is missing and no fallback is registered)");
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
                            + "' (metadata key "
                            + GGUFMetadataKeys.PRE
                            + " is missing and no fallback is registered)");
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

        Tokenizer tokenizer =
                TokenizationPipeline.builder(model)
                        .normalizer(normalizer)
                        .splitter(splitter)
                        .build();

        // SPM models with metaspace normalization need decode wrapping.
        if (PRE_DEFAULT.equals(preKey) && MODEL_LLAMA.equals(modelKey)) {
            tokenizer = wrapSentencePieceDecode(tokenizer);
        }

        return tokenizer;
    }

    private static Tokenizer wrapSentencePieceDecode(Tokenizer base) {
        return new TransformedTokenizer(base) {
            @Override
            public void encodeInto(
                    CharSequence text,
                    int startInclusive,
                    int endExclusive,
                    IntSequence.Builder out) {
                base.encodeInto(text, startInclusive, endExclusive, out);
            }

            @Override
            protected String transformDecoded(String decoded, boolean atStartOfText) {
                return normalizeMetaspaceDecoded(decoded, atStartOfText);
            }

            @Override
            public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
                return base.countTokens(text, startInclusive, endExclusive);
            }
        };
    }

    private static String normalizeMetaspaceDecoded(String decoded, boolean trimLeadingSpace) {
        String normalized = decoded.replace(METASPACE, ' ');
        if (trimLeadingSpace && normalized.length() > 0 && normalized.charAt(0) == ' ') {
            return normalized.substring(1);
        }
        return normalized;
    }

    /**
     * Wrapper contract:
     *
     * <ul>
     *   <li>decode/decodeBytes/countBytes/decodeBytesInto all use transformed decode semantics.
     *   <li>Metaspace leading-space trim is applied only at sequence start ({@code atStartOfText}).
     * </ul>
     */
    private abstract static class TransformedTokenizer implements Tokenizer {
        protected final Tokenizer base;

        TransformedTokenizer(Tokenizer base) {
            this.base = base;
        }

        protected abstract String transformDecoded(String decoded, boolean atStartOfText);

        @Override
        public Vocabulary vocabulary() {
            return base.vocabulary();
        }

        @Override
        public String decode(IntSequence tokens) {
            return transformDecoded(base.decode(tokens), true);
        }

        @Override
        public byte[] decodeBytes(IntSequence tokens) {
            return decode(tokens).getBytes(StandardCharsets.UTF_8);
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
            int length = tokens.length();
            if (tokenStartIndex < 0 || tokenStartIndex > length) {
                throw new IndexOutOfBoundsException("tokenStartIndex: " + tokenStartIndex);
            }
            if (tokenStartIndex == length) {
                return 0;
            }

            int remaining = out.remaining();
            boolean atStartOfText = tokenStartIndex == 0;

            byte[] firstTokenBytes =
                    transformedBytes(tokens, tokenStartIndex, tokenStartIndex + 1, atStartOfText);
            if (firstTokenBytes.length > remaining) {
                throw new IllegalArgumentException("Not enough output space");
            }

            int lo = tokenStartIndex + 1;
            int hi = length;
            while (lo < hi) {
                int mid = lo + ((hi - lo + 1) >>> 1);
                int size = transformedBytes(tokens, tokenStartIndex, mid, atStartOfText).length;
                if (size <= remaining) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }

            byte[] bytes = transformedBytes(tokens, tokenStartIndex, lo, atStartOfText);
            out.put(bytes);
            return lo - tokenStartIndex;
        }

        private byte[] transformedBytes(
                IntSequence tokens, int startInclusive, int endExclusive, boolean atStartOfText) {
            String decoded = base.decode(tokens.subSequence(startInclusive, endExclusive));
            return transformDecoded(decoded, atStartOfText).getBytes(StandardCharsets.UTF_8);
        }

        @Override
        public int countBytes(IntSequence tokens) {
            return decodeBytes(tokens).length;
        }

        @Override
        public float expectedTokensPerChar() {
            return base.expectedTokensPerChar();
        }
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
