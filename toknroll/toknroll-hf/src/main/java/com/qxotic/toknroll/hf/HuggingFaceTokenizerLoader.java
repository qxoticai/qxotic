package com.qxotic.toknroll.hf;

import com.qxotic.format.json.Json;
import com.qxotic.toknroll.ByteLevel;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.TokenizationPipeline;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.TokenizerLoadException;
import com.qxotic.toknroll.Toknroll;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.impl.ImplAccessor;
import com.qxotic.toknroll.impl.IntPair;
import com.qxotic.toknroll.impl.TransformedTokenizer;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer.Form;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Builds a Tok'n'Roll {@link Tokenizer} from HuggingFace tokenizer-format assets.
 *
 * <p>Supported loading modes:
 *
 * <ul>
 *   <li>Local: load from a model directory containing {@code tokenizer.json} or from a direct
 *       {@code tokenizer.json} path.
 *   <li>Remote HuggingFace: fetch tokenizer artifacts from HuggingFace repositories.
 *   <li>Remote ModelScope: fetch tokenizer artifacts from ModelScope repositories.
 * </ul>
 *
 * <p>Remote artifacts are cached on disk via {@link RepositoryArtifactCache}. Cache policy:
 *
 * <ul>
 *   <li>If a requested artifact already exists in cache and {@code forceRefresh=false}, the cached
 *       file is reused.
 *   <li>If {@code forceRefresh=true}, the artifact is re-downloaded and cache is replaced.
 *   <li>If {@code useCacheOnly=true}, network is never used; loading fails on cache miss.
 * </ul>
 */
public final class HuggingFaceTokenizerLoader {
    private static final String DEFAULT_HUGGING_FACE_REVISION = "main";
    private static final String SOURCE_HUGGING_FACE = "huggingface";
    private static final String SOURCE_MODELSCOPE = "modelscope";
    private static final String TOKENIZER_JSON = "tokenizer.json";
    private static final String TIKTOKEN_MODEL_FILE = "tiktoken.model";
    private static final String TOKENIZER_CONFIG_JSON = "tokenizer_config.json";
    private static final String TOKENIZER_CONFIG_PAT_STR = "pat_str";
    private static final int UNICODE_REGEX_FLAGS = Pattern.UNICODE_CHARACTER_CLASS;
    private static final char METASPACE = '\u2581';
    private static final String[] OPTIONAL_TOKENIZER_FILES = {
        TOKENIZER_CONFIG_JSON, "special_tokens_map.json", "added_tokens.json"
    };

    private HuggingFaceTokenizerLoader() {}

    /**
     * Loads a tokenizer from the local filesystem.
     *
     * <p>Accepts either a model directory containing {@code tokenizer.json} or a direct {@code
     * tokenizer.json} path. The loader is strict and fails fast when unsupported tokenizer features
     * are encountered.
     *
     * @param modelDirOrTokenizerJson model directory or direct {@code tokenizer.json} path
     * @return loaded tokenizer
     * @throws IllegalArgumentException if local files are invalid or unsupported
     * @throws TokenizerLoadException if local file I/O fails
     */
    public static Tokenizer fromLocal(Path modelDirOrTokenizerJson) {
        Objects.requireNonNull(modelDirOrTokenizerJson, "modelDirOrTokenizerJson");
        try {
            return loadUnchecked(modelDirOrTokenizerJson.toAbsolutePath().normalize());
        } catch (IOException e) {
            throw new TokenizerLoadException(
                    "[local] Failed to load HuggingFace tokenizer from " + modelDirOrTokenizerJson,
                    e);
        }
    }

    /**
     * Loads a tokenizer from a remote HuggingFace repository at the default revision ({@code
     * main).
     *
     * <p>Equivalent to calling {@link #fromHuggingFace(String, String, String, boolean, boolean)}
     * with {@code revision="main"}, {@code useCacheOnly=false}, and {@code forceRefresh=false}.
     */
    public static Tokenizer fromHuggingFace(String user, String repository) {
        return fromHuggingFace(user, repository, DEFAULT_HUGGING_FACE_REVISION, false, false);
    }

    /**
     * Loads a tokenizer from a remote ModelScope repository at the default revision ({@code
     * master).
     *
     * <p>Equivalent to calling {@link #fromModelScope(String, String, String, boolean, boolean)}
     * with {@code revision=null} (resolved to {@code master}), {@code useCacheOnly=false}, and
     * {@code forceRefresh=false}.
     */
    public static Tokenizer fromModelScope(String user, String repository) {
        return fromModelScope(user, repository, null, false, false);
    }

    /**
     * Loads a tokenizer from a remote HuggingFace repository.
     *
     * <p>The loader first attempts {@code tokenizer.json}. If it is not found (HTTP 404), it falls
     * back to {@code tiktoken.model} reconstruction plus optional metadata from {@code
     * tokenizer_config.json}.
     *
     * @param user repository owner/namespace on HuggingFace
     * @param repository repository name on HuggingFace
     * @param revision git revision, tag, or branch; {@code null} defaults to {@code main}
     * @param useCacheOnly when {@code true}, only cached artifacts are used
     * @param forceRefresh when {@code true}, cached files are re-downloaded
     * @return loaded tokenizer
     * @throws IllegalArgumentException if tokenizer content is invalid or unsupported
     * @throws TokenizerLoadException if remote fetch or file I/O fails
     * @implNote This method uses {@link RepositoryArtifactCache} for artifact caching.
     */
    public static Tokenizer fromHuggingFace(
            String user,
            String repository,
            String revision,
            boolean useCacheOnly,
            boolean forceRefresh) {
        return loadFromRemoteSource(
                SOURCE_HUGGING_FACE, user, repository, revision, useCacheOnly, forceRefresh);
    }

    /**
     * Loads a tokenizer from a remote ModelScope repository.
     *
     * <p>The loader first attempts {@code tokenizer.json}. If it is not found (HTTP 404), it falls
     * back to {@code tiktoken.model} reconstruction plus optional metadata from {@code
     * tokenizer_config.json}.
     *
     * @param user repository owner/namespace on ModelScope
     * @param repository repository name on ModelScope
     * @param revision git revision, tag, or branch; {@code null} defaults to {@code master}
     * @param useCacheOnly when {@code true}, only cached artifacts are used
     * @param forceRefresh when {@code true}, cached files are re-downloaded
     * @return loaded tokenizer
     * @throws IllegalArgumentException if tokenizer content is invalid or unsupported
     * @throws TokenizerLoadException if remote fetch or file I/O fails
     * @implNote This method uses {@link RepositoryArtifactCache} for artifact caching.
     */
    public static Tokenizer fromModelScope(
            String user,
            String repository,
            String revision,
            boolean useCacheOnly,
            boolean forceRefresh) {
        return loadFromRemoteSource(
                SOURCE_MODELSCOPE, user, repository, revision, useCacheOnly, forceRefresh);
    }

    private static Tokenizer loadFromRemoteSource(
            String source,
            String user,
            String repository,
            String revision,
            boolean useCacheOnly,
            boolean forceRefresh) {
        RepositoryArtifactCache cache = RepositoryArtifactCache.create();
        String modelRef = user + "/" + repository;
        try {
            Path tokenizerJson;
            try {
                tokenizerJson =
                        fetchFromSource(
                                source,
                                cache,
                                user,
                                repository,
                                revision,
                                TOKENIZER_JSON,
                                useCacheOnly,
                                forceRefresh);
            } catch (IOException e) {
                if (shouldFallbackToTiktokenModel(e, useCacheOnly)) {
                    return loadFromTiktokenModel(
                            source, cache, user, repository, revision, useCacheOnly, forceRefresh);
                }
                throw e;
            }
            fetchOptionalTokenizerArtifacts(
                    source, cache, user, repository, revision, useCacheOnly, forceRefresh);
            return loadUnchecked(tokenizerJson);
        } catch (IOException e) {
            throw new TokenizerLoadException(
                    "["
                            + source
                            + "] Failed to load tokenizer for "
                            + modelRef
                            + "@"
                            + String.valueOf(revision),
                    e);
        }
    }

    @SuppressWarnings("unchecked")
    private static Tokenizer loadUnchecked(Path input) throws IOException {
        Path tokenizerJson = resolveTokenizerJsonPath(input);

        Map<String, Object> root = parseObject(tokenizerJson);
        Map<String, Object> model = asObject(root.get("model"), "tokenizer.json:model");
        String modelType = asString(model.get("type"), "tokenizer.json:model.type");
        if (!"BPE".equals(modelType)) {
            throw unsupported("tokenizer.json:model.type", modelType, "BPE");
        }

        Map<String, Object> vocabMap = asObject(model.get("vocab"), "tokenizer.json:model.vocab");
        if (vocabMap.isEmpty()) {
            throw new IllegalArgumentException("tokenizer.json:model.vocab is empty");
        }

        TokenEntries entries = buildTokenEntries(vocabMap, root.get("added_tokens"));
        // Keep raw byte-token surface exactly as exported by tokenizer.json.
        // Collapsing <0x..> aliases can drop merge connectivity for some UTF-8 byte paths.

        Object normalizerObj = root.get("normalizer");
        Normalizer normalizer = parseNormalizer(normalizerObj);
        Object preTokenizerObj = root.get("pre_tokenizer");
        boolean hasMetaspace = hasMetaspacePreTokenizer(preTokenizerObj);
        boolean sentencePieceStyle = isSentencePieceSpaceNormalizer(normalizerObj) || hasMetaspace;
        Splitter splitter = parsePreTokenizer(preTokenizerObj);

        if (!sentencePieceStyle) {
            for (int i = 0; i < entries.tokens.length; i++) {
                if (entries.tokens[i] == null) {
                    continue;
                }
                entries.tokens[i] = canonicalizeByteLevelSurface(entries.tokens[i]);
            }
        }

        boolean ignoreMerges = Boolean.TRUE.equals(model.get("ignore_merges"));
        Vocabulary vocabulary = ImplAccessor.createVocabulary(entries.tokens, entries.tokenTypes);
        TokenizationModel tokenizationModel;
        if (sentencePieceStyle) {
            long[][] packed = buildPackedMergesFromJson(vocabulary, model.get("merges"));
            tokenizationModel =
                    ImplAccessor.createSentencePieceBpeModel(vocabulary, packed[0], packed[1]);
            Normalizer base = normalizer == null ? Normalizer.identity() : normalizer;
            normalizer = Normalizer.sequence(base, text -> text.toString().replace(' ', METASPACE));
            splitter = Splitter.identity();
        } else {
            String[] mergeSpecs = extractMerges(model.get("merges"));
            List<Toknroll.MergeRule> merges = buildMerges(vocabulary, mergeSpecs);
            tokenizationModel = ImplAccessor.createTiktokenModel(vocabulary, merges, ignoreMerges);
        }

        Tokenizer tokenizer = new TokenizationPipeline(normalizer, splitter, tokenizationModel);

        if (hasMetaspace) {
            tokenizer = wrapMetaspace(tokenizer);
        }
        if (sentencePieceStyle) {
            tokenizer = wrapSentencePieceDecode(tokenizer);
        }
        return tokenizer;
    }

    /**
     * Internal fallback loader for repositories that ship {@code tiktoken.model} instead of {@code
     * tokenizer.json}.
     */
    @SuppressWarnings("unchecked")
    static Tokenizer loadFromHfTiktokenModel(
            RepositoryArtifactCache cache,
            String user,
            String repository,
            String revision,
            boolean useCacheOnly,
            boolean forceRefresh)
            throws IOException {
        return loadFromTiktokenModel(
                SOURCE_HUGGING_FACE, cache, user, repository, revision, useCacheOnly, forceRefresh);
    }

    /**
     * Internal fallback loader for ModelScope repositories that ship {@code tiktoken.model} instead
     * of {@code tokenizer.json}.
     */
    @SuppressWarnings("unchecked")
    static Tokenizer loadFromModelScopeTiktokenModel(
            RepositoryArtifactCache cache,
            String user,
            String repository,
            String revision,
            boolean useCacheOnly,
            boolean forceRefresh)
            throws IOException {
        return loadFromTiktokenModel(
                SOURCE_MODELSCOPE, cache, user, repository, revision, useCacheOnly, forceRefresh);
    }

    @SuppressWarnings("unchecked")
    private static Tokenizer loadFromTiktokenModel(
            String source,
            RepositoryArtifactCache cache,
            String user,
            String repository,
            String revision,
            boolean useCacheOnly,
            boolean forceRefresh)
            throws IOException {
        Path tiktokenModel =
                fetchFromSource(
                        source,
                        cache,
                        user,
                        repository,
                        revision,
                        TIKTOKEN_MODEL_FILE,
                        useCacheOnly,
                        forceRefresh);

        Map<String, Object> tokenizerConfig =
                loadOptionalTokenizerConfig(
                        source, cache, user, repository, revision, useCacheOnly, forceRefresh);

        Map<String, Integer> mergeableRanks;
        try (BufferedReader reader =
                Files.newBufferedReader(tiktokenModel, StandardCharsets.UTF_8)) {
            mergeableRanks = ImplAccessor.loadTiktokenMergeableRanks(reader);
        }

        Vocabulary vocabulary =
                ImplAccessor.reconstructTiktokenVocabulary(
                        mergeableRanks, extractSpecialTokens(tokenizerConfig));
        List<Toknroll.MergeRule> merges =
                ImplAccessor.reconstructTiktokenMergeRules(mergeableRanks);

        TokenizationModel model = Toknroll.tiktokenModel(vocabulary, merges);
        String patStr =
                resolvePatStr(
                        source,
                        cache,
                        user,
                        repository,
                        revision,
                        tokenizerConfig,
                        useCacheOnly,
                        forceRefresh);
        return Toknroll.pipeline(
                Splitter.regex(Pattern.compile(patStr, UNICODE_REGEX_FLAGS)), model);
    }

    private static void fetchOptionalTokenizerArtifacts(
            String source,
            RepositoryArtifactCache cache,
            String user,
            String repository,
            String revision,
            boolean useCacheOnly,
            boolean forceRefresh)
            throws IOException {
        for (String file : OPTIONAL_TOKENIZER_FILES) {
            try {
                fetchFromSource(
                        source,
                        cache,
                        user,
                        repository,
                        revision,
                        file,
                        useCacheOnly,
                        forceRefresh);
            } catch (IOException e) {
                if (!isNotFoundError(e)) {
                    throw e;
                }
            }
        }
    }

    private static Map<String, Object> loadOptionalTokenizerConfig(
            String source,
            RepositoryArtifactCache cache,
            String user,
            String repository,
            String revision,
            boolean useCacheOnly,
            boolean forceRefresh)
            throws IOException {
        try {
            Path tokenizerConfigPath =
                    fetchFromSource(
                            source,
                            cache,
                            user,
                            repository,
                            revision,
                            TOKENIZER_CONFIG_JSON,
                            useCacheOnly,
                            forceRefresh);
            return parseObject(tokenizerConfigPath);
        } catch (IOException e) {
            if (isNotFoundError(e)) {
                return null;
            }
            throw e;
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Integer> extractSpecialTokens(Map<String, Object> tokenizerConfig) {
        Map<String, Integer> specialTokens = new LinkedHashMap<>();
        if (tokenizerConfig == null) {
            return specialTokens;
        }
        Object addedDecoderObj = tokenizerConfig.get("added_tokens_decoder");
        if (!(addedDecoderObj instanceof Map<?, ?>)) {
            return specialTokens;
        }
        for (Map.Entry<?, ?> entry : ((Map<?, ?>) addedDecoderObj).entrySet()) {
            if (!(entry.getKey() instanceof String) || !(entry.getValue() instanceof Map<?, ?>)) {
                continue;
            }
            int id;
            try {
                id = Integer.parseInt((String) entry.getKey());
            } catch (NumberFormatException ignored) {
                continue;
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> tokenMap = (Map<String, Object>) entry.getValue();
            Object content = tokenMap.get("content");
            if (content instanceof String) {
                specialTokens.put(canonicalizeByteLevelSurface((String) content), id);
            }
        }
        return specialTokens;
    }

    private static String canonicalizeByteLevelSurface(String token) {
        if (ByteLevel.isValidEncoding(token)) {
            return token;
        }
        return ByteLevel.encode(token.getBytes(StandardCharsets.UTF_8));
    }

    @SuppressWarnings("unchecked")
    private static String resolvePatStr(
            String source,
            RepositoryArtifactCache cache,
            String user,
            String repository,
            String revision,
            Map<String, Object> tokenizerConfig,
            boolean useCacheOnly,
            boolean forceRefresh)
            throws IOException {
        String directPatStr = parsePatStrFromTokenizerConfig(tokenizerConfig);
        if (directPatStr != null) {
            return directPatStr;
        }
        String modulePatStr =
                tryResolvePatStrFromAutoMapModule(
                        source,
                        cache,
                        user,
                        repository,
                        revision,
                        tokenizerConfig,
                        useCacheOnly,
                        forceRefresh);
        if (modulePatStr != null) {
            return modulePatStr;
        }
        throw new IllegalArgumentException(
                "["
                        + source
                        + "] tiktoken.model fallback requires pat_str for "
                        + user
                        + "/"
                        + repository
                        + "@"
                        + String.valueOf(revision)
                        + " (not found in tokenizer_config.json.pat_str or auto_map module)");
    }

    private static String parsePatStrFromTokenizerConfig(Map<String, Object> tokenizerConfig) {
        if (tokenizerConfig == null) {
            return null;
        }
        Object patStrObj = tokenizerConfig.get(TOKENIZER_CONFIG_PAT_STR);
        if (patStrObj instanceof String) {
            String patStr = ((String) patStrObj).trim();
            if (!patStr.isEmpty()) {
                return adaptRegexForJava(patStr);
            }
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private static String tryResolvePatStrFromAutoMapModule(
            String source,
            RepositoryArtifactCache cache,
            String user,
            String repository,
            String revision,
            Map<String, Object> tokenizerConfig,
            boolean useCacheOnly,
            boolean forceRefresh)
            throws IOException {
        if (tokenizerConfig == null) {
            return null;
        }
        Object autoMapObj = tokenizerConfig.get("auto_map");
        if (!(autoMapObj instanceof Map<?, ?>)) {
            return null;
        }
        Object autoTokenizerObj = ((Map<?, ?>) autoMapObj).get("AutoTokenizer");
        if (!(autoTokenizerObj instanceof List<?>) || ((List<?>) autoTokenizerObj).isEmpty()) {
            return null;
        }
        Object first = ((List<?>) autoTokenizerObj).get(0);
        if (!(first instanceof String)) {
            return null;
        }
        String tokenizerClassRef = (String) first;
        int dot = tokenizerClassRef.indexOf('.');
        if (dot <= 0) {
            return null;
        }
        String moduleFile = tokenizerClassRef.substring(0, dot) + ".py";
        Path modulePath;
        try {
            modulePath =
                    fetchFromSource(
                            source,
                            cache,
                            user,
                            repository,
                            revision,
                            moduleFile,
                            useCacheOnly,
                            forceRefresh);
        } catch (IOException e) {
            if (isNotFoundError(e)) {
                return null;
            }
            throw e;
        }
        String moduleSource = Files.readString(modulePath, StandardCharsets.UTF_8);
        return parsePatStrFromPythonModule(moduleSource);
    }

    private static String adaptRegexForJava(String pattern) {
        return pattern.replace("\\p{Han}", "\\p{IsHan}");
    }

    private static Path fetchFromSource(
            String source,
            RepositoryArtifactCache cache,
            String user,
            String repository,
            String revision,
            String file,
            boolean useCacheOnly,
            boolean forceRefresh)
            throws IOException {
        if (SOURCE_HUGGING_FACE.equals(source)) {
            return cache.fetchHuggingFace(
                    user, repository, revision, file, useCacheOnly, forceRefresh);
        }
        if (SOURCE_MODELSCOPE.equals(source)) {
            return cache.fetchModelScope(
                    user, repository, revision, file, useCacheOnly, forceRefresh);
        }
        throw new IllegalArgumentException("Unsupported source: " + source);
    }

    /**
     * Extracts and flattens Python {@code pat_str = "|".join([...])} definitions.
     *
     * <p>The extracted pattern is adapted for Java's regex flavor by rewriting {@code \p{Han}} to
     * {@code \p{IsHan}}.
     */
    static String parsePatStrFromPythonModule(String source) {
        int start = source.indexOf("pat_str = \"|\".join([");
        if (start < 0) {
            return null;
        }
        int end = source.indexOf("])", start);
        if (end < 0) {
            return null;
        }
        String block = source.substring(start, end);
        Matcher matcher = Pattern.compile("r\"\"\"(.*?)\"\"\"", Pattern.DOTALL).matcher(block);
        List<String> parts = new ArrayList<>();
        while (matcher.find()) {
            parts.add(matcher.group(1));
        }
        if (parts.isEmpty()) {
            return null;
        }
        String pattern = String.join("|", parts);
        return adaptRegexForJava(pattern);
    }

    @SuppressWarnings("unchecked")
    private static TokenEntries buildTokenEntries(
            Map<String, Object> vocabMap, Object addedTokensObj) {
        int maxId = -1;
        for (Map.Entry<String, Object> e : vocabMap.entrySet()) {
            Object value = e.getValue();
            if (!(value instanceof Number)) {
                throw new IllegalArgumentException(
                        "tokenizer.json:model.vocab['" + e.getKey() + "'] must be numeric");
            }
            maxId = Math.max(maxId, ((Number) value).intValue());
        }

        String[] tokens = new String[maxId + 1];
        for (Map.Entry<String, Object> e : vocabMap.entrySet()) {
            tokens[((Number) e.getValue()).intValue()] = e.getKey();
        }

        int[] tokenTypes = new int[tokens.length];
        Arrays.fill(tokenTypes, StandardTokenType.NORMAL.getId());

        if (addedTokensObj instanceof List<?>) {
            for (Object obj : (List<?>) addedTokensObj) {
                if (!(obj instanceof Map<?, ?>)) {
                    continue;
                }
                Map<String, Object> added = (Map<String, Object>) obj;
                Object idObj = added.get("id");
                Object contentObj = added.get("content");
                if (!(idObj instanceof Number) || !(contentObj instanceof String)) {
                    continue;
                }

                int id = ((Number) idObj).intValue();
                if (id >= tokens.length) {
                    int oldLength = tokens.length;
                    tokens = Arrays.copyOf(tokens, id + 1);
                    tokenTypes = Arrays.copyOf(tokenTypes, id + 1);
                    Arrays.fill(
                            tokenTypes,
                            oldLength,
                            tokenTypes.length,
                            StandardTokenType.NORMAL.getId());
                }
                boolean special = Boolean.TRUE.equals(added.get("special"));
                if (tokens[id] == null || special) {
                    // Keep decode semantics aligned for explicit special/control tokens,
                    // but do not clobber normal vocab entries.
                    tokens[id] = (String) contentObj;
                }
                if (special) {
                    tokenTypes[id] = StandardTokenType.CONTROL.getId();
                }
            }
        }

        for (int i = 0; i < tokens.length; i++) {
            if (isHexByteToken(tokens[i])) {
                tokenTypes[i] = StandardTokenType.BYTE.getId();
            }
        }
        return new TokenEntries(tokens, tokenTypes);
    }

    private static boolean isHexByteToken(String token) {
        if (token == null
                || token.length() != 6
                || !token.startsWith("<0x")
                || !token.endsWith(">")) {
            return false;
        }
        char hi = token.charAt(3);
        char lo = token.charAt(4);
        return Character.digit(hi, 16) >= 0 && Character.digit(lo, 16) >= 0;
    }

    private static boolean isSentencePieceSpaceNormalizer(Object normalizerObj) {
        if (!(normalizerObj instanceof Map<?, ?>)) {
            return false;
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> normalizer = (Map<String, Object>) normalizerObj;
        Object typeObj = normalizer.get("type");
        if (!(typeObj instanceof String) || !"Replace".equals(typeObj)) {
            return false;
        }
        Object contentObj = normalizer.get("content");
        if (!(contentObj instanceof String) || !"▁".equals(contentObj)) {
            return false;
        }
        Object patternObj = normalizer.get("pattern");
        if (!(patternObj instanceof Map<?, ?>)) {
            return false;
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> pattern = (Map<String, Object>) patternObj;
        Object literalObj = pattern.get("String");
        return " ".equals(literalObj);
    }

    private static Path resolveTokenizerJsonPath(Path input) {
        if (Files.isDirectory(input)) {
            Path tokenizerJson = input.resolve("tokenizer.json");
            if (!Files.exists(tokenizerJson)) {
                throw new IllegalArgumentException("Missing tokenizer.json in directory " + input);
            }
            return tokenizerJson;
        }
        if (!Files.exists(input)) {
            throw new IllegalArgumentException("Path does not exist: " + input);
        }
        if (!input.getFileName().toString().equals("tokenizer.json")) {
            throw new IllegalArgumentException(
                    "Expected directory or tokenizer.json path, got " + input);
        }
        return input;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parseObject(Path file) throws IOException {
        String json = Files.readString(file, StandardCharsets.UTF_8);
        Object parsed = Json.parse(json);
        if (!(parsed instanceof Map<?, ?>)) {
            throw new IllegalArgumentException(file + " must contain a JSON object");
        }
        return (Map<String, Object>) parsed;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asObject(Object value, String path) {
        if (!(value instanceof Map<?, ?>)) {
            throw new IllegalArgumentException(path + " must be an object");
        }
        return (Map<String, Object>) value;
    }

    private static String asString(Object value, String path) {
        if (!(value instanceof String)) {
            throw new IllegalArgumentException(path + " must be a string");
        }
        return (String) value;
    }

    private static IllegalArgumentException unsupported(
            String path, String actual, String expected) {
        return new IllegalArgumentException(
                "Unsupported tokenizer config at "
                        + path
                        + ": "
                        + actual
                        + " (supported: "
                        + expected
                        + ")");
    }

    private static boolean isNotFoundError(IOException e) {
        return e.getMessage() != null && e.getMessage().contains("HTTP 404");
    }

    private static boolean shouldFallbackToTiktokenModel(IOException e, boolean useCacheOnly) {
        if (isNotFoundError(e)) {
            return true;
        }
        return useCacheOnly && isArtifactNotCachedError(e);
    }

    private static boolean isArtifactNotCachedError(IOException e) {
        return e.getMessage() != null && e.getMessage().contains("artifact not cached");
    }

    private static Normalizer parseNormalizer(Object normalizerObj) {
        if (normalizerObj == null || normalizerObj == Json.NULL) {
            return null;
        }
        if (!(normalizerObj instanceof Map<?, ?>)) {
            throw new IllegalArgumentException(
                    "Unsupported tokenizer config at tokenizer.json:normalizer");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> normalizer = (Map<String, Object>) normalizerObj;
        String type = asString(normalizer.get("type"), "tokenizer.json:normalizer.type");
        switch (type) {
            case "NFC":
            case "NFD":
            case "NFKC":
            case "NFKD":
                return Normalizer.unicode(Form.valueOf(type));
            case "Lowercase":
                return Normalizer.lowercase();
            case "Replace":
                return parseReplaceNormalizer(normalizer);
            case "Sequence":
                return parseSequenceNormalizer(normalizer);
            default:
                throw unsupported(
                        "tokenizer.json:normalizer.type",
                        type,
                        "NFC/NFD/NFKC/NFKD/Lowercase/Replace/Sequence");
        }
    }

    private static Normalizer parseSequenceNormalizer(Map<String, Object> normalizer) {
        Object childrenObj = normalizer.get("normalizers");
        if (!(childrenObj instanceof List<?>)) {
            throw new IllegalArgumentException(
                    "tokenizer.json:normalizer.normalizers must be a list");
        }
        List<Normalizer> chain = new ArrayList<>();
        for (Object child : (List<?>) childrenObj) {
            chain.add(parseNormalizer(child));
        }
        return chain.isEmpty()
                ? Normalizer.identity()
                : Normalizer.sequence(chain.toArray(new Normalizer[0]));
    }

    @SuppressWarnings("unchecked")
    private static Normalizer parseReplaceNormalizer(Map<String, Object> normalizer) {
        Object patternObj = normalizer.get("pattern");
        String content = asString(normalizer.get("content"), "tokenizer.json:normalizer.content");
        if (!(patternObj instanceof Map<?, ?>)) {
            throw new IllegalArgumentException(
                    "tokenizer.json:normalizer.pattern must be an object");
        }
        Map<String, Object> patternMap = (Map<String, Object>) patternObj;
        Object literal = patternMap.get("String");
        if (!(literal instanceof String)) {
            throw new IllegalArgumentException(
                    "Unsupported tokenizer config at tokenizer.json:normalizer.pattern (only"
                            + " pattern.String is supported)");
        }
        String needle = (String) literal;
        return text -> text.toString().replace(needle, content);
    }

    private static boolean hasMetaspacePreTokenizer(Object preTokenizerObj) {
        if (!(preTokenizerObj instanceof Map<?, ?>)) {
            return false;
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> preTokenizer = (Map<String, Object>) preTokenizerObj;
        String type = (String) preTokenizer.get("type");
        if ("Metaspace".equals(type)) {
            return true;
        }
        if ("Sequence".equals(type)) {
            Object childrenObj = preTokenizer.get("pretokenizers");
            if (childrenObj instanceof List<?>) {
                for (Object childObj : (List<?>) childrenObj) {
                    if (childObj instanceof Map<?, ?>) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> child = (Map<String, Object>) childObj;
                        if ("Metaspace".equals(child.get("type"))) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    private static Splitter parsePreTokenizer(Object preTokenizerObj) {
        if (preTokenizerObj == null || preTokenizerObj == Json.NULL) {
            return null;
        }
        if (!(preTokenizerObj instanceof Map<?, ?>)) {
            throw new IllegalArgumentException(
                    "Unsupported tokenizer config at tokenizer.json:pre_tokenizer");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> preTokenizer = (Map<String, Object>) preTokenizerObj;
        String type = asString(preTokenizer.get("type"), "tokenizer.json:pre_tokenizer.type");
        switch (type) {
            case "Split":
                return parseSplitPreTokenizer(preTokenizer);
            case "Sequence":
                return parseSequencePreTokenizer(preTokenizer);
            case "ByteLevel":
                return Splitter.identity();
            case "Metaspace":
                return Splitter.identity();
            default:
                throw unsupported(
                        "tokenizer.json:pre_tokenizer.type",
                        type,
                        "Split/Sequence/ByteLevel/Metaspace");
        }
    }

    private static Splitter parseSequencePreTokenizer(Map<String, Object> preTokenizer) {
        Object childrenObj = preTokenizer.get("pretokenizers");
        if (!(childrenObj instanceof List<?>)) {
            throw new IllegalArgumentException(
                    "tokenizer.json:pre_tokenizer.pretokenizers must be a list");
        }
        List<Splitter> splitters = new ArrayList<>();
        for (Object childObj : (List<?>) childrenObj) {
            Splitter child = parsePreTokenizer(childObj);
            if (child != null) {
                splitters.add(child);
            }
        }
        // Filter out identity splitters to avoid sequence wrapping overhead.
        // If only one non-identity splitter remains, return it directly.
        List<Splitter> effective = new ArrayList<>(splitters.size());
        for (Splitter s : splitters) {
            if (!isIdentitySplitter(s)) {
                effective.add(s);
            }
        }
        if (effective.isEmpty()) {
            return Splitter.identity();
        }
        if (effective.size() == 1) {
            return effective.get(0);
        }
        return Splitter.sequence(effective.toArray(new Splitter[0]));
    }

    private static boolean isIdentitySplitter(Splitter splitter) {
        return splitter == Splitter.identity();
    }

    @SuppressWarnings("unchecked")
    private static Splitter parseSplitPreTokenizer(Map<String, Object> preTokenizer) {
        Object patternObj = preTokenizer.get("pattern");
        if (!(patternObj instanceof Map<?, ?>)) {
            throw new IllegalArgumentException(
                    "tokenizer.json:pre_tokenizer.pattern must be an object");
        }
        String behavior = (String) preTokenizer.getOrDefault("behavior", "Removed");
        boolean invert = Boolean.TRUE.equals(preTokenizer.get("invert"));
        Map<String, Object> patternMap = (Map<String, Object>) patternObj;

        Object regex = patternMap.get("Regex");
        if (regex instanceof String) {
            String regexPattern = (String) regex;
            Pattern pattern = Pattern.compile(regexPattern, UNICODE_REGEX_FLAGS);
            return (text, startInclusive, endExclusive, consumer) -> {
                CharSequence source = text.subSequence(startInclusive, endExclusive);
                for (Span span : splitByPatternWithBehavior(source, pattern, behavior, invert)) {
                    consumer.accept(
                            text, startInclusive + span.start(), startInclusive + span.end());
                }
            };
        }

        Object literal = patternMap.get("String");
        if (literal instanceof String) {
            String s = (String) literal;
            if (s.isEmpty()) {
                return Splitter.identity();
            }
            return (text, startInclusive, endExclusive, consumer) -> {
                CharSequence sub = text.subSequence(startInclusive, endExclusive);
                String source = sub instanceof String ? (String) sub : sub.toString();
                for (Span span : splitByLiteral(source, s, behavior, invert)) {
                    consumer.accept(
                            text, startInclusive + span.start(), startInclusive + span.end());
                }
            };
        }

        throw new IllegalArgumentException(
                "Unsupported tokenizer config at tokenizer.json:pre_tokenizer.pattern (only Regex"
                        + " and String are supported)");
    }

    private static List<Span> splitByLiteral(
            String text, String literal, String behavior, boolean invert) {
        List<Span> spans = new ArrayList<>();
        int from = 0;
        int index;
        while ((index = text.indexOf(literal, from)) >= 0) {
            spans.add(new Span(index, index + literal.length(), true));
            from = index + literal.length();
        }
        return segmentsFromSpans(text, spans, behavior, invert);
    }

    private static List<Span> splitByPatternWithBehavior(
            CharSequence text, Pattern pattern, String behavior, boolean invert) {
        List<Span> spans = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            spans.add(new Span(matcher.start(), matcher.end(), true));
        }
        return segmentsFromSpans(text, spans, behavior, invert);
    }

    private static List<Span> segmentsFromSpans(
            CharSequence text, List<Span> matchSpans, String behavior, boolean invert) {
        List<Span> out = new ArrayList<>();
        List<Span> all = new ArrayList<>();
        int cursor = 0;
        for (Span m : matchSpans) {
            if (m.start() > cursor) {
                all.add(new Span(cursor, m.start(), false));
            }
            if (m.end() > m.start()) {
                all.add(m);
            }
            cursor = m.end();
        }
        if (cursor < text.length()) {
            all.add(new Span(cursor, text.length(), false));
        }

        if (all.isEmpty()) {
            out.add(new Span(0, text.length(), false));
            return out;
        }

        switch (behavior) {
            case "Isolated":
                for (Span s : all) {
                    if (s.start() < s.end()) {
                        out.add(new Span(s.start(), s.end(), false));
                    }
                }
                break;
            case "Removed":
                for (Span s : all) {
                    if (!isSeparator(s, invert) && s.start() < s.end()) {
                        out.add(new Span(s.start(), s.end(), false));
                    }
                }
                break;
            case "MergedWithPrevious":
                mergeWithPrevious(all, invert, out);
                break;
            case "MergedWithNext":
                mergeWithNext(all, invert, out);
                break;
            default:
                throw unsupported(
                        "tokenizer.json:pre_tokenizer.behavior",
                        behavior,
                        "Removed/Isolated/MergedWithPrevious/MergedWithNext");
        }

        if (out.isEmpty()) {
            out.add(new Span(0, text.length(), false));
        }
        return out;
    }

    private static void mergeWithPrevious(List<Span> all, boolean invert, List<Span> out) {
        int currentStart = -1;
        int currentEnd = -1;
        for (Span s : all) {
            if (isSeparator(s, invert)) {
                if (currentStart >= 0) {
                    currentEnd = s.end();
                }
            } else {
                if (currentStart >= 0) {
                    out.add(new Span(currentStart, currentEnd, false));
                }
                currentStart = s.start();
                currentEnd = s.end();
            }
        }
        if (currentStart >= 0) {
            out.add(new Span(currentStart, currentEnd, false));
        }
    }

    private static void mergeWithNext(List<Span> all, boolean invert, List<Span> out) {
        int pendingStart = -1;
        for (Span s : all) {
            if (isSeparator(s, invert)) {
                if (pendingStart < 0) {
                    pendingStart = s.start();
                }
            } else {
                int start = pendingStart >= 0 ? pendingStart : s.start();
                out.add(new Span(start, s.end(), false));
                pendingStart = -1;
            }
        }
    }

    private static boolean isSeparator(Span span, boolean invert) {
        return invert ? !span.match() : span.match();
    }

    private static String[] extractMerges(Object mergesObj) {
        if (!(mergesObj instanceof List<?>)) {
            return new String[0];
        }
        List<String> merges = new ArrayList<>();
        for (Object item : (List<?>) mergesObj) {
            if (item instanceof String) {
                merges.add((String) item);
                continue;
            }
            if (item instanceof List<?>) {
                List<?> pair = (List<?>) item;
                if (pair.size() == 2
                        && pair.get(0) instanceof String
                        && pair.get(1) instanceof String) {
                    merges.add(pair.get(0) + " " + pair.get(1));
                }
            }
        }
        return merges.toArray(new String[0]);
    }

    private static List<Toknroll.MergeRule> buildMerges(
            Vocabulary vocabulary, String[] mergeSpecs) {
        List<Toknroll.MergeRule> merges = new ArrayList<>();
        for (int rank = 0; rank < mergeSpecs.length; rank++) {
            int space = mergeSpecs[rank].indexOf(' ');
            if (space < 0) {
                continue;
            }
            String left = mergeSpecs[rank].substring(0, space);
            String right = mergeSpecs[rank].substring(space + 1);
            int leftId = ImplAccessor.getIdOrNegative(vocabulary, left);
            int rightId = ImplAccessor.getIdOrNegative(vocabulary, right);
            int mergedId = ImplAccessor.getIdOrNegative(vocabulary, left + right);
            if (leftId >= 0 && rightId >= 0 && mergedId >= 0) {
                merges.add(new Toknroll.MergeRule(leftId, rightId, rank));
            }
        }
        return merges;
    }

    private static long[][] buildPackedMergesFromJson(Vocabulary vocabulary, Object mergesObj) {
        if (!(mergesObj instanceof List<?>)) {
            return new long[][] {new long[0], new long[0]};
        }
        List<?> merges = (List<?>) mergesObj;
        long[] keys = new long[merges.size()];
        long[] values = new long[merges.size()];
        int size = 0;
        int rank = 0;
        for (Object item : merges) {
            String left, right;
            if (item instanceof String) {
                String spec = (String) item;
                int space = spec.indexOf(' ');
                if (space < 0) {
                    continue;
                }
                left = spec.substring(0, space);
                right = spec.substring(space + 1);
            } else if (item instanceof List<?>) {
                List<?> pair = (List<?>) item;
                if (pair.size() != 2
                        || !(pair.get(0) instanceof String)
                        || !(pair.get(1) instanceof String)) {
                    continue;
                }
                left = (String) pair.get(0);
                right = (String) pair.get(1);
            } else {
                continue;
            }
            int leftId = ImplAccessor.getIdOrNegative(vocabulary, left);
            int rightId = ImplAccessor.getIdOrNegative(vocabulary, right);
            if (leftId < 0 || rightId < 0) {
                continue;
            }
            int mergedId = ImplAccessor.getIdOrNegative(vocabulary, left + right);
            if (mergedId < 0) {
                continue;
            }
            keys[size] = IntPair.of(leftId, rightId);
            values[size] = ImplAccessor.packMerge(rank, mergedId);
            size++;
            rank++;
        }
        if (size == 0) {
            return new long[][] {new long[0], new long[0]};
        }
        if (size < merges.size()) {
            keys = Arrays.copyOf(keys, size);
            values = Arrays.copyOf(values, size);
        }
        return new long[][] {keys, values};
    }

    private static Tokenizer wrapMetaspace(Tokenizer base) {
        return new TransformedTokenizer(base) {
            @Override
            public void encodeInto(
                    CharSequence text,
                    int startInclusive,
                    int endExclusive,
                    IntSequence.Builder out) {
                String replaced =
                        applyMetaspace(text, startInclusive, endExclusive, startInclusive == 0);
                base.encodeInto(replaced, 0, replaced.length(), out);
            }

            @Override
            protected String transformDecoded(String decoded, boolean atStartOfText) {
                return TransformedTokenizer.normalizeMetaspaceDecoded(decoded, atStartOfText);
            }

            @Override
            protected boolean trimLeadingSpaceAtStart() {
                return true;
            }

            @Override
            public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
                String replaced =
                        applyMetaspace(text, startInclusive, endExclusive, startInclusive == 0);
                return base.countTokens(replaced, 0, replaced.length());
            }
        };
    }

    private static Tokenizer wrapSentencePieceDecode(Tokenizer base) {
        return new TransformedTokenizer(base) {
            @Override
            protected String transformDecoded(String decoded, boolean atStartOfText) {
                return TransformedTokenizer.normalizeMetaspaceDecoded(decoded, false);
            }

            @Override
            protected boolean trimLeadingSpaceAtStart() {
                return false;
            }
        };
    }

    private static String applyMetaspace(
            CharSequence text, int startInclusive, int endExclusive, boolean prependMarker) {
        String segment = text.subSequence(startInclusive, endExclusive).toString();
        String replaced = segment.replace(' ', METASPACE);
        return prependMarker ? METASPACE + replaced : replaced;
    }

    private static final class Span {
        private final int start;
        private final int end;
        private final boolean match;

        private Span(int start, int end, boolean match) {
            this.start = start;
            this.end = end;
            this.match = match;
        }

        int start() {
            return start;
        }

        int end() {
            return end;
        }

        boolean match() {
            return match;
        }
    }

    private static final class TokenEntries {
        private final String[] tokens;
        private final int[] tokenTypes;

        private TokenEntries(String[] tokens, int[] tokenTypes) {
            this.tokens = tokens;
            this.tokenTypes = tokenTypes;
        }
    }
}
