package com.qxotic.toknroll.hf;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.fetch.Transport;
import com.qxotic.toknroll.*;
import com.qxotic.toknroll.impl.ImplAccessor;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.text.Normalizer.Form;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.StringJoiner;
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
 *
 * <pre>{@code
 * Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromHuggingFace("openai-community", "gpt2");
 * IntSequence tokens = tokenizer.encode("hello world");
 * String text = tokenizer.decode(tokens);
 * }</pre>
 */
public final class HuggingFaceTokenizerLoader {
    private static final String DEFAULT_HUGGING_FACE_REVISION = "main";
    static final String SOURCE_HUGGING_FACE = "huggingface";
    static final String SOURCE_MODELSCOPE = "modelscope";
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
     * main}).
     *
     * <p>Equivalent to calling {@link #fromHuggingFace(String, String, String, boolean, boolean)}
     * with {@code revision="main"}, {@code useCacheOnly=false}, and {@code forceRefresh=false}.
     *
     * @param user repository owner/namespace on HuggingFace
     * @param repository repository name on HuggingFace
     * @return loaded tokenizer
     */
    public static Tokenizer fromHuggingFace(String user, String repository) {
        return fromHuggingFace(user, repository, DEFAULT_HUGGING_FACE_REVISION, false, false);
    }

    /**
     * Loads a tokenizer from a remote ModelScope repository at the default revision ({@code
     * master}).
     *
     * <p>Equivalent to calling {@link #fromModelScope(String, String, String, boolean, boolean)}
     * with {@code revision=null} (resolved to {@code master}), {@code useCacheOnly=false}, and
     * {@code forceRefresh=false}.
     *
     * @param user repository owner/namespace on ModelScope
     * @param repository repository name on ModelScope
     * @return loaded tokenizer
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
        RepositoryArtifactCache cache =
                RepositoryArtifactCache.create(
                        useCacheOnly ? Transport.offline() : Transport.system());
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
                                forceRefresh);
            } catch (IOException e) {
                if (shouldFallbackToTiktokenModel(e, useCacheOnly)) {
                    return loadFromTiktokenModel(
                            source, cache, user, repository, revision, forceRefresh);
                }
                throw e;
            }
            fetchOptionalTokenizerArtifacts(
                    source, cache, user, repository, revision, forceRefresh);
            return loadUnchecked(tokenizerJson);
        } catch (IOException e) {
            throw new TokenizerLoadException(
                    "["
                            + source
                            + "] Failed to load tokenizer for "
                            + modelRef
                            + (revision == null ? "" : "@" + revision), // as a reference spells it
                    e);
        }
    }

    @SuppressWarnings("unchecked")
    private static Tokenizer loadUnchecked(Path input) throws IOException {
        if (Files.isDirectory(input)
                && !Files.exists(input.resolve(TOKENIZER_JSON))
                && Files.exists(input.resolve(TIKTOKEN_MODEL_FILE))) {
            return loadFromTiktokenFolder(input);
        }
        Path tokenizerJson = resolveTokenizerJsonPath(input);

        Map<String, Object> root = parseObject(tokenizerJson);
        Map<String, Object> model = asObject(root.get("model"), "tokenizer.json:model");
        // Files written before tokenizers tagged its models carry no type: merges mean BPE (gpt2),
        // none means Unigram (t5, a scored vocabulary list) or WordPiece (bert), as Hugging Face
        // reads them: a valid file either way, refused by that name
        String modelType =
                model.get("type") != null
                        ? asString(model.get("type"), "tokenizer.json:model.type")
                        : model.containsKey("merges")
                                ? "BPE"
                                : model.get("vocab") instanceof List
                                        ? "Unigram, untyped"
                                        : "WordPiece, untyped";
        if (!"BPE".equals(modelType)) {
            throw unsupported("tokenizer.json:model.type", modelType, "BPE");
        }

        Map<String, Object> vocabMap = asObject(model.get("vocab"), "tokenizer.json:model.vocab");
        if (vocabMap.isEmpty()) {
            throw new IllegalArgumentException("tokenizer.json:model.vocab is empty");
        }

        Object normalizerObj = root.get("normalizer");
        Normalizer normalizer = parseNormalizer(normalizerObj);
        // Keep raw byte-token surface exactly as exported by tokenizer.json.
        // Collapsing <0x..> aliases can drop merge connectivity for some UTF-8 byte paths.
        TokenEntries entries = buildTokenEntries(vocabMap, root.get("added_tokens"), normalizer);

        Object preTokenizerObj = root.get("pre_tokenizer");
        Map<String, Object> metaspace = metaspacePreTokenizer(preTokenizerObj);
        boolean spacesSpelled = isSentencePieceSpaceNormalizer(normalizerObj);
        boolean sentencePieceStyle = spacesSpelled || metaspace != null;

        if (!sentencePieceStyle) {
            for (int i = 0; i < entries.tokens.length; i++) {
                if (entries.tokens[i] == null) {
                    continue;
                }
                entries.tokens[i] = canonicalizeByteLevelSurface(entries.tokens[i]);
            }
        }

        boolean ignoreMerges = Boolean.TRUE.equals(model.get("ignore_merges"));
        Object dropout = model.get("dropout"); // HF tokenizers 1.0 refuses any; 0 changes nothing
        if (dropout instanceof Number && ((Number) dropout).doubleValue() != 0) {
            throw unsupported("tokenizer.json:model.dropout", dropout.toString(), "null or 0");
        }
        Vocabulary vocabulary = ImplAccessor.createVocabulary(entries.tokens, entries.tokenTypes);
        TokenizationModel tokenizationModel;
        Splitter splitter;
        if (sentencePieceStyle) {
            // ponytail: no SentencePiece tokenizer.json sets it; HF's whole-word lookup would need
            // the text uncut, refused rather than encoded wrong
            if (ignoreMerges) {
                throw unsupported(
                        "tokenizer.json:model.ignore_merges", "true", "false for SentencePiece");
            }
            // without it HF emits the unknown token where the bytes would be spelled: refused,
            // never encoded as the other tokenizer
            if (vocabMap.containsKey("<0x00>")
                    && !Boolean.TRUE.equals(model.get("byte_fallback"))) {
                throw unsupported(
                        "tokenizer.json:model.byte_fallback",
                        String.valueOf(model.get("byte_fallback")),
                        "true for a vocabulary with byte tokens");
            }
            refuseOtherSentencePieceStages(preTokenizerObj, spacesSpelled);
            long[][] packed = packMerges(mergeIds(vocabulary, model.get("merges")));
            tokenizationModel =
                    ImplAccessor.createSentencePieceBpeModel(vocabulary, packed[0], packed[1]);
            Normalizer base = normalizer == null ? Normalizer.identity() : normalizer;
            normalizer = Normalizer.sequence(base, text -> text.toString().replace(' ', METASPACE));
            // HF merges the text whole, or with Metaspace's split at every space; whole, words
            // merge alike where no token joins them, far cheaper
            boolean splits = metaspace != null && !Boolean.FALSE.equals(metaspace.get("split"));
            splitter =
                    ImplAccessor.createSentencePieceSplitter(
                            splits ? new String[0] : entries.tokens, false);
        } else {
            splitter = parsePreTokenizer(preTokenizerObj);
            requireByteLevel(preTokenizerObj, root.get("decoder"));
            List<MergeRule> merges = mergeRules(mergeIds(vocabulary, model.get("merges")));
            tokenizationModel = ImplAccessor.createTiktokenModel(vocabulary, merges, ignoreMerges);
        }

        if (sentencePieceStyle) {
            // the wrapper normalizes, then prepends: HF's order, so a space the normalizer makes
            // (NFKC of U+3000) gets no second one
            return ImplAccessor.metaspaceTokenizer(
                    Toknroll.pipeline(splitter, tokenizationModel),
                    normalizer,
                    prependScheme(metaspace),
                    stripsLeadingSpace(root.get("decoder")));
        }
        return Toknroll.pipeline(normalizer, splitter, tokenizationModel);
    }

    /**
     * A repository that ships {@code tiktoken.model} instead of {@code tokenizer.json}, from {@code
     * source}, {@link #SOURCE_HUGGING_FACE} or {@link #SOURCE_MODELSCOPE}.
     */
    static Tokenizer loadFromTiktokenModel(
            String source,
            RepositoryArtifactCache cache,
            String user,
            String repository,
            String revision,
            boolean forceRefresh)
            throws IOException {
        // The cache lands a repository's files in one folder: fetch what the folder form reads
        // (the auto_map module's name is only known from the config), then read it as a folder.
        Path tiktokenModel =
                fetchFromSource(
                        source,
                        cache,
                        user,
                        repository,
                        revision,
                        TIKTOKEN_MODEL_FILE,
                        forceRefresh);
        Map<String, Object> tokenizerConfig =
                loadOptionalTokenizerConfig(
                        source, cache, user, repository, revision, forceRefresh);
        String module = autoMapModule(tokenizerConfig);
        if (module != null) {
            try {
                fetchFromSource(source, cache, user, repository, revision, module, forceRefresh);
            } catch (IOException e) {
                if (!isHttp404(e)) throw e;
            }
        }
        return loadFromTiktokenFolder(tiktokenModel.getParent());
    }

    /** A folder holding {@code tiktoken.model}: a hub checkout, or the artifact cache's copy. */
    private static Tokenizer loadFromTiktokenFolder(Path dir) throws IOException {
        Path tiktokenModel = dir.resolve(TIKTOKEN_MODEL_FILE);
        Path config = dir.resolve(TOKENIZER_CONFIG_JSON);
        Map<String, Object> tokenizerConfig =
                Files.isRegularFile(config) ? parseObject(config) : null;

        Map<String, Integer> mergeableRanks;
        try (BufferedReader reader =
                Files.newBufferedReader(tiktokenModel, StandardCharsets.UTF_8)) {
            mergeableRanks = ImplAccessor.loadTiktokenMergeableRanks(reader);
        }

        Map<String, Integer> addedTokens = new LinkedHashMap<>();
        Set<String> userDefined = new HashSet<>();
        readAddedTokensDecoder(tokenizerConfig, addedTokens, userDefined);
        Vocabulary vocabulary =
                ImplAccessor.reconstructTiktokenVocabulary(
                        mergeableRanks, addedTokens, userDefined);
        List<MergeRule> merges = ImplAccessor.reconstructTiktokenMergeRules(mergeableRanks);

        TokenizationModel model = Toknroll.tiktokenModel(vocabulary, merges);
        String patStr = resolvePatStr(dir, tokenizerConfig);
        return Toknroll.pipeline(
                Splitter.regex(Pattern.compile(patStr, UNICODE_REGEX_FLAGS)), model);
    }

    private static void fetchOptionalTokenizerArtifacts(
            String source,
            RepositoryArtifactCache cache,
            String user,
            String repository,
            String revision,
            boolean forceRefresh)
            throws IOException {
        for (String file : OPTIONAL_TOKENIZER_FILES) {
            try {
                fetchFromSource(source, cache, user, repository, revision, file, forceRefresh);
            } catch (IOException e) {
                if (!isHttp404(e)) {
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
                            forceRefresh);
            return parseObject(tokenizerConfigPath);
        } catch (IOException e) {
            if (isHttp404(e)) {
                return null;
            }
            throw e;
        }
    }

    /**
     * tokenizer_config.json's {@code added_tokens_decoder}, the added tokens of a tiktoken-format
     * model: each one's text to its id, and the texts of those not marked special, which {@link
     * com.qxotic.toknroll.Specials} matches whole all the same (Kimi's tool-call markers).
     */
    private static void readAddedTokensDecoder(
            Map<String, Object> tokenizerConfig,
            Map<String, Integer> addedTokens,
            Set<String> userDefined) {
        Object decoder =
                tokenizerConfig == null ? null : tokenizerConfig.get("added_tokens_decoder");
        if (!(decoder instanceof Map<?, ?>)) return;
        for (Map.Entry<?, ?> entry : ((Map<?, ?>) decoder).entrySet()) {
            if (!(entry.getKey() instanceof String) || !(entry.getValue() instanceof Map<?, ?>)) {
                continue;
            }
            int id;
            try {
                id = Integer.parseInt((String) entry.getKey());
            } catch (NumberFormatException ignored) {
                continue;
            }
            Map<?, ?> token = (Map<?, ?>) entry.getValue();
            if (!(token.get("content") instanceof String)) continue;
            String content = (String) token.get("content");
            refuseUnsupportedMatching(
                    token, content, TOKENIZER_CONFIG_JSON + ":added_tokens_decoder", null);
            String text = canonicalizeByteLevelSurface(content);
            addedTokens.put(text, id);
            if (!Boolean.TRUE.equals(token.get("special"))) userDefined.add(text);
        }
    }

    /**
     * Refuses an added token Hugging Face would match other than as its exact text: {@code lstrip}
     * and {@code rstrip} also take the whitespace beside it, {@code single_word} matches it only
     * between word boundaries, and {@code normalized} matches it in the normalized text, which
     * differs where the normalizer changes the token. {@link com.qxotic.toknroll.Specials} matches
     * the raw text, and a tokenizer that silently disagreed with its reference is worse than one
     * that does not load.
     */
    private static void refuseUnsupportedMatching(
            Map<?, ?> token, String content, String where, Normalizer normalizer) {
        for (String flag : new String[] {"lstrip", "rstrip", "single_word"}) {
            if (Boolean.TRUE.equals(token.get(flag))) {
                throw unsupported(where + "." + flag, "true, for " + content, "false");
            }
        }
        if (Boolean.TRUE.equals(token.get("normalized"))
                && normalizer != null
                && !content.equals(normalizer.apply(content).toString())) {
            throw unsupported(
                    where + ".normalized",
                    content,
                    "false, or a token the normalizer leaves unchanged");
        }
    }

    /**
     * The GPT-2 pre-tokenization pattern that a ByteLevel pre-tokenizer applies.
     *
     * <p>Kept byte-identical to {@code GGUFTokenizerDefaults.GPT2_PATTERN}. It is duplicated rather
     * than shared because {@code toknroll-hf} does not depend on {@code toknroll-gguf}; if those
     * modules ever gain a common home, this belongs there.
     */
    private static final String BYTE_LEVEL_SPLIT_PATTERN =
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    /**
     * Builds the splitter for a {@code ByteLevel} pre-tokenizer: the GPT-2 pattern, as Hugging
     * Face's {@code use_regex} defaults to true; only {@code use_regex: false} leaves the text
     * whole.
     */
    private static Splitter byteLevelSplitter(Map<String, Object> preTokenizer) {
        if (Boolean.TRUE.equals(preTokenizer.get("add_prefix_space"))) { // never prepended here
            throw unsupported("tokenizer.json:pre_tokenizer.add_prefix_space", "true", "false");
        }
        Object useRegex = preTokenizer.get("use_regex");
        if (Boolean.FALSE.equals(useRegex)) {
            return Splitter.identity();
        }
        return Splitter.regex(Pattern.compile(BYTE_LEVEL_SPLIT_PATTERN, UNICODE_REGEX_FLAGS));
    }

    private static String canonicalizeByteLevelSurface(String token) {
        if (ByteLevel.isValidEncoding(token)) {
            return token;
        }
        return ByteLevel.encode(token.getBytes(StandardCharsets.UTF_8));
    }

    @SuppressWarnings("unchecked")
    private static String resolvePatStr(Path dir, Map<String, Object> tokenizerConfig)
            throws IOException {
        String directPatStr = parsePatStrFromTokenizerConfig(tokenizerConfig);
        if (directPatStr != null) {
            return directPatStr;
        }
        String modulePatStr = tryResolvePatStrFromAutoMapModule(dir, tokenizerConfig);
        if (modulePatStr != null) {
            return modulePatStr;
        }
        throw new IllegalArgumentException(
                "tiktoken.model requires pat_str in "
                        + dir
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
    private static String autoMapModule(Map<String, Object> tokenizerConfig) {
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
        return dot <= 0 ? null : tokenizerClassRef.substring(0, dot) + ".py";
    }

    private static String tryResolvePatStrFromAutoMapModule(
            Path dir, Map<String, Object> tokenizerConfig) throws IOException {
        String module = autoMapModule(tokenizerConfig);
        if (module == null || !Files.isRegularFile(dir.resolve(module))) {
            return null;
        }
        return parsePatStrFromPythonModule(
                Files.readString(dir.resolve(module), StandardCharsets.UTF_8));
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
            boolean forceRefresh)
            throws IOException {
        if (SOURCE_HUGGING_FACE.equals(source)) {
            return cache.fetchHuggingFace(user, repository, revision, file, forceRefresh);
        }
        if (SOURCE_MODELSCOPE.equals(source)) {
            return cache.fetchModelScope(user, repository, revision, file, forceRefresh);
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

    /** The largest id a tokenizer may use: 2^24, sixty times the largest vocabulary there is. */
    private static final int MAX_ID =
            (1 << 24) - 1; // ponytail: a fixed bound, raise it if one grows

    /**
     * An id as the file writes it: a whole number from 0 to MAX_ID, else refused, naming where. A
     * cast would truncate 257.9 to 257 and wrap 2^32 to 0; a huge id would size the vocabulary.
     */
    private static int id(Object value, String where) {
        if (!(value instanceof Number))
            throw new IllegalArgumentException(where + " must be numeric");
        double id = ((Number) value).doubleValue();
        if (id >= 0 && id <= MAX_ID && id == Math.rint(id)) return (int) id;
        throw new IllegalArgumentException(
                where + " is " + value + ", not an id: a whole number from 0 to " + MAX_ID);
    }

    @SuppressWarnings("unchecked")
    private static TokenEntries buildTokenEntries(
            Map<String, Object> vocabMap, Object addedTokensObj, Normalizer normalizer) {
        int maxId = -1;
        for (Map.Entry<String, Object> e : vocabMap.entrySet()) {
            String where = "tokenizer.json:model.vocab['" + e.getKey() + "']";
            maxId = Math.max(maxId, id(e.getValue(), where));
        }

        String[] tokens = new String[maxId + 1];
        for (Map.Entry<String, Object> e : vocabMap.entrySet()) {
            tokens[((Number) e.getValue()).intValue()] = e.getKey();
        }

        int[] tokenTypes = new int[tokens.length];
        Arrays.fill(tokenTypes, StandardTokenType.NORMAL.getId());

        // absent, as Hugging Face allows; otherwise a list of objects, each an id and its content
        List<?> addedTokens = List.of();
        if (addedTokensObj != null) {
            if (!(addedTokensObj instanceof List<?>)) {
                throw new IllegalArgumentException("tokenizer.json:added_tokens must be a list");
            }
            addedTokens = (List<?>) addedTokensObj;
        }
        Map<String, Integer> spelled = new HashMap<>(); // an added token's id, by its content
        for (int i = 0; i < addedTokens.size(); i++) {
            String where = "tokenizer.json:added_tokens[" + i + "]";
            Map<String, Object> added = asObject(addedTokens.get(i), where);
            int id = id(added.get("id"), where + ".id");
            String content = asString(added.get("content"), where + ".content");
            refuseUnsupportedMatching(added, content, "tokenizer.json:added_tokens", normalizer);
            if (id >= tokens.length) {
                int oldLength = tokens.length;
                tokens = Arrays.copyOf(tokens, id + 1);
                tokenTypes = Arrays.copyOf(tokenTypes, id + 1);
                Arrays.fill(
                        tokenTypes, oldLength, tokenTypes.length, StandardTokenType.NORMAL.getId());
            }
            // one id, one spelling, as the vocabulary and every other added token have them
            // (Hugging Face renumbers or drops a token that disagrees); a repeat agrees
            if (tokens[id] != null && !tokens[id].equals(content)) {
                throw new IllegalArgumentException(
                        where
                                + " gives id "
                                + id
                                + " to '"
                                + content
                                + "', which is '"
                                + tokens[id]
                                + "'");
            }
            Integer other = spelled.putIfAbsent(content, id);
            if (other == null && vocabMap.get(content) instanceof Number) {
                other = ((Number) vocabMap.get(content)).intValue();
            }
            if (other != null && other != id) {
                throw new IllegalArgumentException(
                        where + " gives '" + content + "' id " + id + ", which is id " + other);
            }
            boolean special = Boolean.TRUE.equals(added.get("special"));
            tokens[id] = content;
            // Specials.of matches every added token whole, as Hugging Face matches it: the
            // flag says only whether decoding may hide it.
            tokenTypes[id] =
                    special
                            ? StandardTokenType.CONTROL.getId()
                            : StandardTokenType.USER_DEFINED.getId();
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
        if (!Files.isRegularFile(input)) {
            throw new IllegalArgumentException(
                    "Expected a directory or a tokenizer.json file, got " + input);
        }
        return input; // any name: the content is the tokenizer.json, the name is the caller's
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

    /** A well-formed field toknroll does not implement; a malformed one is a plain IAE. */
    private static UnsupportedTokenizerException unsupported(
            String path, String actual, String expected) {
        return new UnsupportedTokenizerException(
                "Unsupported tokenizer config at "
                        + path
                        + ": "
                        + actual
                        + " (supported: "
                        + expected
                        + ")");
    }

    private static boolean isHttp404(IOException e) {
        return e instanceof FileNotFoundException;
    }

    private static boolean shouldFallbackToTiktokenModel(IOException e, boolean useCacheOnly) {
        if (isHttp404(e)) {
            return true;
        }
        return useCacheOnly && e instanceof NoSuchFileException;
    }

    private static Normalizer parseNormalizer(Object normalizerObj) {
        if (normalizerObj == null || normalizerObj == Json.NULL) {
            return null;
        }
        if (!(normalizerObj instanceof Map<?, ?>)) {
            throw new IllegalArgumentException("tokenizer.json:normalizer must be an object");
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
        List<?> children = (List<?>) childrenObj;
        List<Normalizer> chain = new ArrayList<>();
        for (int i = 0; i < children.size(); i++) {
            Object child = children.get(i);
            if (child == null || child == Json.NULL) { // Hugging Face refuses it too
                throw new IllegalArgumentException(
                        "tokenizer.json:normalizer.normalizers[" + i + "] is null");
            }
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
            throw unsupported(
                    "tokenizer.json:normalizer.pattern", patternMap.toString(), "a String pattern");
        }
        String needle = (String) literal;
        return text -> text.toString().replace(needle, content);
    }

    /** The Metaspace pre-tokenizer, alone or in a Sequence; else null. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> metaspacePreTokenizer(Object preTokenizerObj) {
        if (!(preTokenizerObj instanceof Map<?, ?>)) {
            return null;
        }
        Map<String, Object> preTokenizer = (Map<String, Object>) preTokenizerObj;
        if ("Metaspace".equals(preTokenizer.get("type"))) {
            return preTokenizer;
        }
        if ("Sequence".equals(preTokenizer.get("type"))
                && preTokenizer.get("pretokenizers") instanceof List<?>) {
            for (Object child : (List<?>) preTokenizer.get("pretokenizers")) {
                Map<String, Object> metaspace = metaspacePreTokenizer(child);
                if (metaspace != null) {
                    return metaspace;
                }
            }
        }
        return null;
    }

    /**
     * Refuses a SentencePiece pre-tokenizer stage the SentencePiece splitter would drop: only
     * Metaspace is taken, and a Split at {@code " "}, which never matches once the normalizer
     * spelled the spaces (Gemma's).
     */
    @SuppressWarnings("unchecked")
    private static void refuseOtherSentencePieceStages(
            Object preTokenizerObj, boolean spacesSpelled) {
        if (preTokenizerObj == null || preTokenizerObj == Json.NULL) {
            return;
        }
        Map<String, Object> preTokenizer =
                asObject(preTokenizerObj, "tokenizer.json:pre_tokenizer");
        String type = String.valueOf(preTokenizer.get("type"));
        if ("Metaspace".equals(type)) {
            Object replacement = preTokenizer.get("replacement");
            if (replacement != null && !String.valueOf(METASPACE).equals(replacement)) {
                throw unsupported(
                        "tokenizer.json:pre_tokenizer.replacement",
                        String.valueOf(replacement),
                        String.valueOf(METASPACE));
            }
            return;
        }
        if ("Sequence".equals(type) && preTokenizer.get("pretokenizers") instanceof List<?>) {
            for (Object child : (List<?>) preTokenizer.get("pretokenizers")) {
                refuseOtherSentencePieceStages(child, spacesSpelled);
            }
            return;
        }
        Object pattern = preTokenizer.get("pattern");
        if ("Split".equals(type)
                && spacesSpelled
                && pattern instanceof Map<?, ?>
                && " ".equals(((Map<?, ?>) pattern).get("String"))
                && !Boolean.TRUE.equals(preTokenizer.get("invert"))) {
            return;
        }
        throw unsupported(
                "tokenizer.json:pre_tokenizer",
                preTokenizer.toString(),
                "Metaspace, or a Split of \" \" after a normalizer that spells spaces, for a"
                        + " SentencePiece model");
    }

    /**
     * Where Metaspace prepends its U+2581: {@code prepend_scheme}, else the older {@code
     * add_prefix_space} ({@code always} when true, Hugging Face's default); {@code never} without a
     * Metaspace, where a normalizer spells the spaces.
     */
    private static String prependScheme(Map<String, Object> metaspace) {
        if (metaspace == null) {
            return "never";
        }
        Object scheme = metaspace.get("prepend_scheme");
        if (scheme instanceof String) {
            return (String) scheme;
        }
        return Boolean.FALSE.equals(metaspace.get("add_prefix_space")) ? "never" : "always";
    }

    /**
     * Whether a SentencePiece model's decoder drops the text's leading space. It must turn U+2581
     * into spaces: a {@code Replace} of it with {@code ' '} or a {@code Metaspace}, which drops the
     * prepended space, with {@code ByteFallback} and {@code Fuse} beside them, alone or in a
     * Sequence. A {@code Strip} of one leading {@code ' '} counts after a {@code Fuse}: before it,
     * Hugging Face strips every token. Any other decoder is refused.
     */
    private static boolean stripsLeadingSpace(Object decoderObj) {
        List<Map<String, Object>> steps = new ArrayList<>();
        flattenDecoder(decoderObj, steps);
        boolean spaced = false;
        boolean fused = false;
        boolean strips = false;
        for (Map<String, Object> step : steps) {
            String type = String.valueOf(step.get("type"));
            switch (type) {
                case "Replace":
                    Object pattern = step.get("pattern");
                    if (!(pattern instanceof Map<?, ?>)
                            || !String.valueOf(METASPACE)
                                    .equals(((Map<?, ?>) pattern).get("String"))
                            || !" ".equals(step.get("content"))) {
                        throw unsupported(
                                "tokenizer.json:decoder", step.toString(), "Replace of ▁ with ' '");
                    }
                    spaced = true;
                    break;
                case "Metaspace":
                    Object replacement = step.get("replacement"); // decoding turns ▁ into spaces
                    if (replacement != null && !String.valueOf(METASPACE).equals(replacement)) {
                        throw unsupported(
                                "tokenizer.json:decoder", step.toString(), "Metaspace of ▁");
                    }
                    spaced = true;
                    strips |= !"never".equals(prependScheme(step));
                    break;
                case "ByteFallback":
                    break;
                case "Fuse":
                    fused = true;
                    break;
                case "Strip":
                    Object start = step.get("start");
                    Object stop = step.get("stop");
                    if (!" ".equals(step.get("content"))
                            || !(start instanceof Number)
                            || ((Number) start).intValue() > 1
                            || !(stop instanceof Number)
                            || ((Number) stop).intValue() != 0
                            || !spaced
                            || !fused) {
                        throw unsupported(
                                "tokenizer.json:decoder",
                                step.toString(),
                                "Strip of one leading ' ' after Fuse");
                    }
                    strips |= ((Number) start).intValue() == 1;
                    break;
                default:
                    throw unsupported(
                            "tokenizer.json:decoder.type",
                            type,
                            "Replace/Metaspace/ByteFallback/Fuse/Strip/Sequence for a SentencePiece"
                                    + " model");
            }
        }
        if (!spaced) {
            throw unsupported(
                    "tokenizer.json:decoder",
                    String.valueOf(decoderObj),
                    "one that turns ▁ into spaces: Replace or Metaspace");
        }
        return strips;
    }

    /**
     * A byte-level model's text becomes bytes and its ids bytes again, as Hugging Face's ByteLevel
     * stages do: a pre-tokenizer without one drops what no token spells, another decoder reads the
     * ids as other text, so either is refused, not ignored. ponytail: a stage not stated at all
     * runs as bytes, where Hugging Face would skip it; no released file leaves one out.
     */
    private static void requireByteLevel(Object preTokenizerObj, Object decoderObj) {
        if (preTokenizerObj != null
                && preTokenizerObj != Json.NULL
                && !byteLevel(preTokenizerObj)) {
            throw unsupported(
                    "tokenizer.json:pre_tokenizer",
                    String.valueOf(((Map<?, ?>) preTokenizerObj).get("type")),
                    "one with a ByteLevel stage for a byte-level model");
        }
        if (decoderObj == null || decoderObj == Json.NULL) return;
        List<Map<String, Object>> steps = new ArrayList<>();
        flattenDecoder(decoderObj, steps);
        if (steps.size() != 1 || !"ByteLevel".equals(steps.get(0).get("type"))) {
            StringJoiner types = new StringJoiner("+").setEmptyValue("an empty Sequence");
            steps.forEach(step -> types.add(String.valueOf(step.get("type"))));
            throw unsupported(
                    "tokenizer.json:decoder", types.toString(), "ByteLevel for a byte-level model");
        }
    }

    /** Whether a pre-tokenizer, or a step of its Sequence, is ByteLevel. */
    private static boolean byteLevel(Object preTokenizer) {
        if (!(preTokenizer instanceof Map<?, ?>)) return false;
        Map<?, ?> stage = (Map<?, ?>) preTokenizer;
        Object steps = stage.get("pretokenizers");
        return "ByteLevel".equals(stage.get("type"))
                || steps instanceof List<?>
                        && ((List<?>) steps)
                                .stream().anyMatch(HuggingFaceTokenizerLoader::byteLevel);
    }

    /** The decoder's steps in order, Sequences flattened. */
    @SuppressWarnings("unchecked")
    private static void flattenDecoder(Object decoderObj, List<Map<String, Object>> steps) {
        if (!(decoderObj instanceof Map<?, ?>)) {
            return;
        }
        Map<String, Object> decoder = (Map<String, Object>) decoderObj;
        if ("Sequence".equals(decoder.get("type"))) {
            Object children = decoder.get("decoders");
            if (!(children instanceof List<?>)) {
                throw new IllegalArgumentException(
                        "tokenizer.json:decoder.decoders must be a list");
            }
            for (Object child : (List<?>) children) {
                flattenDecoder(child, steps);
            }
        } else {
            steps.add(decoder);
        }
    }

    static Splitter parsePreTokenizer(Object preTokenizerObj) {
        if (preTokenizerObj == null || preTokenizerObj == Json.NULL) {
            return null;
        }
        if (!(preTokenizerObj instanceof Map<?, ?>)) {
            throw new IllegalArgumentException("tokenizer.json:pre_tokenizer must be an object");
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
                return byteLevelSplitter(preTokenizer);
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

    /** Laguna's first pre-tokenizer stage, a Split with MergedWithNext. */
    static final String LAGUNA_SPLIT_REGEX = "(?:\\r?\\n)+(?!\\r?\\n)";

    @SuppressWarnings("unchecked")
    private static final List<String> SPLIT_BEHAVIORS =
            List.of("Removed", "Isolated", "MergedWithPrevious", "MergedWithNext");

    private static Splitter parseSplitPreTokenizer(Map<String, Object> preTokenizer) {
        Object patternObj = preTokenizer.get("pattern");
        if (!(patternObj instanceof Map<?, ?>)) {
            throw new IllegalArgumentException(
                    "tokenizer.json:pre_tokenizer.pattern must be an object");
        }
        String behavior =
                asString(
                        preTokenizer.getOrDefault("behavior", "Removed"),
                        "tokenizer.json:pre_tokenizer.behavior");
        if (!SPLIT_BEHAVIORS.contains(behavior)) { // at load, not at the first encode
            throw unsupported(
                    "tokenizer.json:pre_tokenizer.behavior",
                    behavior,
                    String.join("/", SPLIT_BEHAVIORS));
        }
        boolean invert = Boolean.TRUE.equals(preTokenizer.get("invert"));
        Map<String, Object> patternMap = (Map<String, Object>) patternObj;

        Object regex = patternMap.get("Regex");
        if (regex instanceof String) {
            String regexPattern = (String) regex;
            Pattern pattern = Pattern.compile(regexPattern, UNICODE_REGEX_FLAGS);
            // The two spellings of "each match is a piece": Isolated also keeps the text between
            // matches, Removed with invert drops it. Both take the core splitter, and a known
            // pattern its scanner.
            if ("Isolated".equals(behavior) && !invert) {
                return ImplAccessor.createRegexSplitter(pattern, true);
            }
            if ("Removed".equals(behavior) && invert) {
                return ImplAccessor.createRegexSplitter(pattern, false);
            }
            // Laguna's newline stage, each run merged with the text after it: as java.util.regex
            // matches it, it recurses per newline and overflows the stack on a long run. The same
            // pieces as one possessive match each, followed by Qwen2's regex, take the scanner.
            if ("MergedWithNext".equals(behavior)
                    && !invert
                    && LAGUNA_SPLIT_REGEX.equals(regexPattern)) {
                return ImplAccessor.createRegexSplitter(
                        Pattern.compile(ImplAccessor.LAGUNA_NEWLINES, UNICODE_REGEX_FLAGS), true);
            }
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
                "tokenizer.json:pre_tokenizer.pattern must be {\"Regex\": ...} or {\"String\":"
                        + " ...}");
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
                throw new AssertionError(behavior); // SPLIT_BEHAVIORS, checked at load
        }
        return out; // empty when Removed takes it all, as in Hugging Face
    }

    /**
     * Hugging Face's MergedWithPrevious: a delimiter joins the piece before it, unless that piece
     * is a delimiter too or there is none; then it is a piece of its own. Nothing is dropped.
     */
    private static void mergeWithPrevious(List<Span> all, boolean invert, List<Span> out) {
        boolean previousIsDelimiter = false;
        for (Span s : all) {
            boolean delimiter = isSeparator(s, invert);
            if (delimiter && !previousIsDelimiter && !out.isEmpty()) {
                Span previous = out.remove(out.size() - 1);
                out.add(new Span(previous.start(), s.end(), false));
            } else {
                out.add(new Span(s.start(), s.end(), false));
            }
            previousIsDelimiter = delimiter;
        }
    }

    /** MergedWithNext, the mirror image: a delimiter joins the piece after it. */
    private static void mergeWithNext(List<Span> all, boolean invert, List<Span> out) {
        boolean nextIsDelimiter = false;
        for (int i = all.size() - 1; i >= 0; i--) {
            Span s = all.get(i);
            boolean delimiter = isSeparator(s, invert);
            if (delimiter && !nextIsDelimiter && !out.isEmpty()) {
                Span next = out.remove(out.size() - 1);
                out.add(new Span(s.start(), next.end(), false));
            } else {
                out.add(new Span(s.start(), s.end(), false));
            }
            nextIsDelimiter = delimiter;
        }
        Collections.reverse(out);
    }

    private static boolean isSeparator(Span span, boolean invert) {
        return invert ? !span.match() : span.match();
    }

    /**
     * {@code model.merges} as Hugging Face reads it: each a {@code "left right"} string or a {@code
     * ["left", "right"]} pair, joining two vocabulary tokens into a third. Anything else is
     * refused, as Hugging Face refuses it, never skipped. Returns the {left, right, merged} ids of
     * each merge, flat, in rank order.
     */
    private static int[] mergeIds(Vocabulary vocabulary, Object mergesObj) {
        if (!(mergesObj instanceof List<?>)) {
            throw new IllegalArgumentException("tokenizer.json:model.merges must be a list");
        }
        List<?> merges = (List<?>) mergesObj;
        int[] ids = new int[3 * merges.size()];
        int rank = 0;
        for (Object merge : merges) {
            String left = null;
            String right = null;
            if (merge instanceof String) {
                String spec = (String) merge;
                int space = spec.indexOf(' ');
                if (space >= 0 && space == spec.lastIndexOf(' ')) {
                    left = spec.substring(0, space);
                    right = spec.substring(space + 1);
                }
            } else if (merge instanceof List<?>) {
                List<?> pair = (List<?>) merge;
                if (pair.size() == 2
                        && pair.get(0) instanceof String
                        && pair.get(1) instanceof String) {
                    left = (String) pair.get(0);
                    right = (String) pair.get(1);
                }
            }
            if (left == null) {
                throw new IllegalArgumentException(
                        mergeAt(rank)
                                + " is "
                                + (merge instanceof String ? "'" + merge + "'" : merge)
                                + ", not \"left right\" or [\"left\", \"right\"]");
            }
            ids[3 * rank] = mergeToken(vocabulary, left, rank);
            ids[3 * rank + 1] = mergeToken(vocabulary, right, rank);
            ids[3 * rank + 2] = mergeToken(vocabulary, left + right, rank);
            rank++;
        }
        return ids;
    }

    private static int mergeToken(Vocabulary vocabulary, String token, int rank) {
        int id = ImplAccessor.getIdOrNegative(vocabulary, token);
        if (id < 0) {
            throw new IllegalArgumentException(
                    mergeAt(rank) + " names '" + token + "', which is not in the vocabulary");
        }
        return id;
    }

    private static String mergeAt(int rank) {
        return "tokenizer.json:model.merges[" + rank + "]";
    }

    private static List<MergeRule> mergeRules(int[] ids) {
        List<MergeRule> rules = new ArrayList<>(ids.length / 3);
        for (int rank = 0; rank < ids.length / 3; rank++) {
            rules.add(MergeRule.of(ids[3 * rank], ids[3 * rank + 1], rank));
        }
        return rules;
    }

    private static long[][] packMerges(int[] ids) {
        long[] keys = new long[ids.length / 3];
        long[] values = new long[ids.length / 3];
        for (int rank = 0; rank < keys.length; rank++) {
            keys[rank] = ImplAccessor.pairKey(ids[3 * rank], ids[3 * rank + 1]);
            values[rank] = ImplAccessor.packMerge(rank, ids[3 * rank + 2]);
        }
        return new long[][] {keys, values};
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
