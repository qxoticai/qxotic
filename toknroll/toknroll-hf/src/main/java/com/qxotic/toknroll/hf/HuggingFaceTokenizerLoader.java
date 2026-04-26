package com.qxotic.toknroll.hf;

import com.qxotic.format.json.Json;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.TokenizationPipeline;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.impl.ImplAccessor;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.ByteBuffer;
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
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Builds a Tok'n'Roll {@link Tokenizer} from local HuggingFace tokenizer files. */
public final class HuggingFaceTokenizerLoader {
    private static final String DEFAULT_REVISION = "main";
    private static final String TOKENIZER_JSON = "tokenizer.json";
    private static final String TIKTOKEN_MODEL_FILE = "tiktoken.model";
    private static final String TOKENIZER_CONFIG_JSON = "tokenizer_config.json";
    private static final String[] OPTIONAL_TOKENIZER_FILES = {
        TOKENIZER_CONFIG_JSON, "special_tokens_map.json", "added_tokens.json"
    };

    private HuggingFaceTokenizerLoader() {}

    /**
     * Loads a tokenizer from a HuggingFace model directory or a direct tokenizer.json path.
     *
     * <p>The loader is strict and fails fast when unsupported tokenizer features are encountered.
     */
    public static Tokenizer load(Path modelDirOrTokenizerJson) {
        Objects.requireNonNull(modelDirOrTokenizerJson, "modelDirOrTokenizerJson");
        try {
            return loadUnchecked(modelDirOrTokenizerJson.toAbsolutePath().normalize(), null);
        } catch (IOException e) {
            throw new IllegalArgumentException(
                    "Failed to load HuggingFace tokenizer from " + modelDirOrTokenizerJson, e);
        }
    }

    public static Tokenizer fromPretrained(String user, String repository) {
        return fromPretrained(user, repository, DEFAULT_REVISION, false, false);
    }

    public static Tokenizer fromPretrained(
            String user,
            String repository,
            String revision,
            boolean offlineOnly,
            boolean forceRefresh) {
        RepositoryArtifactCache cache = RepositoryArtifactCache.create();
        try {
            Path tokenizerJson;
            try {
                tokenizerJson =
                        cache.fetchHuggingFace(
                                user,
                                repository,
                                revision,
                                TOKENIZER_JSON,
                                offlineOnly,
                                forceRefresh);
            } catch (IOException e) {
                if (isNotFoundError(e)) {
                    return loadFromHfTiktokenModel(
                            cache, user, repository, revision, offlineOnly, forceRefresh);
                }
                throw e;
            }
            for (String file : OPTIONAL_TOKENIZER_FILES) {
                try {
                    cache.fetchHuggingFace(
                            user, repository, revision, file, offlineOnly, forceRefresh);
                } catch (IOException e) {
                    if (!isNotFoundError(e)) {
                        throw e;
                    }
                }
            }
            return loadUnchecked(tokenizerJson, user + "/" + repository);
        } catch (IOException e) {
            throw new IllegalArgumentException(
                    "Failed to load HuggingFace tokenizer for " + user + "/" + repository, e);
        }
    }

    @SuppressWarnings("unchecked")
    private static Tokenizer loadUnchecked(Path input, String modelRefHint) throws IOException {
        Path tokenizerJson = resolveTokenizerJsonPath(input);

        Map<String, Object> root = parseObject(tokenizerJson);
        String tokenizerClass = loadTokenizerClass(tokenizerJson.getParent());
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
        boolean sentencePieceStyle = isSentencePieceSpaceNormalizer(normalizerObj);
        Normalizer normalizer = parseNormalizer(normalizerObj);
        Object preTokenizerObj = root.get("pre_tokenizer");
        boolean hasMetaspace = hasMetaspacePreTokenizer(preTokenizerObj);
        Splitter splitter = parsePreTokenizer(preTokenizerObj, tokenizerClass, modelRefHint);

        List<Tokenizers.MergeRule> merges =
                buildMerges(entries.tokens, extractMerges(model.get("merges")));
        boolean ignoreMerges = Boolean.TRUE.equals(model.get("ignore_merges"));
        Vocabulary vocabulary = ImplAccessor.createVocabulary(entries.tokens, entries.tokenTypes);
        TokenizationModel tokenizationModel;
        if (sentencePieceStyle) {
            tokenizationModel = Tokenizers.sentencePieceBpeModel(vocabulary, merges);
            Normalizer base = normalizer == null ? Normalizer.identity() : normalizer;
            normalizer = Normalizer.sequence(base, text -> text.toString().replace(' ', '\u2581'));
            splitter = Splitter.identity();
        } else {
            try {
                tokenizationModel =
                        ImplAccessor.createTiktokenModel(vocabulary, merges, ignoreMerges);
            } catch (IllegalArgumentException e) {
                if (Boolean.getBoolean("toknroll.hf.debug")) {
                    System.err.println(
                            "[hf-loader] tiktokenModel fallback to sentencePieceBpeModel for "
                                    + modelRefHint
                                    + ": "
                                    + e.getMessage());
                }
                tokenizationModel = Tokenizers.sentencePieceBpeModel(vocabulary, merges);
            }
        }

        TokenizationPipeline.Builder builder = TokenizationPipeline.builder(tokenizationModel);
        if (normalizer != null) {
            builder.normalizer(normalizer);
        }
        builder.splitter(splitter == null ? Splitter.identity() : splitter);

        Tokenizer tokenizer = builder.build();

        if (hasMetaspace) {
            tokenizer = wrapMetaspace(tokenizer);
        }
        if (sentencePieceStyle) {
            tokenizer = wrapSentencePieceDecode(tokenizer);
        }
        return tokenizer;
    }

    @SuppressWarnings("unchecked")
    private static Tokenizer loadFromHfTiktokenModel(
            RepositoryArtifactCache cache,
            String user,
            String repository,
            String revision,
            boolean offlineOnly,
            boolean forceRefresh)
            throws IOException {
        Path tiktokenModel =
                cache.fetchHuggingFace(
                        user, repository, revision, TIKTOKEN_MODEL_FILE, offlineOnly, forceRefresh);

        Map<String, Object> tokenizerConfig = null;
        try {
            Path tokenizerConfigPath =
                    cache.fetchHuggingFace(
                            user,
                            repository,
                            revision,
                            TOKENIZER_CONFIG_JSON,
                            offlineOnly,
                            forceRefresh);
            tokenizerConfig = parseObject(tokenizerConfigPath);
        } catch (IOException e) {
            if (!isNotFoundError(e)) {
                throw e;
            }
        }

        Map<String, Integer> mergeableRanks;
        try (BufferedReader reader =
                Files.newBufferedReader(tiktokenModel, StandardCharsets.UTF_8)) {
            mergeableRanks = ImplAccessor.loadTiktokenMergeableRanks(reader);
        }

        Map<String, Integer> specialTokens = new LinkedHashMap<>();
        if (tokenizerConfig != null) {
            Object addedDecoderObj = tokenizerConfig.get("added_tokens_decoder");
            if (addedDecoderObj instanceof Map<?, ?>) {
                for (Map.Entry<?, ?> entry : ((Map<?, ?>) addedDecoderObj).entrySet()) {
                    if (!(entry.getKey() instanceof String)
                            || !(entry.getValue() instanceof Map<?, ?>)) {
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
                        specialTokens.put((String) content, id);
                    }
                }
            }
        }

        Vocabulary vocabulary =
                ImplAccessor.reconstructTiktokenVocabulary(mergeableRanks, specialTokens);
        List<Tokenizers.MergeRule> merges =
                ImplAccessor.reconstructTiktokenMergeRules(mergeableRanks);

        TokenizationPipeline.Builder pipeline =
                TokenizationPipeline.builder(Tokenizers.tiktokenModel(vocabulary, merges));
        String patStr =
                tryResolvePatStr(
                        cache,
                        user,
                        repository,
                        revision,
                        tokenizerConfig,
                        offlineOnly,
                        forceRefresh);
        if (patStr != null && !patStr.isEmpty()) {
            pipeline.splitter(
                    Splitter.regex(Pattern.compile(patStr, Pattern.UNICODE_CHARACTER_CLASS)));
        }
        return pipeline.build();
    }

    @SuppressWarnings("unchecked")
    private static String tryResolvePatStr(
            RepositoryArtifactCache cache,
            String user,
            String repository,
            String revision,
            Map<String, Object> tokenizerConfig,
            boolean offlineOnly,
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
                    cache.fetchHuggingFace(
                            user, repository, revision, moduleFile, offlineOnly, forceRefresh);
        } catch (IOException e) {
            if (isNotFoundError(e)) {
                return null;
            }
            throw e;
        }
        String source = Files.readString(modulePath, StandardCharsets.UTF_8);
        return parsePatStrFromPythonModule(source);
    }

    private static String parsePatStrFromPythonModule(String source) {
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
        return pattern.replace("\\p{Han}", "\\p{IsHan}");
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
        if (!(addedTokensObj instanceof List<?>)) {
            return new TokenEntries(tokens, tokenTypes);
        }

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
                        tokenTypes, oldLength, tokenTypes.length, StandardTokenType.NORMAL.getId());
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

    private static String loadTokenizerClass(Path modelDir) throws IOException {
        if (modelDir == null) {
            return null;
        }
        Path tokenizerConfigPath = modelDir.resolve(TOKENIZER_CONFIG_JSON);
        if (!Files.exists(tokenizerConfigPath)) {
            return null;
        }
        try {
            Map<String, Object> config = parseObject(tokenizerConfigPath);
            Object value = config.get("tokenizer_class");
            return value instanceof String ? (String) value : null;
        } catch (RuntimeException e) {
            return null;
        }
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

    private static Splitter parsePreTokenizer(
            Object preTokenizerObj, String tokenizerClass, String modelRefHint) {
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
                return parseSplitPreTokenizer(preTokenizer, tokenizerClass, modelRefHint);
            case "Sequence":
                return parseSequencePreTokenizer(preTokenizer, tokenizerClass, modelRefHint);
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

    private static Splitter parseSequencePreTokenizer(
            Map<String, Object> preTokenizer, String tokenizerClass, String modelRefHint) {
        Object childrenObj = preTokenizer.get("pretokenizers");
        if (!(childrenObj instanceof List<?>)) {
            throw new IllegalArgumentException(
                    "tokenizer.json:pre_tokenizer.pretokenizers must be a list");
        }
        List<Splitter> splitters = new ArrayList<>();
        for (Object childObj : (List<?>) childrenObj) {
            Splitter child = parsePreTokenizer(childObj, tokenizerClass, modelRefHint);
            if (child != null) {
                splitters.add(child);
            }
        }
        return splitters.isEmpty()
                ? Splitter.identity()
                : Splitter.sequence(splitters.toArray(new Splitter[0]));
    }

    @SuppressWarnings("unchecked")
    private static Splitter parseSplitPreTokenizer(
            Map<String, Object> preTokenizer, String tokenizerClass, String modelRefHint) {
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
            String regexPattern =
                    adaptRegexForTokenizerClass((String) regex, tokenizerClass, modelRefHint);
            Pattern pattern = Pattern.compile(regexPattern, Pattern.UNICODE_CHARACTER_CLASS);
            return (text, startInclusive, endExclusive, consumer) -> {
                String source = text.subSequence(startInclusive, endExclusive).toString();
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
                String source = text.subSequence(startInclusive, endExclusive).toString();
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

    private static String adaptRegexForTokenizerClass(
            String regex, String tokenizerClass, String modelRefHint) {
        if (regex == null || tokenizerClass == null) {
            return regex;
        }
        return regex;
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
            String text, Pattern pattern, String behavior, boolean invert) {
        List<Span> spans = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            spans.add(new Span(matcher.start(), matcher.end(), true));
        }
        return segmentsFromSpans(text, spans, behavior, invert);
    }

    private static List<Span> segmentsFromSpans(
            String text, List<Span> matchSpans, String behavior, boolean invert) {
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

    private static List<Tokenizers.MergeRule> buildMerges(String[] tokens, String[] mergeSpecs) {
        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] != null) {
                tokenToId.put(tokens[i], i);
            }
        }
        List<Tokenizers.MergeRule> merges = new ArrayList<>();
        for (int rank = 0; rank < mergeSpecs.length; rank++) {
            String[] parts = mergeSpecs[rank].split(" ");
            if (parts.length != 2) {
                continue;
            }
            Integer leftId = tokenToId.get(parts[0]);
            Integer rightId = tokenToId.get(parts[1]);
            Integer mergedId = tokenToId.get(parts[0] + parts[1]);
            if (leftId != null && rightId != null && mergedId != null) {
                merges.add(new Tokenizers.MergeRule(leftId, rightId, rank));
            }
        }
        return merges;
    }

    private static Tokenizer wrapMetaspace(Tokenizer base) {
        return new Tokenizer() {
            @Override
            public Vocabulary vocabulary() {
                return base.vocabulary();
            }

            @Override
            public void encodeInto(
                    CharSequence text,
                    int startInclusive,
                    int endExclusive,
                    IntSequence.Builder out) {
                String segment = text.subSequence(startInclusive, endExclusive).toString();
                String replaced = '\u2581' + segment.replace(' ', '\u2581');
                base.encodeInto(replaced, 0, replaced.length(), out);
            }

            @Override
            public String decode(IntSequence tokens) {
                String decoded = base.decode(tokens).replace('\u2581', ' ');
                return decoded.length() > 0 && decoded.charAt(0) == ' '
                        ? decoded.substring(1)
                        : decoded;
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
                byte[] bytes = decodeBytes(tokens.subSequence(tokenStartIndex, length));
                if (bytes.length > out.remaining()) {
                    throw new IllegalArgumentException("Not enough output space");
                }
                out.put(bytes);
                return length - tokenStartIndex;
            }

            @Override
            public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
                String replaced =
                        text.subSequence(startInclusive, endExclusive)
                                .toString()
                                .replace(' ', '\u2581');
                return base.countTokens(replaced, 0, replaced.length());
            }

            @Override
            public int countBytes(IntSequence tokens) {
                return base.countBytes(tokens);
            }

            @Override
            public float expectedTokensPerChar() {
                return base.expectedTokensPerChar();
            }
        };
    }

    private static Tokenizer wrapSentencePieceDecode(Tokenizer base) {
        return wrapDecoded(base, s -> s.replace('▁', ' '));
    }

    private static Tokenizer wrapDecoded(Tokenizer base, Function<String, String> decodeTransform) {
        return new Tokenizer() {
            @Override
            public Vocabulary vocabulary() {
                return base.vocabulary();
            }

            @Override
            public void encodeInto(
                    CharSequence text,
                    int startInclusive,
                    int endExclusive,
                    IntSequence.Builder out) {
                base.encodeInto(text, startInclusive, endExclusive, out);
            }

            @Override
            public String decode(IntSequence tokens) {
                return decodeTransform.apply(base.decode(tokens));
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
                byte[] bytes = decodeBytes(tokens.subSequence(tokenStartIndex, length));
                if (bytes.length > out.remaining()) {
                    throw new IllegalArgumentException("Not enough output space");
                }
                out.put(bytes);
                return length - tokenStartIndex;
            }

            @Override
            public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
                return base.countTokens(text, startInclusive, endExclusive);
            }

            @Override
            public int countBytes(IntSequence tokens) {
                return base.countBytes(tokens);
            }

            @Override
            public float expectedTokensPerChar() {
                return base.expectedTokensPerChar();
            }
        };
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
