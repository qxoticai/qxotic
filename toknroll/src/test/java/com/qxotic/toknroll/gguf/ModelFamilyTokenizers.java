package com.qxotic.toknroll.gguf;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.json.Json;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.advanced.Normalizer;
import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.advanced.StandardTokenType;
import com.qxotic.toknroll.gguf.TestDataManager.TestModel;
import com.qxotic.toknroll.gguf.TestDataManager.TokenizerMetadata;
import com.qxotic.toknroll.impl.GPT2Tokenizer;
import com.qxotic.toknroll.impl.IntPair;
import com.qxotic.toknroll.impl.LongLongMap;
import com.qxotic.toknroll.impl.VocabularyImpl;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer.Form;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/** Utility factory for recreating tokenizer instances by model family id for tests. */
public final class ModelFamilyTokenizers {
    private static final TestDataManager DATA = new TestDataManager();
    private static final Map<String, Tokenizer> CACHE = new ConcurrentHashMap<>();
    private static final Map<String, ModelFamilySpec> MODEL_FAMILY_SPECS = createModelFamilySpecs();
    private static final Map<String, UrlFamilySpec> URL_FAMILY_SPECS = createUrlFamilySpecs();

    private ModelFamilyTokenizers() {}

    public static Optional<Tokenizer> create(String familyId) {
        try {
            return Optional.of(
                    CACHE.computeIfAbsent(familyId, ModelFamilyTokenizers::createUnchecked));
        } catch (RuntimeException e) {
            return Optional.empty();
        }
    }

    public static Optional<Tokenizer> createFromHfFiles(
            String familyId, String modelRef, String revision) {
        try {
            return Optional.of(createFromHfFilesUnchecked(familyId, modelRef, revision));
        } catch (RuntimeException e) {
            return Optional.empty();
        }
    }

    private static Tokenizer createUnchecked(String familyId) {
        try {
            if ("google.gemma3".equals(familyId)) {
                return fromGemma3();
            }
            if ("google.gemma4".equals(familyId)) {
                return fromGemma4();
            }
            if ("alibaba.qwen3_5".equals(familyId)) {
                return fromQwen35();
            }
            if ("deepseek.v3_0324".equals(familyId)) {
                return fromDeepSeekV3();
            }

            ModelFamilySpec modelSpec = MODEL_FAMILY_SPECS.get(familyId);
            if (modelSpec != null) {
                return fromModel(
                        modelSpec.model(),
                        modelSpec.splitterModelType(),
                        modelSpec.normalizer(),
                        modelSpec.specialTokenMode());
            }

            UrlFamilySpec urlSpec = URL_FAMILY_SPECS.get(familyId);
            if (urlSpec != null) {
                return fromUrls(
                        urlSpec.cachePrefix(),
                        urlSpec.splitterModelType(),
                        urlSpec.normalizer(),
                        urlSpec.specialTokenMode(),
                        urlSpec.failureMessage(),
                        urlSpec.urls());
            }

            throw new IllegalArgumentException("Unsupported family: " + familyId);
        } catch (IOException | InterruptedException e) {
            throw new IllegalStateException("Failed to build tokenizer for " + familyId, e);
        }
    }

    private static Tokenizer createFromHfFilesUnchecked(
            String familyId, String modelRef, String revision) {
        try {
            if ("google.gemma3".equals(familyId)) {
                return fromGemma3();
            }
            if ("alibaba.qwen3_5".equals(familyId)) {
                return fromHfQwen35(modelRef, revision);
            }
            if ("deepseek.v3_0324".equals(familyId)) {
                return fromHfDeepSeekV3(modelRef, revision);
            }

            ModelFamilySpec modelSpec = MODEL_FAMILY_SPECS.get(familyId);
            if (modelSpec != null) {
                return fromHfTokenizerJson(
                        modelRef,
                        revision,
                        modelSpec.splitterModelType(),
                        modelSpec.normalizer(),
                        modelSpec.specialTokenMode());
            }

            UrlFamilySpec urlSpec = URL_FAMILY_SPECS.get(familyId);
            if (urlSpec != null) {
                return fromHfTokenizerJson(
                        modelRef,
                        revision,
                        urlSpec.splitterModelType(),
                        urlSpec.normalizer(),
                        urlSpec.specialTokenMode());
            }

            throw new IllegalArgumentException("Unsupported family: " + familyId);
        } catch (IOException | InterruptedException e) {
            throw new IllegalStateException("Failed to build HF tokenizer for " + familyId, e);
        }
    }

    private static Tokenizer fromModel(
            TestModel model,
            String splitterModelType,
            Normalizer normalizer,
            SpecialTokenMode specialTokenMode)
            throws IOException, InterruptedException {
        GGUF gguf = DATA.getOrDownloadMetadata(model);
        TokenizerMetadata metadata = TestDataManager.extractTokenizerMetadata(gguf);
        return buildTokenizer(metadata, splitterModelType, normalizer, specialTokenMode);
    }

    private static Tokenizer fromGemma3() throws IOException, InterruptedException {
        GGUF gguf = DATA.getOrDownloadMetadata(TestModel.GEMMA_3_4B_UNSLOTH);
        TokenizerMetadata metadata = TestDataManager.extractTokenizerMetadata(gguf);
        Tokenizer base = buildGemmaTokenizer(metadata);
        return wrapWithSpecialTokenInjection(base, SpecialTokenMode.EXACT_LITERAL);
    }

    private static Tokenizer fromGemma4() throws IOException, InterruptedException {
        String[] urls = {
            "https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-e2b-it-Q8_0.gguf",
            "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q8_0.gguf"
        };
        for (int i = 0; i < urls.length; i++) {
            try {
                GGUF gguf =
                        DATA.getOrDownloadMetadata(
                                cacheKey("google_gemma4_v1", i, urls[i]), urls[i]);
                TokenizerMetadata metadata = TestDataManager.extractTokenizerMetadata(gguf);
                Tokenizer base = buildGemmaTokenizer(metadata);
                return wrapWithSpecialTokenInjection(base, SpecialTokenMode.EXACT_LITERAL);
            } catch (IOException e) {
                // try next source
            }
        }
        throw new IOException("No accessible gemma4 GGUF metadata source");
    }

    private static Tokenizer fromQwen35() throws IOException, InterruptedException {
        Tokenizer base =
                fromModel(
                        TestModel.QWEN3_0_6B,
                        "qwen3.5",
                        Normalizer.unicode(Form.NFC),
                        SpecialTokenMode.EXACT_LITERAL);
        return withThaiSegmentation(base);
    }

    private static Tokenizer fromHfQwen35(String modelRef, String revision)
            throws IOException, InterruptedException {
        Tokenizer base =
                fromHfTokenizerJson(
                        modelRef,
                        revision,
                        "qwen3.5",
                        Normalizer.unicode(Form.NFC),
                        SpecialTokenMode.EXACT_LITERAL);
        return withThaiSegmentation(base);
    }

    private static Tokenizer fromDeepSeekV3() throws IOException, InterruptedException {
        String[] urls = {
            "https://huggingface.co/lmstudio-community/DeepSeek-V3-0324-GGUF/resolve/main/DeepSeek-V3-0324-Q4_K_M-00001-of-00011.gguf",
            "https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF/resolve/main/Q4_K_M/DeepSeek-V3-0324-Q4_K_M-00001-of-00011.gguf"
        };
        for (int i = 0; i < urls.length; i++) {
            try {
                GGUF gguf =
                        DATA.getOrDownloadMetadata(
                                cacheKey("deepseek_v3_0324_v1", i, urls[i]), urls[i]);
                TokenizerMetadata metadata = TestDataManager.extractTokenizerMetadata(gguf);
                Tokenizer base =
                        buildTokenizer(
                                metadata,
                                "identity",
                                Normalizer.identity(),
                                SpecialTokenMode.EXACT_LITERAL);
                return withDeepSeekSegmentation(base);
            } catch (IOException e) {
                // try next source
            }
        }
        throw new IOException("No accessible deepseek v3 GGUF metadata source");
    }

    private static Tokenizer fromHfDeepSeekV3(String modelRef, String revision)
            throws IOException, InterruptedException {
        Tokenizer base =
                fromHfTokenizerJson(
                        modelRef,
                        revision,
                        "identity",
                        Normalizer.identity(),
                        SpecialTokenMode.EXACT_LITERAL);
        return withDeepSeekSegmentation(base);
    }

    private static Tokenizer withThaiSegmentation(Tokenizer tokenizer) {
        return withPreSegmentation(tokenizer, ModelFamilyTokenizers::segmentThaiRuns);
    }

    private static Tokenizer withDeepSeekSegmentation(Tokenizer tokenizer) {
        return withPreSegmentation(
                tokenizer, ModelFamilyTokenizers::segmentDeepSeekV3WithNelGrouping);
    }

    private static Tokenizer fromHfTokenizerJson(
            String modelRef,
            String revision,
            String splitterModelType,
            Normalizer normalizer,
            SpecialTokenMode fallbackSpecialMode)
            throws IOException, InterruptedException {
        String rev = (revision == null || revision.isBlank()) ? "main" : revision;
        String url = "https://huggingface.co/" + modelRef + "/resolve/" + rev + "/tokenizer.json";
        String cacheKey = "hf_tok_json_" + Integer.toHexString((modelRef + "@" + rev).hashCode());
        Path tokenizerJson = DATA.getOrDownloadFile(cacheKey, url, ".json");
        String json = Files.readString(tokenizerJson, StandardCharsets.UTF_8);
        @SuppressWarnings("unchecked")
        Map<String, Object> root = (Map<String, Object>) Json.parse(json);
        Map<String, Object> tokenizerConfig =
                loadOptionalHfJson(modelRef, rev, "tokenizer_config.json");

        @SuppressWarnings("unchecked")
        Map<String, Object> model = (Map<String, Object>) root.get("model");
        if (model == null) {
            throw new IOException("tokenizer.json missing model section for " + modelRef);
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> vocabMap = (Map<String, Object>) model.get("vocab");
        if (vocabMap == null || vocabMap.isEmpty()) {
            throw new IOException("tokenizer.json missing vocab for " + modelRef);
        }

        int maxId = -1;
        for (Object v : vocabMap.values()) {
            maxId = Math.max(maxId, ((Number) v).intValue());
        }

        String[] tokens = new String[maxId + 1];
        for (Map.Entry<String, Object> e : vocabMap.entrySet()) {
            int id = ((Number) e.getValue()).intValue();
            tokens[id] = e.getKey();
        }

        int[] tokenTypes = new int[tokens.length];
        java.util.Arrays.fill(tokenTypes, StandardTokenType.NORMAL.getId());
        SpecialTokenMode inferredSpecialMode = fallbackSpecialMode;
        Map<String, AddedTokenSpec> addedTokenSpecs = new LinkedHashMap<>();

        Object addedObj = root.get("added_tokens");
        if (addedObj instanceof List<?>) {
            for (Object obj : (List<?>) addedObj) {
                if (!(obj instanceof Map<?, ?>)) {
                    continue;
                }
                @SuppressWarnings("unchecked")
                Map<String, Object> at = (Map<String, Object>) obj;
                Object idObj = at.get("id");
                Object contentObj = at.get("content");
                if (!(idObj instanceof Number) || !(contentObj instanceof String)) {
                    continue;
                }
                int id = ((Number) idObj).intValue();
                if (id >= tokens.length) {
                    tokens = java.util.Arrays.copyOf(tokens, id + 1);
                    tokenTypes = java.util.Arrays.copyOf(tokenTypes, id + 1);
                    for (int i = 0; i < tokenTypes.length; i++) {
                        if (tokenTypes[i] == 0) {
                            tokenTypes[i] = StandardTokenType.NORMAL.getId();
                        }
                    }
                }
                if (tokens[id] == null) {
                    tokens[id] = (String) contentObj;
                }
                boolean special = Boolean.TRUE.equals(at.get("special"));
                if (special) {
                    tokenTypes[id] = StandardTokenType.CONTROL.getId();
                }
                boolean lstrip = Boolean.TRUE.equals(at.get("lstrip"));
                boolean rstrip = Boolean.TRUE.equals(at.get("rstrip"));
                boolean singleWord = Boolean.TRUE.equals(at.get("single_word"));
                boolean normalized = !Boolean.FALSE.equals(at.get("normalized"));
                addedTokenSpecs.put(
                        (String) contentObj,
                        new AddedTokenSpec(
                                (String) contentObj,
                                special,
                                lstrip,
                                rstrip,
                                singleWord,
                                normalized));
                if (special && (lstrip || rstrip)) {
                    inferredSpecialMode = SpecialTokenMode.TRIM_ONE_SPACE_AROUND;
                }
            }
        }

        mergeSpecialTokensFromMaps(modelRef, rev, tokens, tokenTypes, addedTokenSpecs);

        String[] merges = extractMerges(model.get("merges"));
        TokenizerMetadata metadata =
                new TokenizerMetadata(
                        "gpt2",
                        modelRef,
                        "hf",
                        tokens,
                        null,
                        tokenTypes,
                        merges,
                        null,
                        null,
                        null,
                        null);

        Normalizer parsedNormalizer = parseHfNormalizer(root.get("normalizer"));
        Normalizer effectiveNormalizer =
                parsedNormalizer != null
                        ? parsedNormalizer
                        : (normalizer != null ? normalizer : Normalizer.identity());

        Splitter parsedSplitter = parseHfPreTokenizer(root.get("pre_tokenizer"));
        Splitter effectiveSplitter = parsedSplitter;
        String tokenizerClass =
                tokenizerConfig == null ? null : (String) tokenizerConfig.get("tokenizer_class");
        if (tokenizerClass != null
                && tokenizerClass.contains("GPT2Tokenizer")
                && modelRef.startsWith("ibm-granite/")) {
            effectiveSplitter = ModelTextSplitters.createSplitter("gpt2");
        }
        if (effectiveSplitter == null) {
            effectiveSplitter = ModelTextSplitters.createSplitter(splitterModelType);
        }

        Tokenizer base =
                buildTokenizer(
                        metadata, effectiveNormalizer, effectiveSplitter, inferredSpecialMode);
        if (fallbackSpecialMode == SpecialTokenMode.NONE) {
            return base;
        }
        return addedTokenSpecs.isEmpty() ? base : wrapWithAddedTokenSpecs(base, addedTokenSpecs);
    }

    private static Map<String, Object> loadOptionalHfJson(
            String modelRef, String revision, String filename)
            throws IOException, InterruptedException {
        String url = "https://huggingface.co/" + modelRef + "/resolve/" + revision + "/" + filename;
        String cacheKey =
                "hf_tok_opt_"
                        + Integer.toHexString(
                                (modelRef + "@" + revision + ":" + filename).hashCode());
        try {
            Path file = DATA.getOrDownloadFile(cacheKey, url, ".json");
            String json = Files.readString(file, StandardCharsets.UTF_8);
            @SuppressWarnings("unchecked")
            Map<String, Object> root = (Map<String, Object>) Json.parse(json);
            return root;
        } catch (IOException e) {
            return null;
        }
    }

    private static void mergeSpecialTokensFromMaps(
            String modelRef,
            String revision,
            String[] tokens,
            int[] tokenTypes,
            Map<String, AddedTokenSpec> addedTokenSpecs)
            throws IOException, InterruptedException {
        String base = "https://huggingface.co/" + modelRef + "/resolve/" + revision + "/";
        mergeSpecialTokensFromMapFile(
                base + "special_tokens_map.json", modelRef, tokens, tokenTypes, addedTokenSpecs);
        mergeSpecialTokensFromMapFile(
                base + "tokenizer_config.json", modelRef, tokens, tokenTypes, addedTokenSpecs);
    }

    private static void mergeSpecialTokensFromMapFile(
            String url,
            String modelRef,
            String[] tokens,
            int[] tokenTypes,
            Map<String, AddedTokenSpec> addedTokenSpecs)
            throws IOException, InterruptedException {
        String cacheKey = "hf_tok_meta_" + Integer.toHexString((modelRef + "@" + url).hashCode());
        Path file;
        try {
            file = DATA.getOrDownloadFile(cacheKey, url, ".json");
        } catch (IOException e) {
            return;
        }
        String json = Files.readString(file, StandardCharsets.UTF_8);
        @SuppressWarnings("unchecked")
        Map<String, Object> root = (Map<String, Object>) Json.parse(json);
        collectSpecialTokenContents(root, addedTokenSpecs);
        for (String content : addedTokenSpecs.keySet()) {
            Integer id = findTokenId(tokens, content);
            if (id != null) {
                tokenTypes[id] = StandardTokenType.CONTROL.getId();
            }
        }
    }

    private static void collectSpecialTokenContents(
            Map<String, Object> map, Map<String, AddedTokenSpec> addedTokenSpecs) {
        for (Map.Entry<String, Object> e : map.entrySet()) {
            Object v = e.getValue();
            if (v instanceof String) {
                maybeAddSpecial((String) v, addedTokenSpecs);
            } else if (v instanceof Map<?, ?>) {
                @SuppressWarnings("unchecked")
                Map<String, Object> m = (Map<String, Object>) v;
                Object content = m.get("content");
                if (content instanceof String) {
                    boolean lstrip = Boolean.TRUE.equals(m.get("lstrip"));
                    boolean rstrip = Boolean.TRUE.equals(m.get("rstrip"));
                    boolean singleWord = Boolean.TRUE.equals(m.get("single_word"));
                    boolean normalized = !Boolean.FALSE.equals(m.get("normalized"));
                    addedTokenSpecs.put(
                            (String) content,
                            new AddedTokenSpec(
                                    (String) content,
                                    true,
                                    lstrip,
                                    rstrip,
                                    singleWord,
                                    normalized));
                }
            } else if (v instanceof List<?>) {
                for (Object item : (List<?>) v) {
                    if (item instanceof String) {
                        maybeAddSpecial((String) item, addedTokenSpecs);
                    } else if (item instanceof Map<?, ?>) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> m = (Map<String, Object>) item;
                        Object content = m.get("content");
                        if (content instanceof String) {
                            boolean lstrip = Boolean.TRUE.equals(m.get("lstrip"));
                            boolean rstrip = Boolean.TRUE.equals(m.get("rstrip"));
                            boolean singleWord = Boolean.TRUE.equals(m.get("single_word"));
                            boolean normalized = !Boolean.FALSE.equals(m.get("normalized"));
                            addedTokenSpecs.put(
                                    (String) content,
                                    new AddedTokenSpec(
                                            (String) content,
                                            true,
                                            lstrip,
                                            rstrip,
                                            singleWord,
                                            normalized));
                        }
                    }
                }
            }
        }
    }

    private static void maybeAddSpecial(String token, Map<String, AddedTokenSpec> specs) {
        if (token == null || token.isEmpty()) {
            return;
        }
        specs.putIfAbsent(token, new AddedTokenSpec(token, true, false, false, false, true));
    }

    private static Integer findTokenId(String[] tokens, String content) {
        for (int i = 0; i < tokens.length; i++) {
            if (content.equals(tokens[i])) {
                return i;
            }
        }
        return null;
    }

    private static Normalizer parseHfNormalizer(Object normalizerObj) {
        if (!(normalizerObj instanceof Map<?, ?>)) {
            return null;
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> normalizer = (Map<String, Object>) normalizerObj;
        String type = (String) normalizer.get("type");
        if (type == null) {
            return null;
        }
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
                return null;
        }
    }

    private static Normalizer parseSequenceNormalizer(Map<String, Object> normalizer) {
        Object childrenObj = normalizer.get("normalizers");
        if (!(childrenObj instanceof List<?>)) {
            return null;
        }
        List<Normalizer> chain = new ArrayList<>();
        for (Object child : (List<?>) childrenObj) {
            Normalizer parsed = parseHfNormalizer(child);
            if (parsed != null) {
                chain.add(parsed);
            }
        }
        if (chain.isEmpty()) {
            return Normalizer.identity();
        }
        return Normalizer.sequence(chain.toArray(new Normalizer[0]));
    }

    private static Normalizer parseReplaceNormalizer(Map<String, Object> normalizer) {
        Object patternObj = normalizer.get("pattern");
        String content = (String) normalizer.get("content");
        if (!(patternObj instanceof Map<?, ?>) || content == null) {
            return null;
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> patternMap = (Map<String, Object>) patternObj;
        Object literal = patternMap.get("String");
        if (!(literal instanceof String)) {
            return null;
        }
        String needle = (String) literal;
        return text -> text.toString().replace(needle, content);
    }

    private static Splitter parseHfPreTokenizer(Object preTokenizerObj) {
        if (!(preTokenizerObj instanceof Map<?, ?>)) {
            return null;
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> preTokenizer = (Map<String, Object>) preTokenizerObj;
        String type = (String) preTokenizer.get("type");
        if (type == null) {
            return null;
        }
        switch (type) {
            case "Split":
                return parseSplitPreTokenizer(preTokenizer);
            case "Sequence":
                return parseSequencePreTokenizer(preTokenizer);
            default:
                return null;
        }
    }

    private static Splitter parseSequencePreTokenizer(Map<String, Object> preTokenizer) {
        Object childrenObj = preTokenizer.get("pretokenizers");
        if (!(childrenObj instanceof List<?>)) {
            return null;
        }
        List<Splitter> splitters = new ArrayList<>();
        for (Object childObj : (List<?>) childrenObj) {
            if (!(childObj instanceof Map<?, ?>)) {
                continue;
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> child = (Map<String, Object>) childObj;
            String childType = (String) child.get("type");
            if ("Split".equals(childType)) {
                Splitter s = parseSplitPreTokenizer(child);
                if (s != null) {
                    splitters.add(s);
                }
            }
        }
        if (splitters.isEmpty()) {
            return null;
        }
        return Splitter.sequence(splitters.toArray(new Splitter[0]));
    }

    private static Splitter parseSplitPreTokenizer(Map<String, Object> preTokenizer) {
        Object patternObj = preTokenizer.get("pattern");
        if (!(patternObj instanceof Map<?, ?>)) {
            return null;
        }
        String behavior = (String) preTokenizer.getOrDefault("behavior", "Removed");
        boolean invert = Boolean.TRUE.equals(preTokenizer.get("invert"));
        @SuppressWarnings("unchecked")
        Map<String, Object> patternMap = (Map<String, Object>) patternObj;
        Object regex = patternMap.get("Regex");
        if (regex instanceof String) {
            Pattern pattern = Pattern.compile((String) regex, Pattern.UNICODE_CHARACTER_CLASS);
            if (("Removed".equals(behavior) && invert)
                    || ("Isolated".equals(behavior) && !invert)) {
                return Splitter.regex(pattern);
            }
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
                return null;
            }
            return (text, startInclusive, endExclusive, consumer) -> {
                String source = text.subSequence(startInclusive, endExclusive).toString();
                for (Span span : splitByLiteral(source, s, behavior, invert)) {
                    consumer.accept(
                            text, startInclusive + span.start(), startInclusive + span.end());
                }
            };
        }
        return null;
    }

    private static List<Span> splitByLiteral(
            String text, String literal, String behavior, boolean invert) {
        List<Span> spans = new ArrayList<>();
        int from = 0;
        int idx;
        while ((idx = text.indexOf(literal, from)) >= 0) {
            spans.add(new Span(idx, idx + literal.length(), true));
            from = idx + literal.length();
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
                for (Span s : all) {
                    if (!isSeparator(s, invert) && s.start() < s.end()) {
                        out.add(new Span(s.start(), s.end(), false));
                    }
                }
                break;
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

    private static Map<String, ModelFamilySpec> createModelFamilySpecs() {
        Map<String, ModelFamilySpec> specs = new LinkedHashMap<>();
        // Qwen tokenizer family uses NFC normalization in the official HF tokenizer backend.
        // Source: transformers Qwen2 tokenizer implementation delegates to tokenizer.json backend
        // config (normalizer=NFC):
        // https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/tokenization_qwen2.py
        specs.put(
                "alibaba.qwen3_5",
                new ModelFamilySpec(
                        TestModel.QWEN3_0_6B,
                        "qwen3.5",
                        Normalizer.unicode(Form.NFC),
                        SpecialTokenMode.EXACT_LITERAL));
        specs.put(
                "mistral.tekken",
                new ModelFamilySpec(
                        TestModel.MISTRAL_3_3B_BARTOWSKI,
                        "tekken",
                        Normalizer.identity(),
                        SpecialTokenMode.NONE));
        return java.util.Collections.unmodifiableMap(specs);
    }

    private static Map<String, UrlFamilySpec> createUrlFamilySpecs() {
        Map<String, UrlFamilySpec> specs = new LinkedHashMap<>();
        specs.put(
                "microsoft.phi4",
                new UrlFamilySpec(
                        "microsoft_phi4_v2",
                        "phi4",
                        Normalizer.identity(),
                        SpecialTokenMode.TRIM_ONE_SPACE_AROUND,
                        "No accessible phi4 GGUF metadata source",
                        // Official Microsoft Phi-4 GGUF release.
                        "https://huggingface.co/microsoft/phi-4-gguf/resolve/main/phi-4-Q4_K.gguf",
                        // Fallback mirrors.
                        "https://huggingface.co/unsloth/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf",
                        "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"));
        specs.put(
                "meta.llama3",
                new UrlFamilySpec(
                        "meta_llama3_v1",
                        "llama",
                        Normalizer.identity(),
                        SpecialTokenMode.EXACT_LITERAL,
                        "No accessible llama3 GGUF metadata source",
                        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                        "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"));
        specs.put(
                "moonshot.kimi2_5",
                new UrlFamilySpec(
                        "moonshot_kimi2_5_v1",
                        "llama",
                        Normalizer.identity(),
                        SpecialTokenMode.EXACT_LITERAL,
                        "No accessible kimi2.5 GGUF metadata source",
                        "https://huggingface.co/unsloth/Kimi-K2.5-GGUF/resolve/main/Q4_K_M/Kimi-K2.5-Q4_K_M-00001-of-00013.gguf",
                        "https://huggingface.co/AesSedai/Kimi-K2.5-GGUF/resolve/main/IQ2_S/Kimi-K2.5-IQ2_S-00001-of-00008.gguf"));
        specs.put(
                "ibm.granite4_0",
                new UrlFamilySpec(
                        "ibm_granite4_0_v1",
                        "granite4",
                        Normalizer.identity(),
                        SpecialTokenMode.EXACT_LITERAL,
                        "No accessible granite4.0 GGUF metadata source",
                        "https://huggingface.co/ibm-granite/granite-4.0-h-1b-GGUF/resolve/main/granite-4.0-h-1b-Q4_K_M.gguf",
                        "https://huggingface.co/unsloth/granite-4.0-h-1b-GGUF/resolve/main/granite-4.0-h-1b-Q4_K_M.gguf",
                        "https://huggingface.co/unsloth/granite-4.0-1b-GGUF/resolve/main/granite-4.0-1b-Q4_K_M.gguf"));
        specs.put(
                "huggingface.smollm3",
                new UrlFamilySpec(
                        "huggingface_smollm3_v1",
                        "smollm3",
                        Normalizer.identity(),
                        SpecialTokenMode.EXACT_LITERAL,
                        "No accessible smollm3 GGUF metadata source",
                        "https://huggingface.co/ggml-org/SmolLM3-3B-GGUF/resolve/main/SmolLM3-Q4_K_M.gguf",
                        "https://huggingface.co/unsloth/SmolLM3-3B-GGUF/resolve/main/SmolLM3-3B-Q4_K_M.gguf"));
        specs.put(
                "mistral.gpt2_pretekken",
                new UrlFamilySpec(
                        "mistral_gpt2_pretekken_v1",
                        "llama",
                        Normalizer.identity(),
                        SpecialTokenMode.EXACT_LITERAL,
                        "No accessible mistral pre-tekken GGUF metadata source",
                        "https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-GGUF/resolve/main/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
                        "https://huggingface.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF/resolve/main/Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf"));
        return java.util.Collections.unmodifiableMap(specs);
    }

    private static Tokenizer fromUrls(
            String cachePrefix,
            String splitterModelType,
            Normalizer normalizer,
            SpecialTokenMode specialTokenMode,
            String failureMessage,
            String... urls)
            throws IOException, InterruptedException {
        for (int i = 0; i < urls.length; i++) {
            try {
                GGUF gguf = DATA.getOrDownloadMetadata(cacheKey(cachePrefix, i, urls[i]), urls[i]);
                TokenizerMetadata metadata = TestDataManager.extractTokenizerMetadata(gguf);
                return buildTokenizer(metadata, splitterModelType, normalizer, specialTokenMode);
            } catch (IOException e) {
                // try next source
            }
        }
        throw new IOException(failureMessage);
    }

    private static String cacheKey(String cachePrefix, int index, String url) {
        return cachePrefix + "_" + index + "_" + Integer.toHexString(url.hashCode());
    }

    private static Tokenizer buildTokenizer(
            TokenizerMetadata metadata,
            String splitterModelType,
            Normalizer normalizer,
            SpecialTokenMode specialTokenMode) {
        String[] tokens = metadata.tokens();
        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        for (int i = 0; i < tokens.length; i++) {
            tokenToId.put(tokens[i], i);
        }

        LongLongMap merges = buildMerges(metadata, tokenToId);
        Splitter splitter = ModelTextSplitters.createSplitter(splitterModelType);
        return buildTokenizer(metadata, normalizer, splitter, specialTokenMode, merges);
    }

    private static Tokenizer buildTokenizer(
            TokenizerMetadata metadata,
            Normalizer normalizer,
            Splitter splitter,
            SpecialTokenMode specialTokenMode) {
        String[] tokens = metadata.tokens();
        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        for (int i = 0; i < tokens.length; i++) {
            tokenToId.put(tokens[i], i);
        }
        LongLongMap merges = buildMerges(metadata, tokenToId);
        return buildTokenizer(metadata, normalizer, splitter, specialTokenMode, merges);
    }

    private static Tokenizer buildTokenizer(
            TokenizerMetadata metadata,
            Normalizer normalizer,
            Splitter splitter,
            SpecialTokenMode specialTokenMode,
            LongLongMap merges) {
        Vocabulary vocabulary =
                new VocabularyImpl(metadata.tokens(), metadata.scores(), metadata.tokenTypes());
        Tokenizer base = new GPT2Tokenizer(vocabulary, Normalizer.identity(), splitter, merges);
        Tokenizer normalized =
                normalizer == null
                        ? base
                        : Tokenizers.pipeline(base).normalizer(normalizer).build();
        return wrapWithSpecialTokenInjection(normalized, specialTokenMode);
    }

    private static Tokenizer buildGemmaTokenizer(TokenizerMetadata metadata) {
        VocabularyImpl vocabulary =
                new VocabularyImpl(metadata.tokens(), metadata.scores(), metadata.tokenTypes());
        float[] scores = metadata.scores();
        int byte0 = vocabulary.contains("<0x00>") ? vocabulary.id("<0x00>") : -1;

        return new Tokenizer() {
            @Override
            public Vocabulary vocabulary() {
                return vocabulary;
            }

            @Override
            public void encodeInto(
                    CharSequence text,
                    int startInclusive,
                    int endExclusive,
                    IntSequence.Builder out) {
                out.addAll(encode(text.subSequence(startInclusive, endExclusive).toString()));
            }

            public IntSequence encode(String text) {
                String metaspace = text.replace(' ', '\u2581');
                List<Integer> ids = new ArrayList<>();

                for (int pos = 0; pos < metaspace.length(); ) {
                    if (metaspace.charAt(pos) == '\u2581') {
                        int end = pos;
                        while (end < metaspace.length() && metaspace.charAt(end) == '\u2581') {
                            end++;
                        }
                        int runLength = end - pos;
                        if (runLength > 1) {
                            String spaces = " ".repeat(runLength);
                            if (vocabulary.contains(spaces)) {
                                ids.add(vocabulary.id(spaces));
                                pos = end;
                                continue;
                            }
                        }
                    }

                    if (metaspace.charAt(pos) == '<') {
                        int close = metaspace.indexOf('>', pos + 1);
                        if (close > pos && close - pos <= 32) {
                            String tag = metaspace.substring(pos, close + 1);
                            if (vocabulary.contains(tag)) {
                                ids.add(vocabulary.id(tag));
                                pos = close + 1;
                                continue;
                            }
                        }
                    }

                    int cp = metaspace.codePointAt(pos);
                    String s = new String(Character.toChars(cp));
                    if (vocabulary.contains(s)) {
                        ids.add(vocabulary.id(s));
                    } else {
                        if (byte0 < 0) {
                            throw new NoSuchElementException("Missing <0x00> byte fallback token");
                        }
                        for (byte b : s.getBytes(StandardCharsets.UTF_8)) {
                            ids.add(Byte.toUnsignedInt(b) + byte0);
                        }
                    }
                    pos += Character.charCount(cp);
                }

                while (ids.size() >= 2) {
                    float bestScore = -1.0e10f;
                    int bestId = -1;
                    int bestPos = -1;

                    for (int i = 0; i + 1 < ids.size(); i++) {
                        String merged =
                                vocabulary.token(ids.get(i)) + vocabulary.token(ids.get(i + 1));
                        if (!vocabulary.contains(merged)) {
                            continue;
                        }
                        int mergedId = vocabulary.id(merged);
                        float score =
                                scores != null && mergedId < scores.length ? scores[mergedId] : 0f;
                        if (score > bestScore) {
                            bestScore = score;
                            bestId = mergedId;
                            bestPos = i;
                        }
                    }

                    if (bestPos < 0) {
                        break;
                    }

                    ids.set(bestPos, bestId);
                    ids.remove(bestPos + 1);
                }

                int[] out = new int[ids.size()];
                for (int i = 0; i < ids.size(); i++) {
                    out[i] = ids.get(i);
                }
                return IntSequence.wrap(out);
            }

            @Override
            public String decode(IntSequence tokens) {
                StringBuilder sb = new StringBuilder();
                java.io.ByteArrayOutputStream byteRun = new java.io.ByteArrayOutputStream();
                for (int i = 0; i < tokens.length(); i++) {
                    int id = tokens.intAt(i);
                    String token = vocabulary.token(id);
                    if (vocabulary.isTokenOfType(id, StandardTokenType.BYTE)
                            && token.length() == 6
                            && token.startsWith("<0x")
                            && token.endsWith(">")) {
                        String hex = token.substring(3, 5);
                        byteRun.write(Integer.parseInt(hex, 16));
                    } else {
                        if (byteRun.size() > 0) {
                            sb.append(byteRun.toString(StandardCharsets.UTF_8));
                            byteRun.reset();
                        }
                        sb.append(token.replace('\u2581', ' '));
                    }
                }
                if (byteRun.size() > 0) {
                    sb.append(byteRun.toString(StandardCharsets.UTF_8));
                }
                return sb.toString();
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
            public int countTokens(CharSequence text) {
                return encode(text.toString()).length();
            }

            @Override
            public int countBytes(IntSequence tokens) {
                return decodeBytes(tokens).length;
            }
        };
    }

    private static Tokenizer wrapWithSpecialTokenInjection(
            Tokenizer tokenizer, SpecialTokenMode specialTokenMode) {
        if (specialTokenMode == SpecialTokenMode.NONE) {
            return tokenizer;
        }
        Vocabulary vocabulary = tokenizer.vocabulary();
        Set<String> specials = new java.util.LinkedHashSet<>();
        for (Map.Entry<String, Integer> e : vocabulary) {
            if (e.getKey() == null || e.getKey().isEmpty()) {
                continue;
            }
            if (vocabulary.isTokenOfType(e.getValue(), StandardTokenType.CONTROL)) {
                specials.add(e.getKey());
            }
        }
        if (specials.isEmpty()) {
            return tokenizer;
        }
        return new SpecialAwareTokenizer(tokenizer, specials, specialTokenMode, Map.of());
    }

    private static Tokenizer wrapWithAddedTokenSpecs(
            Tokenizer tokenizer, Map<String, AddedTokenSpec> specs) {
        Set<String> specials = new java.util.LinkedHashSet<>();
        for (Map.Entry<String, AddedTokenSpec> e : specs.entrySet()) {
            if (e.getValue().special()) {
                specials.add(e.getKey());
            }
        }
        if (specials.isEmpty()) {
            return tokenizer;
        }
        return new SpecialAwareTokenizer(
                tokenizer, specials, SpecialTokenMode.EXACT_LITERAL, specs);
    }

    private static Tokenizer withPreSegmentation(
            Tokenizer tokenizer, Function<String, List<String>> segmenter) {
        return Tokenizers.pipeline(tokenizer)
                .splitter(
                        (text, startInclusive, endExclusive, consumer) -> {
                            if (startInclusive >= endExclusive) {
                                return;
                            }
                            int cursor = startInclusive;
                            String slice =
                                    text.subSequence(startInclusive, endExclusive).toString();
                            List<String> segments = segmenter.apply(slice);
                            if (segments == null || segments.isEmpty()) {
                                consumer.accept(text, startInclusive, endExclusive);
                                return;
                            }
                            for (String segment : segments) {
                                if (segment == null
                                        || segment.isEmpty()
                                        || cursor >= endExclusive) {
                                    continue;
                                }
                                int end = cursor + segment.length();
                                if (end > endExclusive) {
                                    end = endExclusive;
                                }
                                if (end <= cursor) {
                                    continue;
                                }
                                consumer.accept(text, cursor, end);
                                cursor = end;
                            }
                            if (cursor < endExclusive) {
                                consumer.accept(text, cursor, endExclusive);
                            }
                        })
                .build();
    }

    private static List<String> segmentThaiRuns(String text) {
        List<String> out = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            int cpLen = Character.charCount(cp);
            if (cp == ' ' && i + cpLen < text.length()) {
                int nextCp = text.codePointAt(i + cpLen);
                if (isThaiCodePoint(nextCp)) {
                    if (current.length() > 0) {
                        out.add(current.toString());
                        current.setLength(0);
                    }
                    out.add(" ");
                    i += cpLen;
                    continue;
                }
            }
            if (isThaiCodePoint(cp)) {
                if (current.length() > 0) {
                    out.add(current.toString());
                    current.setLength(0);
                }
                out.add(new String(Character.toChars(cp)));
            } else {
                current.appendCodePoint(cp);
            }
            i += cpLen;
        }
        if (current.length() > 0) {
            out.add(current.toString());
        }
        if (out.isEmpty()) {
            out.add(text);
        }
        return out;
    }

    private static List<String> segmentDeepSeekV3WithNelGrouping(String text) {
        List<CharSequence> base =
                ModelTextSplitters.createSplitter("deepseek-v3").splitAllToListEagerly(text);
        List<String> chunks = new ArrayList<>(base.size());
        for (CharSequence cs : base) {
            chunks.add(cs.toString());
        }
        List<String> merged = new ArrayList<>(chunks.size());
        for (int i = 0; i < chunks.size(); i++) {
            String current = chunks.get(i);
            if (i + 1 < chunks.size() && " ".equals(current)) {
                String next = chunks.get(i + 1);
                if (!next.isEmpty() && next.charAt(0) == '\u0085') {
                    merged.add(current + next);
                    i++;
                    continue;
                }
            }
            merged.add(current);
        }
        return merged;
    }

    private static boolean isThaiCodePoint(int codePoint) {
        return codePoint >= 0x0E00 && codePoint <= 0x0E7F;
    }

    private static final class SpecialAwareTokenizer implements Tokenizer {
        private final Tokenizer delegate;
        private final Set<String> specials;
        private final Pattern specialPattern;
        private final SpecialTokenMode specialTokenMode;
        private final Map<String, AddedTokenSpec> addedTokenSpecs;

        private SpecialAwareTokenizer(
                Tokenizer delegate,
                Set<String> specials,
                SpecialTokenMode specialTokenMode,
                Map<String, AddedTokenSpec> addedTokenSpecs) {
            this.delegate = delegate;
            this.specials = specials;
            this.specialTokenMode = specialTokenMode;
            this.addedTokenSpecs = addedTokenSpecs;
            this.specialPattern =
                    Pattern.compile(
                            specials.stream()
                                    .map(Pattern::quote)
                                    .collect(Collectors.joining("|", "(", ")")));
        }

        private void encodeSpecialAware(
                CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out) {
            if (startInclusive >= endExclusive) {
                return;
            }
            Matcher matcher = specialPattern.matcher(text);
            matcher.region(startInclusive, endExclusive);
            int lastEnd = startInclusive;
            while (matcher.find()) {
                String token = matcher.group();
                AddedTokenSpec spec = addedTokenSpecs.get(token);
                if (spec != null
                        && spec.singleWord()
                        && !isWordBoundary(text, matcher.start(), matcher.end(), endExclusive)) {
                    continue;
                }

                int start = matcher.start();
                int leftBoundary = start;
                boolean lstrip =
                        (spec != null && spec.lstrip())
                                || (specialTokenMode == SpecialTokenMode.TRIM_ONE_SPACE_AROUND);
                boolean rstrip =
                        (spec != null && spec.rstrip())
                                || (specialTokenMode == SpecialTokenMode.TRIM_ONE_SPACE_AROUND);

                if (lstrip && leftBoundary > lastEnd && text.charAt(leftBoundary - 1) == ' ') {
                    leftBoundary--;
                }
                if (leftBoundary > lastEnd) {
                    delegate.encodeInto(text, lastEnd, leftBoundary, out);
                }
                if (specials.contains(token)) {
                    out.add(delegate.vocabulary().id(token));
                } else {
                    delegate.encodeInto(token, out);
                }
                lastEnd = matcher.end();
                if (rstrip && lastEnd < endExclusive && text.charAt(lastEnd) == ' ') {
                    lastEnd++;
                }
            }
            if (lastEnd < endExclusive) {
                delegate.encodeInto(text, lastEnd, endExclusive, out);
            }
        }

        public IntSequence encode(String text) {
            IntSequence.Builder out = IntSequence.newBuilder(Math.max(8, text.length()));
            encodeSpecialAware(text, 0, text.length(), out);
            return out.build();
        }

        @Override
        public void encodeInto(
                CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out) {
            encodeSpecialAware(text, startInclusive, endExclusive, out);
        }

        private static boolean isWordBoundary(
                CharSequence text, int start, int end, int regionEndExclusive) {
            boolean leftOk = start == 0 || !isWord(Character.codePointBefore(text, start));
            boolean rightOk =
                    end >= regionEndExclusive || !isWord(Character.codePointAt(text, end));
            return leftOk && rightOk;
        }

        private static boolean isWord(int codePoint) {
            return Character.isLetterOrDigit(codePoint) || codePoint == '_';
        }

        @Override
        public Vocabulary vocabulary() {
            return delegate.vocabulary();
        }

        @Override
        public byte[] decodeBytes(IntSequence tokens) {
            return delegate.decodeBytes(tokens);
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
            return delegate.decodeBytesInto(tokens, tokenStartIndex, out);
        }

        @Override
        public int countTokens(CharSequence text) {
            Objects.requireNonNull(text, "text");
            IntSequence.Builder out = IntSequence.newBuilder(Math.max(8, text.length()));
            encodeInto(text, 0, text.length(), out);
            return out.size();
        }

        @Override
        public int countBytes(IntSequence tokens) {
            return delegate.countBytes(tokens);
        }
    }

    private static LongLongMap buildMerges(
            TokenizerMetadata metadata, Map<String, Integer> tokenToId) {
        List<long[]> pairs = new ArrayList<>();
        String[] merges = metadata.merges();
        if (merges != null) {
            for (int rank = 0; rank < merges.length; rank++) {
                String[] parts = merges[rank].split(" ");
                if (parts.length != 2) {
                    continue;
                }
                Integer leftId = tokenToId.get(parts[0]);
                Integer rightId = tokenToId.get(parts[1]);
                Integer mergedId = tokenToId.get(parts[0] + parts[1]);
                if (leftId != null && rightId != null && mergedId != null) {
                    pairs.add(new long[] {IntPair.of(leftId, rightId), IntPair.of(mergedId, rank)});
                }
            }
        }
        long[] keys = new long[pairs.size()];
        long[] values = new long[pairs.size()];
        for (int i = 0; i < pairs.size(); i++) {
            keys[i] = pairs.get(i)[0];
            values[i] = pairs.get(i)[1];
        }
        return new LongLongMap(keys, values);
    }

    private record ModelFamilySpec(
            TestModel model,
            String splitterModelType,
            Normalizer normalizer,
            SpecialTokenMode specialTokenMode) {}

    private record UrlFamilySpec(
            String cachePrefix,
            String splitterModelType,
            Normalizer normalizer,
            SpecialTokenMode specialTokenMode,
            String failureMessage,
            String... urls) {}

    private record AddedTokenSpec(
            String content,
            boolean special,
            boolean lstrip,
            boolean rstrip,
            boolean singleWord,
            boolean normalized) {}

    private record Span(int start, int end, boolean match) {}

    private enum SpecialTokenMode {
        NONE,
        EXACT_LITERAL,
        TRIM_ONE_SPACE_AROUND
    }
}
