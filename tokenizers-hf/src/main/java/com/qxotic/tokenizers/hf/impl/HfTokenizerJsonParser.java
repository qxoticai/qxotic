package com.qxotic.tokenizers.hf.impl;

import com.qxotic.format.json.JSON;
import com.qxotic.tokenizers.hf.HuggingFaceTokenizerException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Parses HuggingFace tokenizer JSON files and extracts BPE tokenizer data.
 *
 * <p>Supports strict extraction with hard-fail semantics for unsupported configurations. Requires
 * explicit regex pattern from the pre_tokenizer configuration.
 */
public class HfTokenizerJsonParser {

    private final Path tokenizerJsonPath;
    private final Map<String, Object> root;

    /**
     * Creates a parser for the given tokenizer.json file.
     *
     * @param tokenizerJsonPath path to tokenizer.json
     * @throws HuggingFaceTokenizerException if file cannot be read
     */
    public HfTokenizerJsonParser(Path tokenizerJsonPath) {
        this(tokenizerJsonPath, null);
    }

    /**
     * Creates a parser for the given tokenizer.json file with optional merges file.
     *
     * @param tokenizerJsonPath path to tokenizer.json
     * @param mergesTxtPath path to merges.txt (optional, may be null)
     * @throws HuggingFaceTokenizerException if file cannot be read
     */
    public HfTokenizerJsonParser(Path tokenizerJsonPath, Path mergesTxtPath) {
        this.tokenizerJsonPath = tokenizerJsonPath;
        this.root = loadJson(tokenizerJsonPath);
        this.mergesTxtPath = mergesTxtPath;
    }

    private final Path mergesTxtPath;

    /**
     * Parses the tokenizer and returns normalized tokenizer data.
     *
     * @return parsed tokenizer data
     * @throws HuggingFaceTokenizerException if tokenizer is not supported
     */
    public TokenizerData parse() {
        // Verify model type is BPE
        String modelType = extractModelType();
        if (!"BPE".equals(modelType)) {
            throw new HuggingFaceTokenizerException(
                    tokenizerJsonPath.toString(),
                    "Unsupported tokenizer type: " + modelType + ". Only BPE is supported.");
        }

        // Extract vocabulary
        Map<String, Integer> vocab = extractVocabulary();

        // Extract merges
        List<String> merges = extractMerges();

        // Extract regex pattern (required, explicit only)
        String regex = extractRegex();

        // Extract added tokens
        Map<String, Integer> addedTokens = extractAddedTokens();

        return new TokenizerData(vocab, merges, regex, addedTokens);
    }

    /** Loads and parses the JSON file. */
    @SuppressWarnings("unchecked")
    private Map<String, Object> loadJson(Path path) {
        try {
            String content = Files.readString(path);
            Object parsed = JSON.parse(content);
            if (!(parsed instanceof Map)) {
                throw new HuggingFaceTokenizerException(
                        path.toString(), "Invalid tokenizer.json: expected JSON object at root");
            }
            return (Map<String, Object>) parsed;
        } catch (HuggingFaceTokenizerException e) {
            throw e;
        } catch (Exception e) {
            throw new HuggingFaceTokenizerException(
                    path.toString(), "Failed to parse tokenizer.json", e);
        }
    }

    /** Extracts the model type from the tokenizer. */
    @SuppressWarnings("unchecked")
    private String extractModelType() {
        Object modelObj = root.get("model");
        if (!(modelObj instanceof Map)) {
            throw new HuggingFaceTokenizerException(
                    tokenizerJsonPath.toString(),
                    "Missing or invalid 'model' section in tokenizer.json");
        }

        Map<String, Object> model = (Map<String, Object>) modelObj;
        Object type = model.get("type");
        if (!(type instanceof String)) {
            throw new HuggingFaceTokenizerException(
                    tokenizerJsonPath.toString(),
                    "Missing or invalid model.type in tokenizer.json");
        }

        return (String) type;
    }

    /** Extracts the vocabulary from the tokenizer. */
    @SuppressWarnings("unchecked")
    private Map<String, Integer> extractVocabulary() {
        Object modelObj = root.get("model");
        if (!(modelObj instanceof Map)) {
            throw new HuggingFaceTokenizerException(
                    tokenizerJsonPath.toString(), "Missing 'model' section");
        }

        Map<String, Object> model = (Map<String, Object>) modelObj;
        Object vocabObj = model.get("vocab");
        if (!(vocabObj instanceof Map)) {
            throw new HuggingFaceTokenizerException(
                    tokenizerJsonPath.toString(),
                    "Missing or invalid model.vocab in tokenizer.json");
        }

        Map<String, Integer> vocab = new HashMap<>();
        Map<String, Object> vocabMap = (Map<String, Object>) vocabObj;

        for (Map.Entry<String, Object> entry : vocabMap.entrySet()) {
            Object value = entry.getValue();
            if (!(value instanceof Number)) {
                throw new HuggingFaceTokenizerException(
                        tokenizerJsonPath.toString(),
                        "Invalid vocab entry for '"
                                + entry.getKey()
                                + "': expected number, got "
                                + (value == null ? "null" : value.getClass().getSimpleName()));
            }
            int id = ((Number) value).intValue();
            if (id < 0) {
                throw new HuggingFaceTokenizerException(
                        tokenizerJsonPath.toString(),
                        "Invalid vocab entry for '"
                                + entry.getKey()
                                + "': negative token ID "
                                + id);
            }
            vocab.put(entry.getKey(), id);
        }

        // Check for duplicate IDs
        List<Integer> ids = new ArrayList<>(vocab.values());
        Collections.sort(ids);
        for (int i = 1; i < ids.size(); i++) {
            if (ids.get(i).equals(ids.get(i - 1))) {
                throw new HuggingFaceTokenizerException(
                        tokenizerJsonPath.toString(), "Duplicate token ID found: " + ids.get(i));
            }
        }

        return vocab;
    }

    /**
     * Extracts the merges from the tokenizer.
     *
     * <p>Handles both inline merges (in tokenizer.json) and external merges.txt file. Also supports
     * both string format ("token1 token2") and array format (["token1", "token2"]).
     */
    @SuppressWarnings("unchecked")
    private List<String> extractMerges() {
        Object modelObj = root.get("model");
        if (!(modelObj instanceof Map)) {
            throw new HuggingFaceTokenizerException(
                    tokenizerJsonPath.toString(), "Missing 'model' section");
        }

        Map<String, Object> model = (Map<String, Object>) modelObj;
        Object mergesObj = model.get("merges");

        // If no inline merges, try external file
        if (mergesObj == null) {
            return loadMergesFromFile();
        }

        if (!(mergesObj instanceof List)) {
            throw new HuggingFaceTokenizerException(
                    tokenizerJsonPath.toString(),
                    "Invalid model.merges: expected array, got "
                            + mergesObj.getClass().getSimpleName());
        }

        List<Object> mergesList = (List<Object>) mergesObj;
        List<String> merges = new ArrayList<>();

        for (int i = 0; i < mergesList.size(); i++) {
            Object merge = mergesList.get(i);

            if (merge instanceof String) {
                // String format: "token1 token2"
                String mergeStr = (String) merge;
                validateMergeFormat(i, mergeStr);
                merges.add(mergeStr);
            } else if (merge instanceof List) {
                // Array format: ["token1", "token2"]
                List<Object> mergeArray = (List<Object>) merge;
                if (mergeArray.size() != 2) {
                    throw new HuggingFaceTokenizerException(
                            tokenizerJsonPath.toString(),
                            "Invalid merge at index "
                                    + i
                                    + ": expected 2 elements, got "
                                    + mergeArray.size());
                }
                Object left = mergeArray.get(0);
                Object right = mergeArray.get(1);
                if (!(left instanceof String) || !(right instanceof String)) {
                    throw new HuggingFaceTokenizerException(
                            tokenizerJsonPath.toString(),
                            "Invalid merge at index " + i + ": expected strings");
                }
                String mergeStr = left + " " + right;
                merges.add(mergeStr);
            } else {
                throw new HuggingFaceTokenizerException(
                        tokenizerJsonPath.toString(),
                        "Invalid merge at index "
                                + i
                                + ": expected string or array, got "
                                + (merge == null ? "null" : merge.getClass().getSimpleName()));
            }
        }

        return merges;
    }

    /** Loads merges from external merges.txt file. */
    private List<String> loadMergesFromFile() {
        Path mergesFile = mergesTxtPath;

        // If no explicit path provided, look for merges.txt in same directory as tokenizer.json
        if (mergesFile == null) {
            mergesFile = tokenizerJsonPath.getParent().resolve("merges.txt");
        }

        if (!Files.exists(mergesFile)) {
            // No merges file is acceptable for some BPE tokenizers
            return Collections.emptyList();
        }

        try {
            List<String> lines = Files.readAllLines(mergesFile);
            List<String> merges = new ArrayList<>();

            for (int i = 0; i < lines.size(); i++) {
                String line = lines.get(i).trim();

                // Skip empty lines and comments
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }

                validateMergeFormat(i, line);
                merges.add(line);
            }

            return merges;
        } catch (Exception e) {
            throw new HuggingFaceTokenizerException(
                    mergesFile.toString(), "Failed to read merges.txt", e);
        }
    }

    /** Validates merge format (should be "token1 token2"). */
    private void validateMergeFormat(int index, String mergeStr) {
        String[] parts = mergeStr.split(" ");
        if (parts.length != 2) {
            throw new HuggingFaceTokenizerException(
                    tokenizerJsonPath.toString(),
                    "Invalid merge format at index "
                            + index
                            + ": expected 'token1 token2', got '"
                            + mergeStr
                            + "'");
        }
    }

    /**
     * Extracts the explicit regex pattern from the pre_tokenizer.
     *
     * <p>Supports nested Sequence pretokenizers. Requires explicit Regex pattern. Hard-fails if no
     * explicit regex is found.
     */
    @SuppressWarnings("unchecked")
    private String extractRegex() {
        Object preTokenizerObj = root.get("pre_tokenizer");
        if (preTokenizerObj == null) {
            throw new HuggingFaceTokenizerException(
                    tokenizerJsonPath.toString(),
                    "Missing pre_tokenizer configuration in tokenizer.json");
        }

        if (!(preTokenizerObj instanceof Map)) {
            throw new HuggingFaceTokenizerException(
                    tokenizerJsonPath.toString(),
                    "Invalid pre_tokenizer: expected object, got "
                            + preTokenizerObj.getClass().getSimpleName());
        }

        Map<String, Object> preTokenizer = (Map<String, Object>) preTokenizerObj;
        String regex = extractRegexFromPreTokenizer(preTokenizer);

        if (regex == null) {
            throw new HuggingFaceTokenizerException(
                    tokenizerJsonPath.toString(),
                    "No explicit regex pattern found in pre_tokenizer. "
                            + "Only explicit Split pretokenizers with Regex patterns are supported.");
        }

        return regex;
    }

    /** Recursively extracts regex from a pretokenizer configuration. */
    @SuppressWarnings("unchecked")
    private String extractRegexFromPreTokenizer(Map<String, Object> preTokenizer) {
        Object type = preTokenizer.get("type");
        if (!(type instanceof String)) {
            return null;
        }

        String typeStr = (String) type;

        // Direct Split with Regex pattern
        if ("Split".equals(typeStr)) {
            return extractRegexFromSplit(preTokenizer);
        }

        // Sequence - look in pretokenizers array
        if ("Sequence".equals(typeStr)) {
            Object pretokenizersObj = preTokenizer.get("pretokenizers");
            if (pretokenizersObj instanceof List) {
                List<Object> pretokenizers = (List<Object>) pretokenizersObj;
                for (Object p : pretokenizers) {
                    if (p instanceof Map) {
                        String regex = extractRegexFromPreTokenizer((Map<String, Object>) p);
                        if (regex != null) {
                            return regex;
                        }
                    }
                }
            }
        }

        return null;
    }

    /** Extracts regex from a Split pretokenizer. */
    @SuppressWarnings("unchecked")
    private String extractRegexFromSplit(Map<String, Object> split) {
        Object patternObj = split.get("pattern");
        if (!(patternObj instanceof Map)) {
            return null;
        }

        Map<String, Object> pattern = (Map<String, Object>) patternObj;
        Object regex = pattern.get("Regex");
        if (regex instanceof String) {
            return (String) regex;
        }

        // Also check lowercase "regex" key (some HF versions)
        regex = pattern.get("regex");
        if (regex instanceof String) {
            return (String) regex;
        }

        return null;
    }

    /** Extracts added tokens from the tokenizer. */
    @SuppressWarnings("unchecked")
    private Map<String, Integer> extractAddedTokens() {
        Object addedTokensObj = root.get("added_tokens");
        if (addedTokensObj == null) {
            return Collections.emptyMap();
        }

        if (!(addedTokensObj instanceof List)) {
            throw new HuggingFaceTokenizerException(
                    tokenizerJsonPath.toString(),
                    "Invalid added_tokens: expected array, got "
                            + addedTokensObj.getClass().getSimpleName());
        }

        List<Object> tokensList = (List<Object>) addedTokensObj;
        Map<String, Integer> addedTokens = new HashMap<>();

        for (int i = 0; i < tokensList.size(); i++) {
            Object tokenObj = tokensList.get(i);
            if (!(tokenObj instanceof Map)) {
                throw new HuggingFaceTokenizerException(
                        tokenizerJsonPath.toString(),
                        "Invalid added token at index " + i + ": expected object");
            }

            Map<String, Object> token = (Map<String, Object>) tokenObj;

            Object id = token.get("id");
            Object content = token.get("content");

            if (!(id instanceof Number)) {
                throw new HuggingFaceTokenizerException(
                        tokenizerJsonPath.toString(),
                        "Invalid added token at index " + i + ": missing or invalid 'id'");
            }

            if (!(content instanceof String)) {
                throw new HuggingFaceTokenizerException(
                        tokenizerJsonPath.toString(),
                        "Invalid added token at index " + i + ": missing or invalid 'content'");
            }

            addedTokens.put((String) content, ((Number) id).intValue());
        }

        return addedTokens;
    }

    /** Holder for parsed tokenizer data. */
    public record TokenizerData(
            Map<String, Integer> vocab,
            List<String> merges,
            String regex,
            Map<String, Integer> addedTokens) {}
}
