package com.qxotic.tokenizers.hf.impl;

import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.hf.HuggingFaceTokenizerException;
import com.qxotic.tokenizers.impl.Tiktoken;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Creates a JTokkit-backed tokenizer from HuggingFace tokenizer data.
 *
 * <p>Builds the tokenizer using the JTokkit adapter for maximum performance. Requires BPE tokenizer
 * with explicit regex pattern.
 */
public class HfJTokkitFactory {

    private final HfTokenizerJsonParser.TokenizerData data;
    private final Map<String, Integer> specialTokens;

    /**
     * Creates a factory with parsed tokenizer data.
     *
     * @param data the parsed tokenizer data
     * @param specialTokens special tokens map
     */
    public HfJTokkitFactory(
            HfTokenizerJsonParser.TokenizerData data, Map<String, Integer> specialTokens) {
        this.data = data;
        this.specialTokens = specialTokens;
    }

    /**
     * Builds and returns a JTokkit-backed tokenizer.
     *
     * @return the tokenizer
     * @throws HuggingFaceTokenizerException if tokenizer cannot be built
     */
    public Tokenizer build() {
        try {
            // Build mergeable ranks map (vocab with merged tokens)
            Map<String, Integer> mergeableRanks = buildMergeableRanks();

            // Compile the regex pattern
            Pattern pattern = Pattern.compile(data.regex());

            // Generate a unique name for this tokenizer
            String name = generateTokenizerName();

            // Filter special tokens to only include JTokkit-compatible ones
            // JTokkit requires special tokens to match pattern: <|...|>
            Map<String, Integer> compatibleSpecialTokens = filterCompatibleSpecialTokens();

            // Create the tokenizer via JTokkit adapter
            return Tiktoken.createFromTiktoken(
                    name, mergeableRanks, pattern, compatibleSpecialTokens);

        } catch (Exception e) {
            throw new HuggingFaceTokenizerException(
                    "Failed to build JTokkit tokenizer: " + e.getMessage(), e);
        }
    }

    /**
     * Filters special tokens to only include ones compatible with JTokkit.
     *
     * <p>JTokkit requires special tokens to follow the pattern: <|...|> Tokens like "</think>" are
     * filtered out but remain in the vocabulary.
     */
    private Map<String, Integer> filterCompatibleSpecialTokens() {
        Map<String, Integer> compatible = new HashMap<>();
        for (Map.Entry<String, Integer> entry : specialTokens.entrySet()) {
            String token = entry.getKey();
            // JTokkit requires tokens to start with "<|" and end with "|>"
            if (token.startsWith("<|") && token.endsWith("|>")) {
                compatible.put(token, entry.getValue());
            }
        }
        return compatible;
    }

    /**
     * Builds the mergeable ranks map from vocabulary and merges.
     *
     * <p>For BPE, the mergeable ranks include both single-byte tokens and merged tokens. The merges
     * define the order in which token pairs are combined.
     */
    private Map<String, Integer> buildMergeableRanks() {
        // Start with the vocabulary
        Map<String, Integer> ranks = new HashMap<>(data.vocab());

        // The merges in HF format are just strings like "Ġthe"
        // We don't need to validate them here - JTokkit will handle the merging logic
        // But we ensure all merge tokens exist in vocab
        validateMerges(ranks);

        return ranks;
    }

    /** Validates that all merge tokens exist in the vocabulary. */
    private void validateMerges(Map<String, Integer> vocab) {
        for (String merge : data.merges()) {
            String[] parts = merge.split(" ");
            if (parts.length != 2) {
                continue; // Already validated in parser
            }

            String left = parts[0];
            String right = parts[1];
            String combined = left + right;

            if (!vocab.containsKey(left)) {
                throw new HuggingFaceTokenizerException(
                        "Merge token '" + left + "' not found in vocabulary");
            }

            if (!vocab.containsKey(right)) {
                throw new HuggingFaceTokenizerException(
                        "Merge token '" + right + "' not found in vocabulary");
            }

            if (!vocab.containsKey(combined)) {
                throw new HuggingFaceTokenizerException(
                        "Merged token '"
                                + combined
                                + "' (from merge '"
                                + merge
                                + "') not found in vocabulary");
            }
        }
    }

    /** Generates a unique name for the tokenizer. */
    private String generateTokenizerName() {
        // Use vocabulary size as part of the name for uniqueness
        return "hf-" + data.vocab().size() + "-" + Math.abs(data.regex().hashCode());
    }
}
