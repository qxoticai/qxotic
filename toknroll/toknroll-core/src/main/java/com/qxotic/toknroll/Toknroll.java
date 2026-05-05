package com.qxotic.toknroll;

import com.qxotic.toknroll.impl.ImplAccessor;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Static factory for building {@link Tokenizer} pipelines and {@link TokenizationModel} instances.
 *
 * <p>This class assembles the building blocks of a tokenizer: {@link Vocabulary}, {@link
 * TokenizationModel} (tiktoken or SentencePiece BPE), optional {@link Normalizer}, and optional
 * {@link Splitter}. Callers chain them into a {@link TokenizationPipeline}, which implements {@link
 * Tokenizer}.
 */
public final class Toknroll {
    private Toknroll() {}

    /**
     * Creates a vocabulary where token IDs are assigned by array position ({@code tokens[0]} = 0,
     * {@code tokens[1]} = 1, etc.).
     *
     * @param tokens token strings indexed by their ID
     * @return vocabulary with positional ID assignment
     * @throws NullPointerException if any token is null
     * @throws IllegalArgumentException if duplicate tokens are found
     */
    public static Vocabulary vocabulary(String... tokens) {
        return ImplAccessor.createVocabulary(indexVocabulary(tokens));
    }

    /**
     * Creates a vocabulary with special tokens merged in. Special token IDs must not overlap with
     * the positional token IDs.
     *
     * @param specialTokens map from special token strings to their IDs
     * @param tokens base token strings indexed by their positional ID
     * @return vocabulary with special tokens included
     * @throws NullPointerException if any argument is null
     * @throws IllegalArgumentException if special token IDs overlap with positional IDs
     */
    public static Vocabulary vocabulary(Map<String, Integer> specialTokens, String... tokens) {
        Map<String, Integer> tokenToId = indexVocabulary(tokens);
        return ImplAccessor.createVocabularyWithSpecials(
                tokenToId, validateSpecialTokens(specialTokens, tokenToId));
    }

    private static Map<String, Integer> indexVocabulary(String[] tokens) {
        Objects.requireNonNull(tokens, "tokens");
        Map<String, Integer> tokenToId = new LinkedHashMap<>(tokens.length);
        for (int i = 0; i < tokens.length; i++) {
            String token = Objects.requireNonNull(tokens[i], "tokens[" + i + "]");
            Integer previousId = tokenToId.putIfAbsent(token, i);
            if (previousId != null) {
                throw new IllegalArgumentException(
                        "Duplicate token '" + token + "' at indexes " + previousId + " and " + i);
            }
        }
        return tokenToId;
    }

    private static Map<String, Integer> validateSpecialTokens(
            Map<String, Integer> specialTokens, Map<String, Integer> tokenToId) {
        Objects.requireNonNull(specialTokens, "specialTokens");
        if (specialTokens.isEmpty()) {
            return specialTokens;
        }

        Set<Integer> seenIds = new HashSet<>();
        for (Map.Entry<String, Integer> entry : specialTokens.entrySet()) {
            String token =
                    Objects.requireNonNull(entry.getKey(), "specialTokens contains null key");
            Integer tokenId =
                    Objects.requireNonNull(
                            entry.getValue(),
                            "specialTokens contains null id for token '" + token + "'");
            if (tokenId < 0) {
                throw new IllegalArgumentException(
                        "special token id must be non-negative for token '"
                                + token
                                + "' ("
                                + tokenId
                                + ")");
            }
            if (tokenToId.containsKey(token)) {
                throw new IllegalArgumentException(
                        "special token overlaps with vocabulary token: '" + token + "'");
            }
            if (tokenToId.containsValue(tokenId)) {
                throw new IllegalArgumentException(
                        "special token id overlaps with vocabulary id: " + tokenId);
            }
            if (!seenIds.add(tokenId)) {
                throw new IllegalArgumentException("Duplicate special token id: " + tokenId);
            }
        }

        return specialTokens;
    }

    /**
     * Creates a tiktoken/GPT-2 style BPE model.
     *
     * <p><strong>Important:</strong> every token surface in {@code vocabulary} must already be
     * byte-level encoded (see {@link ByteLevel#encode(byte[])}). The model validates this invariant
     * at construction and throws if any token is not a valid byte-level symbol string (see {@link
     * ByteLevel#isValidEncoding(CharSequence)}).
     *
     * @param vocabulary vocabulary with byte-level encoded token surfaces
     * @param merges ranked merge rules
     * @return tiktoken-compatible BPE model
     * @throws IllegalArgumentException if {@code vocabulary} contains non-byte-level token strings
     */
    public static TokenizationModel tiktokenModel(Vocabulary vocabulary, List<MergeRule> merges) {
        return ImplAccessor.createTiktokenModel(vocabulary, merges, false);
    }

    /**
     * Creates a SentencePiece BPE model from ranked merges. Unlike tiktoken models, tokens are raw
     * UTF-8 strings (not byte-level encoded).
     *
     * @param vocabulary vocabulary with raw UTF-8 token surfaces
     * @param merges ranked merge rules
     * @return SentencePiece BPE model
     */
    public static TokenizationModel sentencePieceBpeModel(
            Vocabulary vocabulary, List<MergeRule> merges) {
        return ImplAccessor.createSentencePieceBpeModel(vocabulary, merges);
    }

    /**
     * Creates a SentencePiece BPE model from token scores. Scores are converted to ranks by sorting
     * descending (higher score = lower rank = higher merge priority).
     *
     * @param vocabulary vocabulary with raw UTF-8 token surfaces
     * @param scores token scores, one per vocabulary entry
     * @return SentencePiece BPE model
     */
    public static TokenizationModel sentencePieceBpeModel(Vocabulary vocabulary, float[] scores) {
        return ImplAccessor.createSentencePieceBpeModel(vocabulary, scores);
    }

    /**
     * Splits input before feeding chunks to the model. No normalization applied.
     *
     * @param splitter text splitter
     * @param model tokenization model
     * @return pipeline that splits then encodes
     */
    public static TokenizationPipeline pipeline(Splitter splitter, TokenizationModel model) {
        return new TokenizationPipeline(null, splitter, model);
    }

    /**
     * Normalizes input before feeding it directly to the model. No splitting applied.
     *
     * @param normalizer text normalizer
     * @param model tokenization model
     * @return pipeline that normalizes then encodes
     */
    public static TokenizationPipeline pipeline(Normalizer normalizer, TokenizationModel model) {
        return new TokenizationPipeline(normalizer, null, model);
    }

    /**
     * Normalizes input, then splits, then feeds chunks to the model.
     *
     * @param normalizer text normalizer (may be {@code null} for identity)
     * @param splitter text splitter (may be {@code null} for identity)
     * @param model tokenization model
     * @return fully assembled pipeline
     */
    public static TokenizationPipeline pipeline(
            Normalizer normalizer, Splitter splitter, TokenizationModel model) {
        return new TokenizationPipeline(normalizer, splitter, model);
    }
}
