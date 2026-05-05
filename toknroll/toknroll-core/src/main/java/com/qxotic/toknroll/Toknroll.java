package com.qxotic.toknroll;

import com.qxotic.toknroll.impl.ImplAccessor;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class Toknroll {
    private Toknroll() {}

    /**
     * Creates a vocabulary where token IDs are assigned by array position ({@code tokens[0]} = 0,
     * {@code tokens[1]} = 1, etc.).
     */
    public static Vocabulary vocabulary(String... tokens) {
        return ImplAccessor.createVocabulary(indexVocabulary(tokens));
    }

    /**
     * Creates a vocabulary with special tokens merged in. Special token IDs must not overlap with
     * the positional token IDs.
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
     * @throws IllegalArgumentException if {@code vocabulary} contains non-byte-level token strings
     */
    public static TokenizationModel tiktokenModel(Vocabulary vocabulary, List<MergeRule> merges) {
        return ImplAccessor.createTiktokenModel(vocabulary, merges, false);
    }

    /**
     * Creates a SentencePiece BPE model from ranked merges. Unlike tiktoken models, tokens are raw
     * UTF-8 strings (not byte-level encoded).
     */
    public static TokenizationModel sentencePieceBpeModel(
            Vocabulary vocabulary, List<MergeRule> merges) {
        return ImplAccessor.createSentencePieceBpeModel(vocabulary, merges);
    }

    /**
     * Creates a SentencePiece BPE model from token scores. Scores are converted to ranks by sorting
     * descending (higher score = lower rank = higher merge priority).
     */
    public static TokenizationModel sentencePieceBpeModel(Vocabulary vocabulary, float[] scores) {
        return ImplAccessor.createSentencePieceBpeModel(vocabulary, scores);
    }

    /** Splits input before feeding chunks to the model. No normalization applied. */
    public static TokenizationPipeline pipeline(Splitter splitter, TokenizationModel model) {
        return new TokenizationPipeline(null, splitter, model);
    }

    /** Normalizes input before feeding it directly to the model. No splitting applied. */
    public static TokenizationPipeline pipeline(Normalizer normalizer, TokenizationModel model) {
        return new TokenizationPipeline(normalizer, null, model);
    }

    /** Normalizes input, then splits, then feeds chunks to the model. */
    public static TokenizationPipeline pipeline(
            Normalizer normalizer, Splitter splitter, TokenizationModel model) {
        return new TokenizationPipeline(normalizer, splitter, model);
    }

    /** A BPE merge rule: merge token {@code leftId} with {@code rightId} at the given rank. */
    public static final class MergeRule {
        private final int leftId;
        private final int rightId;
        private final int rank;

        public static MergeRule of(int leftId, int rightId, int rank) {
            return new MergeRule(leftId, rightId, rank);
        }

        MergeRule(int leftId, int rightId, int rank) {
            if (leftId < 0) {
                throw new IllegalArgumentException("leftId must be non-negative: " + leftId);
            }
            if (rightId < 0) {
                throw new IllegalArgumentException("rightId must be non-negative: " + rightId);
            }
            this.leftId = leftId;
            this.rightId = rightId;
            this.rank = rank;
        }

        public int leftId() {
            return leftId;
        }

        public int rightId() {
            return rightId;
        }

        public int rank() {
            return rank;
        }
    }
}
