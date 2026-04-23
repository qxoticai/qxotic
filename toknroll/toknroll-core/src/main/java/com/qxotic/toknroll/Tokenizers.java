package com.qxotic.toknroll;

import com.qxotic.toknroll.impl.ImplAccessor;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class Tokenizers {
    private Tokenizers() {}

    public static Vocabulary vocabulary(String... tokens) {
        return ImplAccessor.createVocabulary(indexVocabulary(tokens));
    }

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

        Map<String, Integer> validatedSpecials = new LinkedHashMap<>(specialTokens.size());
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
            validatedSpecials.put(token, tokenId);
        }

        return validatedSpecials;
    }

    public static TokenizationModel tiktokenModel(Vocabulary vocabulary, List<MergeRule> merges) {
        return ImplAccessor.createTiktokenModel(vocabulary, merges, false);
    }

    public static TokenizationModel tiktokenModel(
            Vocabulary vocabulary, List<MergeRule> merges, boolean ignoreMerges) {
        return ImplAccessor.createTiktokenModel(vocabulary, merges, ignoreMerges);
    }

    public static TokenizationModel sentencePieceBpeModel(
            Vocabulary vocabulary, List<MergeRule> merges) {
        return ImplAccessor.createSentencePieceBpeModel(vocabulary, merges);
    }

    public static TokenizationModel sentencePieceBpeModel(Vocabulary vocabulary, float[] scores) {
        return ImplAccessor.createSentencePieceBpeModel(vocabulary, scores);
    }

    public static TokenizationPipeline.Builder pipeline(TokenizationModel model) {
        return TokenizationPipeline.builder(model);
    }

    public static final class MergeRule {
        private final int leftId;
        private final int rightId;
        private final int rank;

        public MergeRule(int leftId, int rightId, int rank) {
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
