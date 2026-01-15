package ai.qxotic.tokenizers.impl;

import ai.qxotic.tokenizers.Vocabulary;
import ai.qxotic.tokenizers.StandardTokenType;
import ai.qxotic.tokenizers.TokenType;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.stream.Collectors;

class VocabularyWithSpecials implements Vocabulary {
    private final Vocabulary innerVocabulary;
    private final Map<String, Integer> specialToIndex;
    private final Map<Integer, String> indexToSpecial;


    private VocabularyWithSpecials(Vocabulary innerVocabulary, Map<String, Integer> specialToIndex) {
        this.innerVocabulary = innerVocabulary;
        this.specialToIndex = Map.copyOf(specialToIndex);
        this.indexToSpecial = specialToIndex.entrySet().stream()
                .collect(Collectors.toUnmodifiableMap(Map.Entry::getValue, Map.Entry::getKey));

        assert this.specialToIndex.keySet().stream().noneMatch(innerVocabulary::contains);
        assert this.indexToSpecial.keySet().stream().noneMatch(innerVocabulary::contains);
    }

    static Vocabulary create(Vocabulary innerVocabulary, Map<String, Integer> specialToIndex) {
        if (specialToIndex.isEmpty()) {
            return innerVocabulary;
        } else if (innerVocabulary instanceof VocabularyWithSpecials innerWithSpecials) {
            Map<String, Integer> combinedSpecials = new HashMap<>(innerWithSpecials.specialToIndex);
            // No overlap.
            assert specialToIndex.keySet().stream().noneMatch(innerWithSpecials.specialToIndex::containsKey);
            combinedSpecials.putAll(specialToIndex);
            return new VocabularyWithSpecials(innerWithSpecials.innerVocabulary, combinedSpecials);
        } else {
            return new VocabularyWithSpecials(innerVocabulary, specialToIndex);
        }
    }

    @Override
    public int size() {
        return innerVocabulary.size() + specialToIndex.size();
    }

    @Override
    public String token(int tokenIndex) {
        String tokenString = indexToSpecial.get(tokenIndex);
        if (tokenString != null) {
            return tokenString;
        } else {
            return innerVocabulary.token(tokenIndex);
        }
    }

    @Override
    public int id(String tokenString) {
        Integer tokenIndex = specialToIndex.get(tokenString);
        if (tokenIndex != null) {
            return tokenIndex;
        } else {
            return innerVocabulary.id(tokenString);
        }
    }

    @Override
    public boolean contains(int tokenIndex) {
        return indexToSpecial.containsKey(tokenIndex) || innerVocabulary.contains(tokenIndex);
    }

    @Override
    public boolean contains(String tokenString) {
        return specialToIndex.containsKey(tokenString) || innerVocabulary.contains(tokenString);
    }

    @Override
    public boolean isTokenOfType(int tokenIndex, TokenType tokenType) {
        if (indexToSpecial.containsKey(tokenIndex)) {
            return tokenType == StandardTokenType.CONTROL;
        }
        return innerVocabulary.isTokenOfType(tokenIndex, tokenType);
    }

    @Override
    public Iterator<Map.Entry<String, Integer>> iterator() {
        return IteratorCombiner.of(innerVocabulary.iterator(), specialToIndex.entrySet().iterator());
    }
}
