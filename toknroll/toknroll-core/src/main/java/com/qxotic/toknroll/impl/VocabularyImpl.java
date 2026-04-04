package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.advanced.StandardTokenType;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class VocabularyImpl implements Vocabulary {
    protected final String[] tokens;
    protected final float[] scores;
    protected final Map<String, Integer> tokenToId;
    protected final int[] tokenTypes;

    private VocabularyImpl(
            String[] tokens, float[] scores, Map<String, Integer> tokenToId, int[] tokenTypes) {
        this.tokens = tokens;
        this.scores = scores;
        this.tokenToId = Collections.unmodifiableMap(tokenToId);
        this.tokenTypes = tokenTypes;
    }

    public VocabularyImpl(String... tokens) {
        this(tokens, null, null);
    }

    public VocabularyImpl(Map<String, Integer> tokenToId) {
        this(computeTokens(tokenToId), null, tokenToId, null);
    }

    private static String[] computeTokens(Map<String, Integer> tokenToId) {
        int maxId = -1;
        for (Map.Entry<String, Integer> entry : tokenToId.entrySet()) {
            Integer id = entry.getValue();
            if (id == null) {
                throw new IllegalArgumentException(
                        "token id cannot be null for token: " + entry.getKey());
            }
            if (id < 0) {
                throw new IllegalArgumentException(
                        "token id must be non-negative for token: "
                                + entry.getKey()
                                + " ("
                                + id
                                + ")");
            }
            if (id > maxId) {
                maxId = id;
            }
        }

        String[] tokens = new String[Math.max(0, maxId + 1)];
        for (Map.Entry<String, Integer> entry : tokenToId.entrySet()) {
            int id = entry.getValue();
            String token = entry.getKey();
            String previous = tokens[id];
            if (previous != null && !previous.equals(token)) {
                throw new IllegalArgumentException(
                        "Duplicate token id "
                                + id
                                + " for tokens '"
                                + previous
                                + "' and '"
                                + token
                                + "'");
            }
            tokens[id] = token;
        }
        return tokens;
    }

    public VocabularyImpl(String[] vocabulary, float[] scores, int[] tokenTypes) {
        this(vocabulary, scores, computeTokenMap(vocabulary), tokenTypes);
        assert scores == null || scores.length == vocabulary.length;
        assert tokenTypes == null || tokenTypes.length == vocabulary.length;
    }

    private static Map<String, Integer> computeTokenMap(String[] vocabulary) {
        return IntStream.range(0, vocabulary.length)
                .filter(i -> vocabulary[i] != null)
                .boxed()
                .collect(Collectors.toUnmodifiableMap(i -> vocabulary[i], i -> i));
    }

    public String token(int tokenId) {
        if (Integer.compareUnsigned(tokenId, this.tokens.length) >= 0) {
            throw new NoSuchElementException(String.valueOf(tokenId));
        }
        String token = tokens[tokenId];
        if (token == null) {
            throw new NoSuchElementException(String.valueOf(tokenId));
        }
        return token;
    }

    @Override
    public int id(String token) {
        Integer tokenId = this.tokenToId.get(token);
        if (tokenId == null) {
            throw new NoSuchElementException(token);
        }
        return tokenId;
    }

    @Override
    public boolean contains(int tokenId) {
        return 0 <= tokenId && tokenId < size() && tokens[tokenId] != null;
    }

    @Override
    public boolean contains(String token) {
        return tokenToId.containsKey(token);
    }

    @Override
    public int size() {
        return tokens.length;
    }

    @Override
    public boolean isTokenOfType(int tokenId, TokenType tokenType) {
        token(tokenId);
        if (tokenTypes != null) {
            if (tokenType instanceof StandardTokenType) {
                return this.tokenTypes[tokenId] == ((StandardTokenType) tokenType).getId();
            }
        }
        return Vocabulary.super.isTokenOfType(tokenId, tokenType);
    }

    @Override
    public Iterator<Map.Entry<String, Integer>> iterator() {
        return tokenToId.entrySet().iterator();
    }
}
