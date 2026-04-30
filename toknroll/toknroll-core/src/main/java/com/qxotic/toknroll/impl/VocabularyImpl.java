package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Vocabulary;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;

/** Internal implementation of {@link Vocabulary} backed by flat arrays. */
class VocabularyImpl implements Vocabulary {
    final String[] tokens;
    final Map<String, Integer> tokenToId;
    final int[] tokenTypes;

    private VocabularyImpl(String[] tokens, Map<String, Integer> tokenToId, int[] tokenTypes) {
        this.tokens = tokens;
        this.tokenToId = Collections.unmodifiableMap(tokenToId);
        this.tokenTypes = tokenTypes;
    }

    public VocabularyImpl(String... tokens) {
        this(tokens, computeTokenMap(tokens), null);
    }

    public VocabularyImpl(Map<String, Integer> tokenToId) {
        this(computeTokens(tokenToId), tokenToId, null);
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

    public VocabularyImpl(String[] vocabulary, int[] tokenTypes) {
        this(vocabulary, computeTokenMap(vocabulary), tokenTypes);
        if (tokenTypes != null && tokenTypes.length != vocabulary.length) {
            throw new IllegalArgumentException(
                    "tokenTypes length "
                            + tokenTypes.length
                            + " != vocabulary length "
                            + vocabulary.length);
        }
    }

    private static Map<String, Integer> computeTokenMap(String[] vocabulary) {
        Map<String, Integer> tokenToId = new HashMap<>(vocabulary.length * 2);
        for (int i = 0; i < vocabulary.length; i++) {
            String token = vocabulary[i];
            if (token != null) {
                tokenToId.put(token, i);
            }
        }
        return tokenToId;
    }

    @Override
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

    int getIdOrNegative(String token) {
        Integer tokenId = this.tokenToId.get(token);
        return tokenId == null ? -1 : tokenId;
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
