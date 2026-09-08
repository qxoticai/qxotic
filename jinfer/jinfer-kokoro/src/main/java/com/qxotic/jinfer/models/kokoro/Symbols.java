package com.qxotic.jinfer.models.kokoro;

import java.util.HashMap;
import java.util.Map;

/** The sparse Unicode phoneme vocabulary embedded in a Kokoro GGUF. */
public final class Symbols {

    private final Map<Integer, Integer> ids;

    Symbols(String[] tokens) {
        Map<Integer, Integer> indexed = HashMap.newHashMap(tokens.length);
        for (int id = 0; id < tokens.length; id++) {
            String token = tokens[id];
            if (token.isEmpty()) continue;
            if (token.codePointCount(0, token.length()) != 1)
                throw new IllegalArgumentException(
                        "Kokoro: token " + id + " is not one Unicode code point: '" + token + "'");
            indexed.putIfAbsent(token.codePointAt(0), id);
        }
        ids = Map.copyOf(indexed);
    }

    /** Maps phonemes to model IDs, dropping code points absent from the sparse vocabulary. */
    public int[] toRaw(String phonemes) {
        return phonemes.codePoints()
                .map(cp -> ids.getOrDefault(cp, -1))
                .filter(id -> id >= 0)
                .toArray();
    }
}
