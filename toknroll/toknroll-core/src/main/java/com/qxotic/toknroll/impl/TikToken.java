package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.ByteLevel;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Tokenizers;
import java.io.BufferedReader;
import java.util.Base64;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * @deprecated Internal compatibility shim; use {@code com.qxotic.toknroll.Tokenizers}.
 */
@Deprecated(forRemoval = false, since = "0.1.0")
public class TikToken {

    private TikToken() {}

    public static Map<String, Integer> loadMergeableRanks(BufferedReader reader) {
        Map<String, Integer> mergeableRanks = new HashMap<>();

        reader.lines()
                .forEachOrdered(
                        line -> {
                            String[] parts = line.split(" ");
                            assert parts.length == 2;
                            byte[] bytes = Base64.getDecoder().decode(parts[0]);
                            String key = ByteLevel.encode(bytes);

                            int value = Integer.parseInt(parts[1]);
                            assert !mergeableRanks.containsKey(key);

                            mergeableRanks.put(key, value);
                        });

        return mergeableRanks;
    }

    /** Backward-compatible factory for examples and older call sites. */
    public static Tokenizer createFromTiktoken(
            String name,
            Map<String, Integer> mergeableRanks,
            Pattern splitPattern,
            Map<String, Integer> specialTokens) {
        return Tokenizers.tikToken(
                mergeableRanks,
                specialTokens == null ? Collections.emptyMap() : specialTokens,
                Splitter.regex(splitPattern));
    }
}
