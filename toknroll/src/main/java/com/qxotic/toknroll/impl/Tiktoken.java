package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.advanced.ByteEncoding;
import java.io.BufferedReader;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

public class Tiktoken {

    public static Map<String, Integer> loadMergeableRanks(BufferedReader reader) {
        Map<String, Integer> mergeableRanks = new HashMap<>();

        reader.lines()
                .forEachOrdered(
                        line -> {
                            String[] parts = line.split(" ");
                            assert parts.length == 2;
                            byte[] bytes = Base64.getDecoder().decode(parts[0]);
                            String key = ByteEncoding.bytesToString(bytes);

                            int value = Integer.parseInt(parts[1]);
                            assert !mergeableRanks.containsKey(key);

                            mergeableRanks.put(key, value);
                        });

        return mergeableRanks;
    }

    public static Tokenizer createFromTiktoken(
            String name,
            Map<String, Integer> mergeableRanks,
            Pattern splitPattern,
            Map<String, Integer> specialTokens) {
        return JTokkitAdapter.create(name, splitPattern, mergeableRanks, specialTokens);
    }
}
