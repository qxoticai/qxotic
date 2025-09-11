package com.llm4j.tokenizers.impl;

import com.llm4j.tokenizers.Tokenizer;
import com.llm4j.tokenizers.ByteEncoding;

import java.io.BufferedReader;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

public class Tiktoken {

    public static Map<String, Integer> loadMergeableRanks(BufferedReader reader) {
        Map<String, Integer> mergeableRanks = new HashMap<>();

        reader.lines()
                .forEachOrdered(line -> {
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

    public static Tokenizer createFromTiktoken(String name, Map<String, Integer> mergeableRanks, Pattern splitPattern, Map<String, Integer> specialTokens) {
        return JTokkitAdapter.create(name, splitPattern, mergeableRanks, specialTokens);
    }

//    public static Map<String, Integer> loadFromJSON(String json) {
//        Map<String, Integer> mergeableRanks = new HashMap<>();
//
//        Map<String, ?> parse = (Map<String, ?>) JSON.parse(json);
//        List<Map<String, ?>> vocab = (List<Map<String, ?>>) parse.get("vocab");
//        for (Map<String, ?> entry : vocab) {
//            int rank = ((Number) entry.get("rank")).intValue();
//            String base64TokenBytes = (String) entry.get("token_bytes");
//            byte[] tokenBytes = Base64.getDecoder().decode(base64TokenBytes);
//            String token = ByteEncoding.bytesToString(tokenBytes);
//            mergeableRanks.put(token, rank);
//        }
//
//        return mergeableRanks;
//    }
}
