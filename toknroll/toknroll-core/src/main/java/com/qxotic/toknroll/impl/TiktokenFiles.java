package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.ByteLevel;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

/** Utilities for loading .tiktoken mergeable ranks files. */
public final class TiktokenFiles {

    private TiktokenFiles() {}

    public static Map<String, Integer> loadMergeableRanks(String blobPath, String expectedHash)
            throws IOException, InterruptedException {
        byte[] bytes = FileCache.readFileCached(blobPath, expectedHash);
        return loadMergeableRanks(
                new BufferedReader(new InputStreamReader(new ByteArrayInputStream(bytes))));
    }

    public static Map<String, Integer> loadMergeableRanks(BufferedReader reader) {
        Map<String, Integer> mergeableRanks = new HashMap<>();
        reader.lines()
                .forEachOrdered(
                        line -> {
                            String[] parts = line.split(" ");
                            byte[] bytes = Base64.getDecoder().decode(parts[0]);
                            String key = ByteLevel.encode(bytes);
                            int value = Integer.parseInt(parts[1]);
                            mergeableRanks.put(key, value);
                        });
        return mergeableRanks;
    }
}
