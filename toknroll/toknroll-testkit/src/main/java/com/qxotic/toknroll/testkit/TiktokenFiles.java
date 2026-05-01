package com.qxotic.toknroll.testkit;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

/** Utilities for loading .tiktoken mergeable ranks files. */
public final class TiktokenFiles {

    private static final char[] BYTE_SYMBOLS = {
        '\u0100', '\u0101', '\u0102', '\u0103', '\u0104', '\u0105', '\u0106', '\u0107', '\u0108',
                '\u0109', '\u010a', '\u010b',
        '\u010c', '\u010d', '\u010e', '\u010f', '\u0110', '\u0111', '\u0112', '\u0113', '\u0114',
                '\u0115', '\u0116', '\u0117',
        '\u0118', '\u0119', '\u011a', '\u011b', '\u011c', '\u011d', '\u011e', '\u011f', '\u0120',
                '!', '"', '#',
        '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
        '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
        'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
        '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
        'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
        'x', 'y', 'z', '{', '|', '}', '~', '\u0121', '\u0122', '\u0123', '\u0124', '\u0125',
        '\u0126', '\u0127', '\u0128', '\u0129', '\u012a', '\u012b', '\u012c', '\u012d', '\u012e',
                '\u012f', '\u0130', '\u0131',
        '\u0132', '\u0133', '\u0134', '\u0135', '\u0136', '\u0137', '\u0138', '\u0139', '\u013a',
                '\u013b', '\u013c', '\u013d',
        '\u013e', '\u013f', '\u0140', '\u0141', '\u0142', '\u00a1', '\u00a2', '\u00a3', '\u00a4',
                '\u00a5', '\u00a6', '\u00a7',
        '\u00a8', '\u00a9', '\u00aa', '\u00ab', '\u00ac', '\u0143', '\u00ae', '\u00af', '\u00b0',
                '\u00b1', '\u00b2', '\u00b3',
        '\u00b4', '\u00b5', '\u00b6', '\u00b7', '\u00b8', '\u00b9', '\u00ba', '\u00bb', '\u00bc',
                '\u00bd', '\u00be', '\u00bf',
        '\u00c0', '\u00c1', '\u00c2', '\u00c3', '\u00c4', '\u00c5', '\u00c6', '\u00c7', '\u00c8',
                '\u00c9', '\u00ca', '\u00cb',
        '\u00cc', '\u00cd', '\u00ce', '\u00cf', '\u00d0', '\u00d1', '\u00d2', '\u00d3', '\u00d4',
                '\u00d5', '\u00d6', '\u00d7',
        '\u00d8', '\u00d9', '\u00da', '\u00db', '\u00dc', '\u00dd', '\u00de', '\u00df', '\u00e0',
                '\u00e1', '\u00e2', '\u00e3',
        '\u00e4', '\u00e5', '\u00e6', '\u00e7', '\u00e8', '\u00e9', '\u00ea', '\u00eb', '\u00ec',
                '\u00ed', '\u00ee', '\u00ef',
        '\u00f0', '\u00f1', '\u00f2', '\u00f3', '\u00f4', '\u00f5', '\u00f6', '\u00f7', '\u00f8',
                '\u00f9', '\u00fa', '\u00fb',
        '\u00fc', '\u00fd', '\u00fe', '\u00ff'
    };

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
                            String key = encodeBytes(bytes);
                            int value = Integer.parseInt(parts[1]);
                            mergeableRanks.put(key, value);
                        });
        return mergeableRanks;
    }

    private static String encodeBytes(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length);
        for (byte b : bytes) {
            sb.append(BYTE_SYMBOLS[b & 0xFF]);
        }
        return sb.toString();
    }
}
