package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Vocabulary;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Test;

class SentencePieceBpeModelTest {

    /** Bytes, letters, and pairs and runs of them that merge: words of one id and of many. */
    private static SentencePieceBpeModel lettersModel() {
        List<String> tokens = new ArrayList<>();
        for (int b = 0; b < 256; b++) {
            tokens.add(String.format("<0x%02X>", b));
        }
        String[] merges = {
            "a", "b", "c", "d", "\u2581", "ab", "cd", "abcd", "\u2581a", "ba", "abab"
        };
        Collections.addAll(tokens, merges);
        float[] scores = new float[tokens.size()];
        int[] types = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            types[i] = (i < 256 ? StandardTokenType.BYTE : StandardTokenType.NORMAL).getId();
            scores[i] = i < 256 ? 0f : i - 256;
        }
        return SentencePieceBpeModel.fromVocabulary(
                new VocabularyImpl(tokens.toArray(new String[0]), types), scores);
    }

    /**
     * The word cache is shared by every thread without a lock: words of every size (past the cached
     * chars and ids too), on many threads at once, in slots they share, encode as a model of their
     * own encodes them, and a word given as a range of a longer text as the word alone.
     */
    @Test
    void wordsEncodeAlikeOnManyThreadsSharingTheCache() throws Exception {
        String[] parts = {"a", "b", "c", "d", "\u2581", "x", "\u00e9", "\uD83D\uDE00"};
        Random random = new Random(7);
        String[] words = new String[5000];
        for (int i = 0; i < words.length; i++) {
            StringBuilder w = new StringBuilder();
            for (int n = 1 + random.nextInt(random.nextInt(8) == 0 ? 30 : 8); n > 0; n--) {
                w.append(parts[random.nextInt(parts.length)]);
            }
            words[i] = w.toString();
        }
        SentencePieceBpeModel reference = lettersModel();
        int[][] want = new int[words.length][];
        for (int i = 0; i < words.length; i++) {
            want[i] = lettersModel().encode(words[i]).toArray(); // a fresh cache each time
            assertArrayEquals(want[i], reference.encode(words[i]).toArray(), words[i]);
        }
        SentencePieceBpeModel shared = lettersModel();
        ExecutorService pool = Executors.newFixedThreadPool(16);
        try {
            List<Future<?>> runs = new ArrayList<>();
            for (int t = 0; t < 16; t++) {
                int seed = t;
                runs.add(
                        pool.submit(
                                () -> {
                                    Random r = new Random(seed);
                                    for (int k = 0; k < 40_000; k++) {
                                        int i = r.nextInt(words.length);
                                        String text = "zz" + words[i] + "z";
                                        IntSequence.Builder out = IntSequence.newBuilder();
                                        shared.encodeInto(text, 2, 2 + words[i].length(), out);
                                        assertArrayEquals(want[i], out.build().toArray(), words[i]);
                                    }
                                    return null;
                                }));
            }
            for (Future<?> run : runs) {
                run.get();
            }
        } finally {
            pool.shutdown();
        }
    }

    @Test
    void encodeGreedilyMergesHighestScorePairs() {
        String[] tokens = {"<0x00>", " ", "h", "e", "l", "o", "he", "ll", "llo", "hello", " hello"};
        float[] scores = {0f, 0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f, 5f};
        int[] tokenTypes = new int[tokens.length];
        for (int i = 0; i < tokenTypes.length; i++) {
            tokenTypes[i] = StandardTokenType.NORMAL.getId();
        }
        tokenTypes[0] = StandardTokenType.BYTE.getId();

        Vocabulary vocabulary = new VocabularyImpl(tokens, tokenTypes);
        SentencePieceBpeModel model = SentencePieceBpeModel.fromVocabulary(vocabulary, scores);

        IntSequence encoded = model.encode(" hello");

        assertArrayEquals(new int[] {10}, encoded.toArray());
        assertEquals(" hello", model.decode(encoded));
    }

    @Test
    void encodeMergesMultipleSpacesIntoMetaspaceRun() {
        String[] tokens = {"<0x00>", " ", "  ", "a"};
        float[] scores = {0f, 0f, 1f, 0f};
        int[] tokenTypes = {
            StandardTokenType.BYTE.getId(),
            StandardTokenType.NORMAL.getId(),
            StandardTokenType.NORMAL.getId(),
            StandardTokenType.NORMAL.getId()
        };

        Vocabulary vocabulary = new VocabularyImpl(tokens, tokenTypes);
        SentencePieceBpeModel model = SentencePieceBpeModel.fromVocabulary(vocabulary, scores);

        IntSequence encoded = model.encode("  ");

        assertArrayEquals(new int[] {2}, encoded.toArray());
        assertEquals("  ", model.decode(encoded));
    }

    @Test
    void byteFallbackUsesExplicitByteTokenIds() {
        String[] tokens = new String[900];
        float[] scores = new float[tokens.length];
        int[] tokenTypes = new int[tokens.length];
        for (int i = 0; i < tokenTypes.length; i++) {
            tokenTypes[i] = StandardTokenType.NORMAL.getId();
        }

        for (int b = 0; b < 256; b++) {
            int id = 100 + (b * 3);
            tokens[id] = String.format("<0x%02X>", b);
            tokenTypes[id] = StandardTokenType.BYTE.getId();
        }

        Vocabulary vocabulary = new VocabularyImpl(tokens, tokenTypes);
        SentencePieceBpeModel model = SentencePieceBpeModel.fromVocabulary(vocabulary, scores);

        IntSequence encoded = model.encode("é");

        assertArrayEquals(new int[] {685, 607}, encoded.toArray());
        assertEquals("é", model.decode(encoded));
    }

    @Test
    void byteFallbackMissingByteTokenThrowsClearError() {
        String[] tokens = {"<0x00>", " ", "x"};
        float[] scores = {0f, 0f, 0f};
        int[] tokenTypes = {
            StandardTokenType.BYTE.getId(),
            StandardTokenType.NORMAL.getId(),
            StandardTokenType.NORMAL.getId()
        };

        Vocabulary vocabulary = new VocabularyImpl(tokens, tokenTypes);
        SentencePieceBpeModel model = SentencePieceBpeModel.fromVocabulary(vocabulary, scores);

        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> model.encode("é"));
        assertTrue(ex.getMessage().contains("<0xC3>"));
    }
}
