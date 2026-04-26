package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.json.Json;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("slow")
@Tag("local-external")
class HuggingFaceSwissApertusFullEnwik8RegressionTest {

    private static final Path CORE_GOLDEN_ENWIK8_DIR =
            Path.of("..", "toknroll-core", "src", "test", "resources", "golden", "enwik8");

    @Test
    @SuppressWarnings("unchecked")
    void swissApertusParityOnFullCorpusChunk() throws IOException {
        Path chunksPath =
                CORE_GOLDEN_ENWIK8_DIR.resolve("chunks.json").toAbsolutePath().normalize();
        Path groundTruthPath =
                CORE_GOLDEN_ENWIK8_DIR
                        .resolve("hf_swiss_apertus8b_ground_truth.json")
                        .toAbsolutePath()
                        .normalize();

        assertTrue(Files.exists(chunksPath), "Missing chunks fixture: " + chunksPath);
        assertTrue(Files.exists(groundTruthPath), "Missing swiss fixture: " + groundTruthPath);

        List<Object> chunkRows =
                (List<Object>) Json.parse(Files.readString(chunksPath, StandardCharsets.UTF_8));
        Assumptions.assumeTrue(!chunkRows.isEmpty(), "Missing enwik8 chunks");

        Map<String, Object> firstChunk = (Map<String, Object>) chunkRows.get(0);
        int size = ((Number) firstChunk.get("size")).intValue();
        Assumptions.assumeTrue(
                size == 100_000_000,
                () ->
                        "Full-corpus regression requires --chunk-sizes=100000000 fixture; found"
                                + " size="
                                + size);

        String text = (String) firstChunk.get("text");
        List<Object> gtRows =
                (List<Object>)
                        Json.parse(Files.readString(groundTruthPath, StandardCharsets.UTF_8));
        Assumptions.assumeTrue(!gtRows.isEmpty(), "Missing swiss ground-truth rows");

        Map<String, Object> firstGt = (Map<String, Object>) gtRows.get(0);
        List<Object> tokenList = (List<Object>) firstGt.get("tokens");
        int[] expected = new int[tokenList.size()];
        for (int i = 0; i < tokenList.size(); i++) {
            expected[i] = ((Number) tokenList.get(i)).intValue();
        }

        Tokenizer tokenizer =
                HuggingFaceTokenizerLoader.fromHuggingFace(
                        "swiss-ai", "Apertus-8B-Instruct-2509", "main", false, false);
        int[] actual = tokenizer.encodeToArray(text);

        assertEquals(
                expected.length,
                actual.length,
                () ->
                        "Full-corpus swiss/apertus drift: likely regex pre-tokenizer semantic"
                                + " mismatch\n"
                                + "length expected="
                                + expected.length
                                + " actual="
                                + actual.length
                                + "\n"
                                + describeFirstDiff(tokenizer, expected, actual));

        for (int i = 0; i < expected.length; i++) {
            if (expected[i] != actual[i]) {
                throw new AssertionError(describeFirstDiff(tokenizer, expected, actual));
            }
        }
    }

    private static String describeFirstDiff(Tokenizer tokenizer, int[] expected, int[] actual) {
        int min = Math.min(expected.length, actual.length);
        int first = -1;
        for (int i = 0; i < min; i++) {
            if (expected[i] != actual[i]) {
                first = i;
                break;
            }
        }
        if (first < 0) {
            return "No differing prefix token id; only lengths differ.";
        }

        int from = Math.max(0, first - 8);
        int to = Math.min(min, first + 8);
        return "first_diff_index="
                + first
                + "\nexpected_window="
                + renderWindow(tokenizer, expected, from, to)
                + "\nactual_window="
                + renderWindow(tokenizer, actual, from, to);
    }

    private static String renderWindow(Tokenizer tokenizer, int[] tokens, int from, int to) {
        return java.util.stream.IntStream.range(from, to)
                .mapToObj(i -> i + ":" + tokens[i] + "(" + safeToken(tokenizer, tokens[i]) + ")")
                .collect(Collectors.joining(", "));
    }

    private static String safeToken(Tokenizer tokenizer, int tokenId) {
        try {
            return tokenizer.vocabulary().token(tokenId).replace("\n", "\\n").replace("\r", "\\r");
        } catch (RuntimeException e) {
            return "?";
        }
    }
}
