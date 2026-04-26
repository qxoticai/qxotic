package com.qxotic.toknroll.testkit;

import com.qxotic.format.json.Json;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.ToIntFunction;

/** Shared parity runner for tokenizer implementations (HF, GGUF, etc.). */
public final class TokenizerParityHarness {

    private TokenizerParityHarness() {}

    public static byte[] loadCorpusBytes(Path corpusPath) throws IOException {
        return Files.readAllBytes(corpusPath.toAbsolutePath().normalize());
    }

    public static void runParity(
            String tokenizerLabel,
            Path goldenPath,
            byte[] corpusBytes,
            int chunkSize,
            int maxChunks,
            Function<String, int[]> encode,
            ToIntFunction<String> countTokens,
            Function<int[], String> decode,
            IntFunction<String> tokenSurface) {
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("chunkSize must be > 0");
        }
        if (maxChunks <= 0) {
            throw new IllegalArgumentException("maxChunks must be > 0");
        }

        Map<String, int[]> expectedByHash;
        try {
            expectedByHash = loadGoldenByHash(goldenPath);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load golden file: " + goldenPath, e);
        }

        List<Chunk> chunks = buildChunks(corpusBytes, chunkSize);
        int checked = 0;
        for (int i = 0; i < chunks.size() && checked < maxChunks; i++) {
            Chunk chunk = chunks.get(i);
            int[] expected = expectedByHash.get(chunk.hash);
            if (expected == null) {
                continue;
            }
            checked++;
            assertChunk(tokenizerLabel, chunk, expected, encode, countTokens, decode, tokenSurface);
        }

        if (checked == 0) {
            throw new AssertionError(
                    "No matching chunks between generated corpus chunks and golden file "
                            + goldenPath
                            + " for "
                            + tokenizerLabel
                            + " (chunkSize="
                            + chunkSize
                            + ")");
        }
    }

    @SuppressWarnings("unchecked")
    public static Map<String, int[]> loadGoldenByHash(Path goldenPath) throws IOException {
        String json =
                Files.readString(goldenPath.toAbsolutePath().normalize(), StandardCharsets.UTF_8);
        Object parsed = Json.parse(json);
        if (!(parsed instanceof List<?>)) {
            throw new IOException("Golden file must be a JSON list: " + goldenPath);
        }
        List<Object> rows = (List<Object>) parsed;
        Map<String, int[]> byHash = new HashMap<>(rows.size());
        for (Object rowObj : rows) {
            if (!(rowObj instanceof Map<?, ?>)) {
                throw new IOException("Invalid golden row in " + goldenPath);
            }
            Map<String, Object> row = (Map<String, Object>) rowObj;
            String hash = (String) row.get("chunk_hash");
            if (hash == null || hash.isEmpty()) {
                throw new IOException("Golden row missing chunk_hash in " + goldenPath);
            }
            if (byHash.containsKey(hash)) {
                throw new IOException("Duplicate chunk_hash " + hash + " in " + goldenPath);
            }
            Object tokensObj = row.get("tokens");
            if (!(tokensObj instanceof List<?>)) {
                throw new IOException("Golden row missing tokens for hash " + hash);
            }
            List<Object> tokenList = (List<Object>) tokensObj;
            int[] tokens = new int[tokenList.size()];
            for (int i = 0; i < tokenList.size(); i++) {
                tokens[i] = ((Number) tokenList.get(i)).intValue();
            }
            byHash.put(hash, tokens);
        }
        return byHash;
    }

    public static List<Chunk> buildChunks(byte[] data, int chunkSize) {
        List<Chunk> out = new ArrayList<>();
        if (data.length == 0) {
            return out;
        }
        int offset = 0;
        int index = 0;
        while (offset < data.length) {
            int end = Math.min(offset + chunkSize, data.length);
            while (end < data.length && (data[end] & 0xC0) == 0x80) {
                end++;
            }
            int size = end - offset;
            byte[] chunkBytes = Arrays.copyOfRange(data, offset, end);
            String text = new String(chunkBytes, StandardCharsets.UTF_8);
            out.add(new Chunk(index, offset, size, text, computeChunkHash(offset, size)));
            index++;
            offset = end;
        }
        return out;
    }

    private static void assertChunk(
            String tokenizerLabel,
            Chunk chunk,
            int[] expected,
            Function<String, int[]> encode,
            ToIntFunction<String> countTokens,
            Function<int[], String> decode,
            IntFunction<String> tokenSurface) {
        int[] actual = encode.apply(chunk.text);
        if (!Arrays.equals(expected, actual)) {
            throw new AssertionError(
                    "encode mismatch for "
                            + tokenizerLabel
                            + " chunkHash="
                            + chunk.hash
                            + " index="
                            + chunk.index
                            + " expectedLen="
                            + expected.length
                            + " actualLen="
                            + actual.length
                            + "\n"
                            + describeFirstDiff(expected, actual, tokenSurface));
        }

        int actualCount = countTokens.applyAsInt(chunk.text);
        if (actualCount != expected.length) {
            throw new AssertionError(
                    "countTokens mismatch for "
                            + tokenizerLabel
                            + " chunkHash="
                            + chunk.hash
                            + " index="
                            + chunk.index
                            + " expected="
                            + expected.length
                            + " actual="
                            + actualCount);
        }

        String decoded = decode.apply(expected);
        if (!chunk.text.equals(decoded)) {
            throw new AssertionError(
                    "decode mismatch for "
                            + tokenizerLabel
                            + " chunkHash="
                            + chunk.hash
                            + " index="
                            + chunk.index);
        }
    }

    private static String describeFirstDiff(
            int[] expected, int[] actual, IntFunction<String> tokenSurface) {
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
                + renderWindow(expected, from, to, tokenSurface)
                + "\nactual_window="
                + renderWindow(actual, from, to, tokenSurface);
    }

    private static String renderWindow(
            int[] tokens, int from, int to, IntFunction<String> tokenSurface) {
        StringBuilder b = new StringBuilder();
        for (int i = from; i < to; i++) {
            if (i > from) {
                b.append(", ");
            }
            b.append(i).append(':').append(tokens[i]).append('(');
            try {
                String surface = tokenSurface.apply(tokens[i]);
                b.append(surface == null ? "?" : surface.replace("\n", "\\n").replace("\r", "\\r"));
            } catch (RuntimeException e) {
                b.append('?');
            }
            b.append(')');
        }
        return b.toString();
    }

    private static String computeChunkHash(int offset, int size) {
        byte[] digest;
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            digest = md.digest((offset + ":" + size).getBytes(StandardCharsets.UTF_8));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available", e);
        }
        return hexLower(digest, 8);
    }

    private static String hexLower(byte[] bytes, int byteCount) {
        char[] out = new char[byteCount * 2];
        final char[] hex = "0123456789abcdef".toCharArray();
        for (int i = 0; i < byteCount; i++) {
            int v = bytes[i] & 0xFF;
            out[i * 2] = hex[v >>> 4];
            out[i * 2 + 1] = hex[v & 0x0F];
        }
        return new String(out);
    }

    public static final class Chunk {
        public final int index;
        public final int offset;
        public final int size;
        public final String text;
        public final String hash;

        public Chunk(int index, int offset, int size, String text, String hash) {
            this.index = index;
            this.offset = offset;
            this.size = size;
            this.text = text;
            this.hash = hash;
        }
    }
}
