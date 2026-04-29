package com.qxotic.toknroll.testkit;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.ToIntFunction;

/** Shared smoke checks for tokenizer invariants on small input strings. */
public final class TokenizerInvariantHarness {

    @FunctionalInterface
    public interface EncodeIntoFn {
        int[] encodeInto(String text);
    }

    @FunctionalInterface
    public interface DecodeBytesIntoFn {
        int decodeBytesInto(int[] tokens, int tokenStartIndex, ByteBuffer out);
    }

    @FunctionalInterface
    public interface EncodeSliceFn {
        int[] encodeSlice(CharSequence text, int start, int end);
    }

    @FunctionalInterface
    public interface CountTokensSliceFn {
        int countTokensSlice(CharSequence text, int start, int end);
    }

    /** A deliberately small buffer to force partial writes in {@link #decodeBytesInto}. */
    private static final int DECODE_CHUNK_SIZE = 17;

    public static void runSmokeChecks(
            String tokenizerLabel,
            List<String> inputs,
            Function<String, int[]> encode,
            EncodeIntoFn encodeInto,
            ToIntFunction<String> countTokens,
            Function<int[], String> decode,
            Function<int[], byte[]> decodeBytes,
            ToIntFunction<int[]> countBytes,
            DecodeBytesIntoFn decodeBytesInto) {
        Objects.requireNonNull(tokenizerLabel, "tokenizerLabel");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(encode, "encode");
        Objects.requireNonNull(encodeInto, "encodeInto");
        Objects.requireNonNull(countTokens, "countTokens");
        Objects.requireNonNull(decode, "decode");
        Objects.requireNonNull(decodeBytes, "decodeBytes");
        Objects.requireNonNull(countBytes, "countBytes");
        Objects.requireNonNull(decodeBytesInto, "decodeBytesInto");

        for (String text : inputs) {
            // Skip empty input: some tokenizers (e.g. SPM with metaspace prepend) inject a
            // synthetic boundary token even for "", which is correct behavior but breaks the
            // round-trip-with-text-equality contract.
            if (text.isEmpty()) {
                continue;
            }

            int[] encoded = encode.apply(text);
            int[] encodedAgain = encode.apply(text);
            assertIntArrayEquals(tokenizerLabel, text, "encode determinism", encoded, encodedAgain);

            int[] encodedInto = encodeInto.encodeInto(text);
            assertIntArrayEquals(
                    tokenizerLabel, text, "encodeInto(full) parity", encoded, encodedInto);

            int count = countTokens.applyAsInt(text);
            if (count != encoded.length) {
                throw new AssertionError(
                        tokenizerLabel
                                + " invariant failed [countTokens == encode.length] for text="
                                + quoted(text)
                                + " expected="
                                + encoded.length
                                + " actual="
                                + count);
            }

            String decoded = decode.apply(encoded);
            if (!text.equals(decoded)) {
                throw new AssertionError(
                        tokenizerLabel
                                + " invariant failed [decode(encode(text)) == text] for text="
                                + quoted(text)
                                + " decoded="
                                + quoted(decoded));
            }

            byte[] decodedBytes = decodeBytes.apply(encoded);
            int countedBytes = countBytes.applyAsInt(encoded);
            if (countedBytes != decodedBytes.length) {
                throw new AssertionError(
                        tokenizerLabel
                                + " invariant failed [countBytes == decodeBytes.length] for text="
                                + quoted(text)
                                + " expected="
                                + decodedBytes.length
                                + " actual="
                                + countedBytes);
            }

            String decodedFromBytes = new String(decodedBytes, StandardCharsets.UTF_8);
            if (!decoded.equals(decodedFromBytes)) {
                throw new AssertionError(
                        tokenizerLabel
                                + " invariant failed [decode == UTF-8(decodeBytes)] for text="
                                + quoted(text));
            }

            assertDecodeBytesIntoProgress(
                    tokenizerLabel, text, encoded, decodedBytes, decodeBytesInto);
        }
    }

    private static void assertDecodeBytesIntoProgress(
            String tokenizerLabel,
            String text,
            int[] encoded,
            byte[] expectedBytes,
            DecodeBytesIntoFn decodeBytesInto) {
        int tokenCount = encoded.length;
        if (tokenCount == 0) {
            int consumed = decodeBytesInto.decodeBytesInto(encoded, 0, ByteBuffer.allocate(8));
            if (consumed != 0) {
                throw new AssertionError(
                        tokenizerLabel
                                + " invariant failed [decodeBytesInto empty consumes 0] for text="
                                + quoted(text)
                                + " actual="
                                + consumed);
            }
            return;
        }

        ByteArrayOutputStream rebuilt = new ByteArrayOutputStream(expectedBytes.length);
        ByteBuffer chunk = ByteBuffer.allocate(DECODE_CHUNK_SIZE);
        int tokenIndex = 0;
        while (tokenIndex < tokenCount) {
            chunk.clear();
            int consumed = decodeBytesInto.decodeBytesInto(encoded, tokenIndex, chunk);
            if (consumed <= 0) {
                throw new AssertionError(
                        tokenizerLabel
                                + " invariant failed [decodeBytesInto progress] for text="
                                + quoted(text)
                                + " at tokenIndex="
                                + tokenIndex);
            }
            tokenIndex += consumed;
            rebuilt.write(chunk.array(), 0, chunk.position());
        }

        byte[] rebuiltBytes = rebuilt.toByteArray();
        if (!Arrays.equals(expectedBytes, rebuiltBytes)) {
            throw new AssertionError(
                    tokenizerLabel
                            + " invariant failed [decodeBytesInto loop parity] for text="
                            + quoted(text));
        }
    }

    private static void assertIntArrayEquals(
            String tokenizerLabel, String text, String check, int[] expected, int[] actual) {
        if (!Arrays.equals(expected, actual)) {
            throw new AssertionError(
                    tokenizerLabel
                            + " invariant failed ["
                            + check
                            + "] for text="
                            + quoted(text)
                            + " expected="
                            + Arrays.toString(expected)
                            + " actual="
                            + Arrays.toString(actual));
        }
    }

    // ------------------------------------------------------------------
    // Slicing invariants — encodeInto(4-arg) and countTokens(3-arg)
    // ------------------------------------------------------------------

    /**
     * Verifies that {@code encodeSlice} (the 4-arg {@code encodeInto(text, start, end, out)}) and
     * {@code countTokensSlice} (the 3-arg {@code countTokens(text, start, end)}) produce the same
     * results as their full-text counterparts for the full range [0, textLen).
     *
     * <p>Sub-range parity with independent encoding of the substring is not tested here because
     * some tokenizers (e.g., metaspace) are position-dependent.
     */
    public static void runSlicingInvariants(
            String tokenizerLabel,
            List<String> inputs,
            Function<String, int[]> encode,
            EncodeSliceFn encodeSlice,
            ToIntFunction<String> countTokens,
            CountTokensSliceFn countTokensSlice) {
        Objects.requireNonNull(tokenizerLabel, "tokenizerLabel");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(encode, "encode");
        Objects.requireNonNull(encodeSlice, "encodeSlice");
        Objects.requireNonNull(countTokens, "countTokens");
        Objects.requireNonNull(countTokensSlice, "countTokensSlice");

        for (String text : inputs) {
            if (text.isEmpty()) {
                continue;
            }
            int textLen = text.length();

            // I10: encodeSlice(full range) == encode(text)
            int[] fullSlice = encodeSlice.encodeSlice(text, 0, textLen);
            int[] fullEncode = encode.apply(text);
            assertIntArrayEquals(
                    tokenizerLabel, text, "encodeSlice(full) parity", fullEncode, fullSlice);

            // I11: countTokensSlice(full range) == countTokens(text)
            int countFull = countTokens.applyAsInt(text);
            int countSliceFull = countTokensSlice.countTokensSlice(text, 0, textLen);
            if (countFull != countSliceFull) {
                throw new AssertionError(
                        tokenizerLabel
                                + " invariant failed [countTokensSlice(full) parity] for text="
                                + quoted(text)
                                + " expected="
                                + countFull
                                + " actual="
                                + countSliceFull);
            }
        }
    }

    // ------------------------------------------------------------------
    // Convenience overload invariants — encodeToArray, decode(int[]), etc.
    // ------------------------------------------------------------------

    /**
     * Verifies that the {@code int[]}-based convenience overloads produce the same results as their
     * {@link com.qxotic.toknroll.IntSequence}-based counterparts.
     */
    public static void runConvenienceOverloadInvariants(
            String tokenizerLabel,
            List<String> inputs,
            Function<String, int[]> encode,
            Function<String, int[]> encodeToArray,
            Function<int[], String> decode,
            Function<int[], String> decodeFromArray,
            Function<int[], byte[]> decodeBytes,
            Function<int[], byte[]> decodeBytesFromArray,
            ToIntFunction<int[]> countBytes,
            ToIntFunction<int[]> countBytesFromArray) {
        Objects.requireNonNull(tokenizerLabel, "tokenizerLabel");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(encode, "encode");
        Objects.requireNonNull(encodeToArray, "encodeToArray");
        Objects.requireNonNull(decode, "decode");
        Objects.requireNonNull(decodeFromArray, "decodeFromArray");
        Objects.requireNonNull(decodeBytes, "decodeBytes");
        Objects.requireNonNull(decodeBytesFromArray, "decodeBytesFromArray");
        Objects.requireNonNull(countBytes, "countBytes");
        Objects.requireNonNull(countBytesFromArray, "countBytesFromArray");

        for (String text : inputs) {
            if (text.isEmpty()) {
                continue;
            }

            int[] encoded = encode.apply(text);

            // I12: encodeToArray == encode().toArray()
            int[] encodedViaArray = encodeToArray.apply(text);
            assertIntArrayEquals(
                    tokenizerLabel, text, "encodeToArray parity", encoded, encodedViaArray);

            // I13: decode(int[]) == decode(IntSequence.wrap(int[]))
            String decoded = decode.apply(encoded);
            String decodedFromArray = decodeFromArray.apply(encoded);
            if (!decoded.equals(decodedFromArray)) {
                throw new AssertionError(
                        tokenizerLabel
                                + " invariant failed [decode(int[]) parity] for text="
                                + quoted(text));
            }

            // I14: decodeBytes(int[]) == decodeBytes(IntSequence.wrap(int[]))
            byte[] bytes = decodeBytes.apply(encoded);
            byte[] bytesFromArray = decodeBytesFromArray.apply(encoded);
            if (!Arrays.equals(bytes, bytesFromArray)) {
                throw new AssertionError(
                        tokenizerLabel
                                + " invariant failed [decodeBytes(int[]) parity] for text="
                                + quoted(text));
            }

            // I15: countBytes(int[]) == countBytes(IntSequence.wrap(int[]))
            int countB = countBytes.applyAsInt(encoded);
            int countBFromArray = countBytesFromArray.applyAsInt(encoded);
            if (countB != countBFromArray) {
                throw new AssertionError(
                        tokenizerLabel
                                + " invariant failed [countBytes(int[]) parity] for text="
                                + quoted(text)
                                + " expected="
                                + countB
                                + " actual="
                                + countBFromArray);
            }
        }
    }

    // ------------------------------------------------------------------
    // expectedTokensPerChar sanity
    // ------------------------------------------------------------------

    /** Verifies that {@code expectedTokensPerChar} is in (0, 1] for the tokenizer. */
    public static void assertExpectedTokensPerCharRange(
            String tokenizerLabel, float expectedTokensPerChar) {
        Objects.requireNonNull(tokenizerLabel, "tokenizerLabel");
        if (expectedTokensPerChar <= 0f || expectedTokensPerChar > 1f) {
            throw new AssertionError(
                    tokenizerLabel
                            + " invariant failed [0 < expectedTokensPerChar <= 1] value="
                            + expectedTokensPerChar);
        }
    }

    private static String quoted(String text) {
        return '"'
                + text.replace("\\", "\\\\")
                        .replace("\n", "\\n")
                        .replace("\r", "\\r")
                        .replace("\t", "\\t")
                + '"';
    }
}
