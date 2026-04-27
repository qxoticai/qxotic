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

    private TokenizerInvariantHarness() {}

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
        ByteBuffer chunk = ByteBuffer.allocate(17);
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

    private static String quoted(String text) {
        return '"'
                + text.replace("\\", "\\\\")
                        .replace("\n", "\\n")
                        .replace("\r", "\\r")
                        .replace("\t", "\\t")
                + '"';
    }
}
