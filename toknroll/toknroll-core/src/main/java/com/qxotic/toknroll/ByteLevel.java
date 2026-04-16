package com.qxotic.toknroll;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Byte-level encoding used by GPT-2-style tokenizers.
 *
 * <p>It encodes arbitrary byte sequences into printable symbol strings and decodes them back
 * losslessly. This lets BPE tokenizers operate on text-safe symbols while preserving exact bytes.
 */
public final class ByteLevel {

    private static final char BYTE_ZERO_SYMBOL = '\u0100'; // 'Ā'

    private static final Map<Integer, Integer> BYTE_TO_SYMBOL = bytesToUnicode();

    private static final char[] FAST_BYTE_ENCODER = asCharArray(BYTE_TO_SYMBOL);

    private static final byte[] FAST_BYTE_DECODER = asByteArray(BYTE_TO_SYMBOL);

    private ByteLevel() {}

    /**
     * Encodes raw bytes into GPT-2 byte-level symbols.
     *
     * @param bytes input bytes to encode
     * @return symbol string where each character represents one original byte
     */
    public static String encode(byte[] bytes) {
        Objects.requireNonNull(bytes, "bytes");
        StringBuilder sb = new StringBuilder(bytes.length);
        for (byte b : bytes) {
            sb.append(encodeSingle(b));
        }
        return sb.toString();
    }

    /**
     * Decodes GPT-2 byte-level symbols back into raw bytes.
     *
     * @param symbols symbol string produced by {@link #encode(byte[])}
     * @return decoded bytes
     * @throws IllegalArgumentException if any symbol is not part of the byte-level mapping
     */
    public static byte[] decode(String symbols) {
        Objects.requireNonNull(symbols, "symbols");
        byte[] bytes = new byte[symbols.length()];
        for (int i = 0; i < symbols.length(); ++i) {
            bytes[i] = decodeSingle(symbols.charAt(i));
        }
        return bytes;
    }

    private static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        boolean[] used = new boolean[256];
        IntStream.rangeClosed('!', '~').forEach(bs::add);
        IntStream.rangeClosed('¡', '¬').forEach(bs::add);
        IntStream.rangeClosed('®', 'ÿ').forEach(bs::add);
        for (int b : bs) {
            used[b] = true;
        }

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!used[b]) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }

        return IntStream.range(0, bs.size())
                .boxed()
                .collect(Collectors.toUnmodifiableMap(bs::get, cs::get));
    }

    private static char[] asCharArray(Map<Integer, Integer> mapping) {
        int maxElement = mapping.keySet().stream().mapToInt(Integer::intValue).max().orElseThrow();
        char[] fastMapping = new char[maxElement + 1];
        for (Map.Entry<Integer, Integer> entry : mapping.entrySet()) {
            fastMapping[entry.getKey()] = (char) entry.getValue().intValue();
        }
        return fastMapping;
    }

    private static byte[] asByteArray(Map<Integer, Integer> byteToSymbol) {
        int maxSymbol =
                byteToSymbol.values().stream().mapToInt(Integer::intValue).max().orElseThrow();
        byte[] fastMapping = new byte[maxSymbol + 1];
        for (Map.Entry<Integer, Integer> entry : byteToSymbol.entrySet()) {
            fastMapping[entry.getValue()] = (byte) (entry.getKey() & 0xFF);
        }
        return fastMapping;
    }

    /**
     * Decodes a single GPT-2 byte-level symbol to its byte value.
     *
     * @param ch encoded symbol character
     * @return decoded byte
     * @throws IllegalArgumentException if {@code ch} is not part of the byte-level mapping
     */
    public static byte decodeSingle(char ch) {
        if (ch < FAST_BYTE_DECODER.length) {
            byte decoded = FAST_BYTE_DECODER[ch];
            if (decoded != 0 || ch == BYTE_ZERO_SYMBOL) {
                return decoded;
            }
        }
        throw invalidByteLevelSymbol(ch);
    }

    private static IllegalArgumentException invalidByteLevelSymbol(char ch) {
        return new IllegalArgumentException(
                "Invalid byte-level symbol U+" + String.format("%04X", (int) ch));
    }

    /**
     * Encodes a single byte to its GPT-2 byte-level symbol.
     *
     * @param b byte to encode
     * @return encoded symbol character
     */
    public static char encodeSingle(byte b) {
        return FAST_BYTE_ENCODER[Byte.toUnsignedInt(b)];
    }
}
