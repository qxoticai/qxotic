package com.qxotic.tokenizers.advanced;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Implements the byte-to-unicode encoding scheme used in GPT-2 tokenization. This encoding allows
 * arbitrary byte sequences to be represented as printable Unicode strings, which is essential for
 * tokenizer vocabularies that need to handle raw bytes while avoiding special characters that could
 * interfere with tokenization.
 *
 * <p>The encoding scheme works by:
 *
 * <ul>
 *   <li>Using printable ASCII characters ('!' to '~') directly
 *   <li>Using selected Unicode characters ('¡' to 'ÿ') for additional byte values
 *   <li>Mapping remaining bytes to unused Unicode characters (starting at code point 256)
 * </ul>
 *
 * <p>This encoding is reversible, allowing perfect reconstruction of the original byte sequences.
 * It's particularly useful for handling unknown Unicode characters and binary data in tokenization
 * pipelines.
 *
 * @see <a href="https://github.com/openai/gpt-2/blob/master/src/encoder.py">Original GPT-2
 *     Implementation</a>
 */
public final class ByteEncoding {

    /**
     * Generates the byte-to-unicode mapping used for encoding.
     *
     * <p>The mapping ensures that:
     *
     * <ul>
     *   <li>All byte values (0-255) are mapped to printable Unicode characters
     *   <li>Common ASCII characters maintain their original values
     *   <li>Control characters and whitespace are avoided in the encoding
     * </ul>
     *
     * @return an unmodifiable map from byte values to Unicode code points
     */
    private static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        IntStream.rangeClosed('!', '~').forEach(bs::add);
        IntStream.rangeClosed('¡', '¬').forEach(bs::add);
        IntStream.rangeClosed('®', 'ÿ').forEach(bs::add);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }

        return IntStream.range(0, bs.size())
                .boxed()
                .collect(Collectors.toUnmodifiableMap(bs::get, cs::get));
    }

    /** Pre-computed encoding lookup table for fast byte-to-unicode conversion. */
    private static final int[] FAST_BYTE_ENCODER = asArray(bytesToUnicode());

    /** Pre-computed decoding lookup table for fast unicode-to-byte conversion. */
    private static final int[] FAST_BYTE_DECODER =
            asArray(
                    bytesToUnicode().entrySet().stream()
                            .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey)));

    /**
     * Converts a mapping to an array for faster lookup operations.
     *
     * @param mapping the mapping to convert
     * @return an array representation of the mapping
     */
    private static int[] asArray(Map<Integer, Integer> mapping) {
        int maxElement = mapping.keySet().stream().mapToInt(Integer::intValue).max().orElseThrow();
        int[] fastMapping = new int[maxElement + 1];
        for (Map.Entry<Integer, Integer> entry : mapping.entrySet()) {
            fastMapping[entry.getKey()] = entry.getValue();
        }
        return fastMapping;
    }

    /**
     * Converts a byte array to its encoded string representation.
     *
     * @param bytes the byte array to encode
     * @return a string containing the encoded representation of the bytes
     */
    public static String bytesToString(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length);
        for (byte b : bytes) {
            char encodedByte = encodeSingle(b);
            sb.append(encodedByte);
        }
        return sb.toString();
    }

    /**
     * Converts an encoded string back to its original byte array.
     *
     * @param string the encoded string to decode
     * @return the original byte array
     */
    public static byte[] stringToBytes(String string) {
        byte[] bytes = new byte[string.length()];
        stringToBytes(string, bytes, 0);
        return bytes;
    }

    /**
     * Decodes an encoded string into a provided byte array at the specified offset.
     *
     * @param string the encoded string to decode
     * @param destBytes the destination byte array
     * @param destOffset the offset in the destination array where to start writing
     * @throws IndexOutOfBoundsException if destOffset is negative or if destBytes is too small
     */
    public static void stringToBytes(String string, byte[] destBytes, int destOffset) {
        for (int j = 0; j < string.length(); ++j) {
            char ch = string.charAt(j);
            byte decodedByte = decodeSingle(ch);
            destBytes[destOffset + j] = decodedByte;
        }
    }

    /**
     * Decodes a single character back to its original byte value.
     *
     * @param ch the character to decode
     * @return the original byte value
     * @throws ArrayIndexOutOfBoundsException if the character is not in the encoding map
     */
    public static byte decodeSingle(char ch) {
        return (byte) FAST_BYTE_DECODER[ch];
    }

    /**
     * Encodes a single byte to its character representation.
     *
     * @param b the byte to encode
     * @return the encoded character
     */
    public static char encodeSingle(byte b) {
        return (char) FAST_BYTE_ENCODER[Byte.toUnsignedInt(b)];
    }
}
