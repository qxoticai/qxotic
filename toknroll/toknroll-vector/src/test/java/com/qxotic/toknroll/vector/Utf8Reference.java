package com.qxotic.toknroll.vector;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/** The JDK's UTF-8 with a lone surrogate as U+FFFD: what {@link Utf8} must produce. */
final class Utf8Reference {
    private Utf8Reference() {}

    static byte[] encode(CharSequence text, int start, int end, int padding) {
        CharsetEncoder encoder =
                StandardCharsets.UTF_8
                        .newEncoder()
                        .onMalformedInput(CodingErrorAction.REPLACE)
                        .replaceWith(new byte[] {(byte) 0xEF, (byte) 0xBF, (byte) 0xBD});
        try {
            ByteBuffer bytes = encoder.encode(CharBuffer.wrap(text, start, end));
            return Arrays.copyOf(
                    Arrays.copyOf(bytes.array(), bytes.limit()), bytes.limit() + padding);
        } catch (CharacterCodingException e) {
            throw new IllegalStateException(e);
        }
    }
}
