package com.qxotic.toknroll.libtoknroll;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * Whether bytes are well-formed UTF-8: the check that sends a caller's text to the tokenizer as it
 * is, without decoding it first. Ill-formed text is decoded as before, each bad sequence a U+FFFD,
 * so the ids of every input stay what they were.
 */
final class WellFormed {
    private WellFormed() {}

    private static final VarHandle LONGS =
            MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);

    /**
     * Whether {@code b[from, to)} is well-formed UTF-8 (Unicode table 3-7): no overlong form, no
     * surrogate, nothing past U+10FFFF, no sequence cut by {@code to}. Exactly what the JDK decodes
     * without a replacement.
     */
    static boolean utf8(byte[] b, int from, int to) {
        int i = from;
        while (i < to) {
            if (i + 8 <= to && ((long) LONGS.get(b, i) & 0x8080808080808080L) == 0) {
                i += 8; // eight ASCII bytes
                continue;
            }
            int c = b[i] & 0xFF;
            if (c < 0x80) {
                i++;
            } else if (c < 0xC2) {
                return false; // a continuation byte, or the lead of an overlong pair
            } else if (c < 0xE0) {
                if (i + 1 >= to || !continuation(b[i + 1])) return false;
                i += 2;
            } else if (c < 0xF0) {
                if (i + 2 >= to) return false;
                int c1 = b[i + 1] & 0xFF;
                int low = c == 0xE0 ? 0xA0 : 0x80; // E0: no overlong
                int high = c == 0xED ? 0x9F : 0xBF; // ED: no surrogate
                if (c1 < low || c1 > high || !continuation(b[i + 2])) return false;
                i += 3;
            } else if (c < 0xF5) {
                if (i + 3 >= to) return false;
                int c1 = b[i + 1] & 0xFF;
                int low = c == 0xF0 ? 0x90 : 0x80; // F0: no overlong
                int high = c == 0xF4 ? 0x8F : 0xBF; // F4: nothing past U+10FFFF
                if (c1 < low || c1 > high || !continuation(b[i + 2]) || !continuation(b[i + 3]))
                    return false;
                i += 4;
            } else {
                return false;
            }
        }
        return true;
    }

    private static boolean continuation(byte b) {
        return (b & 0xC0) == 0x80;
    }
}
