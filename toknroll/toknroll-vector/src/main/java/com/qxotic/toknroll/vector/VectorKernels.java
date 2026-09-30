package com.qxotic.toknroll.vector;

import com.qxotic.toknroll.spi.Kernels;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * {@link Kernels} with the Java Vector API. Found by core through {@code ServiceLoader} when the
 * JVM runs with {@code --add-modules jdk.incubator.vector}.
 */
public final class VectorKernels implements Kernels {
    /** Chars per vector; the species are constants so the JIT bakes the shapes in. */
    private static final VectorSpecies<Short> CHARS = ShortVector.SPECIES_PREFERRED;

    /**
     * Bytes per vector, as many lanes as {@link #CHARS}: what a vector of ASCII chars narrows to.
     */
    private static final VectorSpecies<Byte> BYTES =
            VectorSpecies.of(byte.class, VectorShape.forBitSize(CHARS.vectorBitSize() / 2));

    /** Chars copied out of the text at a time: in L1, and a surrogate pair is never cut. */
    private static final int CHUNK = 4096;

    @Override
    public byte[] utf8(CharSequence text, int start, int end, int padding) {
        char[] chunk = new char[CHUNK];
        int length = 0;
        for (int p = start; p < end; ) {
            int n = take(text, p, end, chunk);
            length += utf8Length(chunk, n);
            p += n;
        }
        byte[] out = new byte[length + padding];
        int o = 0;
        for (int p = start; p < end; ) {
            int n = take(text, p, end, chunk);
            o = utf8Write(chunk, n, out, o);
            p += n;
        }
        return out;
    }

    /** Copies the next chunk of the text into {@code chunk} and returns its length. */
    private static int take(CharSequence text, int p, int end, char[] chunk) {
        int n = Math.min(CHUNK, end - p);
        if (text instanceof String) {
            ((String) text).getChars(p, p + n, chunk, 0);
        } else if (text instanceof StringBuilder) {
            ((StringBuilder) text).getChars(p, p + n, chunk, 0);
        } else {
            for (int i = 0; i < n; i++) {
                chunk[i] = text.charAt(p + i);
            }
        }
        if (n > 1 && p + n < end && Character.isHighSurrogate(chunk[n - 1])) {
            n--; // its partner is in the next chunk
        }
        return n;
    }

    private static int utf8Length(char[] chunk, int n) {
        int length = 0;
        int i = 0;
        int lanes = CHARS.length();
        while (i + lanes <= n) {
            ShortVector v = ShortVector.fromCharArray(CHARS, chunk, i);
            if (v.compare(VectorOperators.ULT, (short) 0x80).allTrue()) {
                length += lanes;
                i += lanes;
            } else {
                long r = scalar(chunk, i, i + lanes, n, null, 0);
                length += (int) (r >>> 32);
                i += (int) r;
            }
        }
        return length + (int) (scalar(chunk, i, n, n, null, 0) >>> 32);
    }

    private static int utf8Write(char[] chunk, int n, byte[] out, int o) {
        int i = 0;
        int lanes = CHARS.length();
        while (i + lanes <= n) {
            ShortVector v = ShortVector.fromCharArray(CHARS, chunk, i);
            if (v.compare(VectorOperators.ULT, (short) 0x80).allTrue()) {
                ((ByteVector) v.convertShape(VectorOperators.S2B, BYTES, 0)).intoArray(out, o);
                o += lanes;
                i += lanes;
            } else {
                long r = scalar(chunk, i, i + lanes, n, out, o);
                o += (int) (r >>> 32);
                i += (int) r;
            }
        }
        return o + (int) (scalar(chunk, i, n, n, out, o) >>> 32);
    }

    /**
     * Converts {@code chunk[from, to)} the plain way, writing to {@code out} at {@code o} when
     * {@code out} is given, and one char more when a surrogate pair straddles {@code to}. Returns
     * the bytes produced in the high half and the chars consumed in the low half.
     */
    private static long scalar(char[] chunk, int from, int to, int n, byte[] out, int o) {
        int bytes = 0;
        int i = from;
        while (i < to) {
            char c = chunk[i++];
            if (c < 0x80) {
                if (out != null) {
                    out[o + bytes] = (byte) c;
                }
                bytes++;
            } else if (c < 0x800) {
                if (out != null) {
                    out[o + bytes] = (byte) (0xC0 | (c >> 6));
                    out[o + bytes + 1] = (byte) (0x80 | (c & 0x3F));
                }
                bytes += 2;
            } else if (Character.isHighSurrogate(c)
                    && i < n
                    && Character.isLowSurrogate(chunk[i])) {
                int cp = Character.toCodePoint(c, chunk[i++]);
                if (out != null) {
                    out[o + bytes] = (byte) (0xF0 | (cp >> 18));
                    out[o + bytes + 1] = (byte) (0x80 | ((cp >> 12) & 0x3F));
                    out[o + bytes + 2] = (byte) (0x80 | ((cp >> 6) & 0x3F));
                    out[o + bytes + 3] = (byte) (0x80 | (cp & 0x3F));
                }
                bytes += 4;
            } else {
                int cp = Character.isSurrogate(c) ? 0xFFFD : c;
                if (out != null) {
                    out[o + bytes] = (byte) (0xE0 | (cp >> 12));
                    out[o + bytes + 1] = (byte) (0x80 | ((cp >> 6) & 0x3F));
                    out[o + bytes + 2] = (byte) (0x80 | (cp & 0x3F));
                }
                bytes += 3;
            }
        }
        return ((long) bytes << 32) | (i - from);
    }
}
