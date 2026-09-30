package com.qxotic.toknroll.spi;

/** The kernels as plain Java, the reference every provider is held to. */
final class ScalarKernels implements Kernels {
    static final ScalarKernels INSTANCE = new ScalarKernels();

    private ScalarKernels() {}

    @Override
    public byte[] utf8(CharSequence text, int start, int end, int padding) {
        int length = 0;
        for (int i = start; i < end; i++) {
            char c = text.charAt(i);
            if (c < 0x80) {
                length++;
            } else if (c < 0x800) {
                length += 2;
            } else if (Character.isHighSurrogate(c)
                    && i + 1 < end
                    && Character.isLowSurrogate(text.charAt(i + 1))) {
                length += 4;
                i++;
            } else {
                length += 3;
            }
        }
        byte[] out = new byte[length + padding];
        int o = 0;
        for (int i = start; i < end; i++) {
            char c = text.charAt(i);
            if (c < 0x80) {
                out[o++] = (byte) c;
            } else if (c < 0x800) {
                out[o++] = (byte) (0xC0 | (c >> 6));
                out[o++] = (byte) (0x80 | (c & 0x3F));
            } else if (Character.isHighSurrogate(c)
                    && i + 1 < end
                    && Character.isLowSurrogate(text.charAt(i + 1))) {
                int cp = Character.toCodePoint(c, text.charAt(++i));
                out[o++] = (byte) (0xF0 | (cp >> 18));
                out[o++] = (byte) (0x80 | ((cp >> 12) & 0x3F));
                out[o++] = (byte) (0x80 | ((cp >> 6) & 0x3F));
                out[o++] = (byte) (0x80 | (cp & 0x3F));
            } else {
                int cp = Character.isSurrogate(c) ? 0xFFFD : c;
                out[o++] = (byte) (0xE0 | (cp >> 12));
                out[o++] = (byte) (0x80 | ((cp >> 6) & 0x3F));
                out[o++] = (byte) (0x80 | (cp & 0x3F));
            }
        }
        return out;
    }

    @Override
    public int split(int rules, byte[] text, int start, int end, int[] ends) {
        return -1; // the scanner in core is the reference
    }
}
