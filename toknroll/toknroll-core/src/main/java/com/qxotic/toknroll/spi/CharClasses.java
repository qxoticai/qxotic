package com.qxotic.toknroll.spi;

/**
 * The character classes of the tiktoken family of pre-tokenization regexes, as java.util.regex
 * defines them under {@code UNICODE_CHARACTER_CLASS}: a bit per class, tabulated for the BMP. The
 * scanner in core and a {@link Kernels#split} provider classify with this same table.
 */
public final class CharClasses {
    public static final int SPACE = 1; // \s
    public static final int LETTER = 2; // \p{L}
    public static final int NUMBER = 4; // \p{N}
    public static final int MARK = 8; // \p{M}
    public static final int UPPER = 16; // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
    public static final int LOWER = 32; // [\p{Ll}\p{Lm}\p{Lo}\p{M}]
    public static final int NEWLINE = 64; // [\r\n]

    private static final byte[] BMP = new byte[0x10000];

    static {
        for (int c = 0; c < BMP.length; c++) {
            BMP[c] = (byte) compute(c);
        }
    }

    private CharClasses() {}

    /** The classes of a code point, as bits. */
    public static int of(int codePoint) {
        return codePoint < 0x10000 ? BMP[codePoint] : compute(codePoint);
    }

    /** The classes of a code point, from {@link Character#getType}. */
    public static int compute(int cp) {
        int type = Character.getType(cp);
        int cls = 0;
        switch (type) {
            case Character.UPPERCASE_LETTER:
            case Character.TITLECASE_LETTER:
                cls = LETTER | UPPER;
                break;
            case Character.LOWERCASE_LETTER:
                cls = LETTER | LOWER;
                break;
            case Character.MODIFIER_LETTER:
            case Character.OTHER_LETTER:
                cls = LETTER | UPPER | LOWER;
                break;
            case Character.NON_SPACING_MARK:
            case Character.ENCLOSING_MARK:
            case Character.COMBINING_SPACING_MARK:
                cls = MARK | UPPER | LOWER;
                break;
            case Character.DECIMAL_DIGIT_NUMBER:
            case Character.LETTER_NUMBER:
            case Character.OTHER_NUMBER:
                cls = NUMBER;
                break;
            case Character.SPACE_SEPARATOR:
            case Character.LINE_SEPARATOR:
            case Character.PARAGRAPH_SEPARATOR:
                cls = SPACE;
                break;
            default:
                break;
        }
        if ((cp >= 0x9 && cp <= 0xD) || cp == 0x85) {
            cls |= SPACE;
        }
        if (cp == '\r' || cp == '\n') {
            cls |= NEWLINE;
        }
        return cls;
    }
}
