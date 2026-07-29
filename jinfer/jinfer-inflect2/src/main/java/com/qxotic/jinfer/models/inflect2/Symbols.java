// Inflect-Nano-v2 phoneme symbol table — 178 symbols, model embedding vocabulary.
// Index 0 = blank/pad. Mirrors runtime/text/symbols.py byte-for-byte.
//
// Codepoint-safe: iterates code points, not UTF-16 chars. IPA symbols like
// 'ɡ' (U+0261), 'ˈ' (U+02C8), 'ɜ' (U+025C) are BMP but the symbols table
// may contain supplementary characters reached via int[]{...} escape.
package com.qxotic.jinfer.models.inflect2;

import java.util.*;

public final class Symbols {
    private Symbols() {}

    private static final String TABLE =
            "_" // 0  pad
                    + ";:,.!?\u00A1\u00BF\u2014\u2026\"\u00AB\u00BB\u201C\u201D " // 1-16
                    // punctuation
                    + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" // 17-42 uppercase
                    + "abcdefghijklmnopqrstuvwxyz" // 43-68 lowercase
                    + "\u0251\u0250\u0252\u00E6\u0253\u0299\u03B2\u0254\u0255\u00E7\u0257\u0256" // 69-80
                    + "\u00F0\u02A4\u0259\u0258\u025A\u025B\u025C\u025D\u025E\u025F\u0284\u0261" // 81-92
                    + "\u0260\u0262\u029B\u0266\u0267\u0127\u0265\u029C\u0268\u026A\u029D\u026D" // 93-104
                    + "\u026C\u026B\u026E\u029F\u0271\u026F\u0270\u014B\u0273\u0272\u0274\u00F8" // 105-116
                    + "\u0275\u0278\u03B8\u0153\u0276\u0298\u0279\u027A\u027E\u027B\u0280\u0281" // 117-128
                    + "\u027D\u0282\u0283\u0288\u02A7\u0289\u028A\u028B\u2C71\u028C\u0263\u0264" // 129-140
                    + "\u028D\u03C7\u028E\u028F\u0291\u0290\u0292\u0294\u02A1\u0295\u02A2\u01C0" // 141-152
                    + "\u01C1\u01C2\u01C3\u02C8\u02CC\u02D0\u02D1\u02BC\u02B4\u02B0\u02B1\u02B2" // 153-164
                    + "\u02B7\u02E0\u02E4\u02DE\u2193\u2191\u2192\u2197\u2198\u0027\u0329\u0027\u1D7B"; // 165-177

    private static final int[] CODEPOINTS = TABLE.codePoints().toArray();
    private static final Map<Integer, Integer> CP_TO_ID = buildMap();

    private static Map<Integer, Integer> buildMap() {
        var m = new HashMap<Integer, Integer>(178);
        for (int i = 0; i < CODEPOINTS.length; i++) m.putIfAbsent(CODEPOINTS[i], i);
        return m;
    }

    /** Number of symbols (= vocabulary size for embedding). */
    public static int count() {
        return CODEPOINTS.length;
    }

    static {
        assert count() == 178 : "Symbol count " + count() + " != 178";
    }

    /** Convert a codepoint to symbol ID, or 0 (blank) if unknown. */
    public static int idOf(int codepoint) {
        return CP_TO_ID.getOrDefault(codepoint, 0);
    }

    /** Convert cleaned IPA phoneme text to raw symbol IDs (no blanks). */
    public static int[] toRaw(String phonemes) {
        int[] cps = phonemes.codePoints().toArray();
        int[] ids = new int[cps.length];
        for (int i = 0; i < cps.length; i++) ids[i] = CP_TO_ID.getOrDefault(cps[i], 0);
        return ids;
    }

    /** Raw symbol IDs → blank-interspersed: [0, a, 0, b, 0, c, 0]. */
    public static int[] blankIntersperse(int[] raw) {
        int[] out = new int[raw.length * 2 + 1];
        int j = 0;
        out[j++] = 0;
        for (int id : raw) {
            out[j++] = id;
            out[j++] = 0;
        }
        return out;
    }

    /** Convert IPA phoneme text → blank-interspersed token sequence for the model. */
    public static int[] toTokens(String phonemes) {
        return blankIntersperse(toRaw(phonemes));
    }
}
