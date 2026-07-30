// Inflect-Nano-v2 phoneme symbol table — 178 symbols, the model's embedding vocabulary.
// Index 0 is the blank/pad; mirrors runtime/text/symbols.py.
//
// Codepoint-safe throughout: IPA symbols are iterated as code points, not UTF-16 chars.
package com.qxotic.jinfer.models.inflect2;

public final class Symbols {
    private Symbols() {}

    /** The vocabulary, in embedding-row order. */
    private static final String TABLE =
            "_" // 0  pad
                    + ";:,.!?¡¿—…\"«»“” " // 1-16
                    // punctuation
                    + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" // 17-42 uppercase
                    + "abcdefghijklmnopqrstuvwxyz" // 43-68 lowercase
                    + "ɑɐɒæɓʙβɔɕçɗɖ" // 69-80
                    + "ðʤəɘɚɛɜɝɞɟʄɡ" // 81-92
                    + "ɠɢʛɦɧħɥʜɨɪʝɭ" // 93-104
                    + "ɬɫɮʟɱɯɰŋɳɲɴø" // 105-116
                    + "ɵɸθœɶʘɹɺɾɻʀʁ" // 117-128
                    + "ɽʂʃʈʧʉʊʋⱱʌɣɤ" // 129-140
                    + "ʍχʎʏʑʐʒʔʡʕʢǀ" // 141-152
                    + "ǁǂǃˈˌːˑʼʴʰʱʲ" // 153-164
                    + "ʷˠˤ˞↓↑→↗↘'̩'ᵻ"; // 165-177

    /** The vocabulary size the checked-in weights were trained with. */
    private static final int EXPECTED_COUNT = 178;

    private static final int[] CODEPOINTS = TABLE.codePoints().toArray();

    /**
     * Codepoint → symbol id, direct-indexed (the table's highest codepoint is U+2C71, so this is
     * ~45 KB and needs no boxing). Zero doubles as "not in the table", which is right: the only
     * symbol with id 0 is the pad, and the table's duplicate apostrophes keep their first id.
     */
    private static final int[] IDS = index();

    private static int[] index() {
        if (CODEPOINTS.length != EXPECTED_COUNT)
            throw new IllegalStateException(
                    "symbol table has "
                            + CODEPOINTS.length
                            + " entries, expected "
                            + EXPECTED_COUNT);
        int highest = 0;
        for (int codePoint : CODEPOINTS) highest = Math.max(highest, codePoint);
        int[] ids = new int[highest + 1];
        for (int id = 0; id < CODEPOINTS.length; id++)
            if (ids[CODEPOINTS[id]] == 0) ids[CODEPOINTS[id]] = id;
        return ids;
    }

    /** Number of symbols — the embedding table's row count. */
    public static int count() {
        return CODEPOINTS.length;
    }

    /** Symbol id for a codepoint, or 0 (the blank) if it is not in the table. */
    public static int idOf(int codePoint) {
        return codePoint >= 0 && codePoint < IDS.length ? IDS[codePoint] : 0;
    }

    /** Cleaned IPA phoneme text → symbol ids, one per code point. */
    public static int[] toRaw(String phonemes) {
        return phonemes.codePoints().map(Symbols::idOf).toArray();
    }

    /** Symbol ids → blank-interspersed, as the model expects: {@code [0, a, 0, b, 0]}. */
    public static int[] blankIntersperse(int[] ids) {
        int[] out = new int[ids.length * 2 + 1];
        for (int i = 0; i < ids.length; i++) out[i * 2 + 1] = ids[i];
        return out;
    }

    /** IPA phoneme text → the blank-interspersed token sequence the model takes. */
    public static int[] toTokens(String phonemes) {
        return blankIntersperse(toRaw(phonemes));
    }
}
