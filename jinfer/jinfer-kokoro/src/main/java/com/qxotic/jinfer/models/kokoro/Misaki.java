package com.qxotic.jinfer.models.kokoro;

import com.qxotic.jinfer.codecs.Espeak;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;
import java.util.regex.Pattern;

/**
 * espeak's IPA rewritten into the dialect Kokoro v1.0 was trained on: misaki's. Kokoro's table has
 * {@code A I O W Y} for the diphthongs, {@code ʤ ʧ} for the affricates and {@code ᵊ} for a syllabic
 * schwa because misaki writes them that way before lookup - fed espeak's {@code e ɪ} and {@code d
 * ʒ} instead, the model saw two symbols where it learned one.
 *
 * <p>A transcription of misaki's {@code espeak.EspeakFallback} (English) and {@code EspeakG2P}
 * (everything else), applied to {@link Espeak#tiedIpa tied} espeak output: {@code d^ʒ} is an
 * affricate, {@code dʒ} two phonemes, and only the tie tells them apart ("headjoint" against
 * "nightshirt"). Faithful to the reference, quirks included - Kokoro heard the quirks too.
 */
final class Misaki implements UnaryOperator<String> {

    private static final String SYLLABIC = "̩";
    private static final Pattern SYLLABIC_AFTER = Pattern.compile("(\\S)" + SYLLABIC);

    // Rewrite tables: (from, to), applied in order.

    /** misaki's E2M for English, longest key first as it sorts them. */
    private static final List<Map.Entry<String, String>> ENGLISH =
            List.of(
                    Map.entry("ʔˌn" + SYLLABIC, "ʔn"),
                    Map.entry("ʔn" + SYLLABIC, "ʔn"),
                    Map.entry("a^ɪ", "I"),
                    Map.entry("a^ʊ", "W"),
                    Map.entry("d^ʒ", "ʤ"),
                    Map.entry("e^ɪ", "A"),
                    Map.entry("t^ʃ", "ʧ"),
                    Map.entry("ɔ^ɪ", "Y"),
                    Map.entry("ə^l", "ᵊl"),
                    Map.entry("ʲo", "jo"),
                    Map.entry("ʲə", "jə"),
                    Map.entry("e", "A"),
                    Map.entry("ʲ", ""),
                    Map.entry("ɚ", "əɹ"),
                    Map.entry("r", "ɹ"),
                    Map.entry("x", "k"),
                    Map.entry("ç", "k"),
                    Map.entry("ɐ", "ə"),
                    Map.entry("ɬ", "l"),
                    Map.entry("̃", ""));

    private static final List<Map.Entry<String, String>> AMERICAN =
            List.of(
                    Map.entry("o^ʊ", "O"),
                    Map.entry("ɜːɹ", "ɜɹ"),
                    Map.entry("ɜː", "ɜɹ"),
                    Map.entry("ɪə", "iə"),
                    Map.entry("ː", ""));
    private static final List<Map.Entry<String, String>> BRITISH =
            List.of(Map.entry("e^ə", "ɛː"), Map.entry("iə", "ɪə"), Map.entry("ə^ʊ", "Q"));

    /** Every English voice, last: v1.0 writes the flap and the glottal stop as T and t. */
    private static final List<Map.Entry<String, String>> ENGLISH_TAIL =
            List.of(
                    Map.entry("o", "ɔ"),
                    Map.entry("ɾ", "T"),
                    Map.entry("ʔ", "t"),
                    Map.entry(Espeak.TIE, ""));

    /** misaki's e2m for the other espeak languages: ties to single symbols, nothing else. */
    private static final List<Map.Entry<String, String>> OTHER =
            List.of(
                    Map.entry("a^ɪ", "I"),
                    Map.entry("a^ʊ", "W"),
                    Map.entry("d^z", "ʣ"),
                    Map.entry("d^ʒ", "ʤ"),
                    Map.entry("e^ɪ", "A"),
                    Map.entry("o^ʊ", "O"),
                    Map.entry("ə^ʊ", "Q"),
                    Map.entry("s^s", "S"),
                    Map.entry("t^s", "ʦ"),
                    Map.entry("t^ʃ", "ʧ"),
                    Map.entry("ɔ^ɪ", "Y"),
                    Map.entry(Espeak.TIE, ""),
                    Map.entry("-", ""));

    /** The English voice's regional pass, or null for a language misaki treats as "other". */
    private final List<Map.Entry<String, String>> regional;

    private Misaki(List<Map.Entry<String, String>> regional) {
        this.regional = regional;
    }

    /** The rewrite for an espeak voice, as {@link Kokoro} names them from the voice pack. */
    static Misaki forLanguage(String language) {
        return new Misaki(
                switch (language) {
                    case "en-us" -> AMERICAN;
                    case "en-gb" -> BRITISH;
                    default -> null;
                });
    }

    @Override
    public String apply(String tiedIpa) {
        if (regional == null) return rewrite(tiedIpa, OTHER);
        String ipa = rewrite(tiedIpa, ENGLISH);
        ipa = SYLLABIC_AFTER.matcher(ipa).replaceAll("ᵊ$1").replace(SYLLABIC, "");
        return rewrite(rewrite(ipa, regional), ENGLISH_TAIL);
    }

    private static String rewrite(String ipa, List<Map.Entry<String, String>> rules) {
        for (var rule : rules) ipa = ipa.replace(rule.getKey(), rule.getValue());
        return ipa;
    }
}
