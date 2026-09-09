package com.qxotic.jinfer.models.kokoro;

import java.util.List;
import java.util.function.UnaryOperator;
import java.util.regex.Pattern;

/**
 * espeak's IPA rewritten into the dialect Kokoro v1.0 was trained on: misaki's. Kokoro's table has
 * {@code A I O W Y} for the diphthongs, {@code ʤ ʧ} for the affricates and {@code ᵊ} for a syllabic
 * schwa because misaki writes them that way before lookup - fed espeak's {@code e ɪ} and {@code d
 * ʒ} instead, the model saw two symbols where it learned one.
 *
 * <p>A transcription of misaki's {@code espeak.EspeakFallback} (English) and {@code EspeakG2P}
 * (everything else), applied to TIED espeak output: {@code d^ʒ} is an affricate, {@code dʒ} two
 * phonemes, and only the tie tells them apart ("headjoint" against "nightshirt"). Faithful to the
 * reference, quirks included - Kokoro heard the quirks too.
 */
final class Misaki implements UnaryOperator<String> {

    /** The tie {@link com.qxotic.jinfer.codecs.Espeak#ipa(String, String, String)} is asked for. */
    static final String TIE = "^";

    private static final String SYLLABIC = "̩";
    private static final Pattern SYLLABIC_AFTER = Pattern.compile("(\\S)" + SYLLABIC);

    /** misaki's E2M for English, longest key first as it sorts them. */
    private static final List<String[]> ENGLISH =
            List.of(
                    new String[] {"ʔˌn" + SYLLABIC, "ʔn"},
                    new String[] {"ʔn" + SYLLABIC, "ʔn"},
                    new String[] {"a^ɪ", "I"},
                    new String[] {"a^ʊ", "W"},
                    new String[] {"d^ʒ", "ʤ"},
                    new String[] {"e^ɪ", "A"},
                    new String[] {"t^ʃ", "ʧ"},
                    new String[] {"ɔ^ɪ", "Y"},
                    new String[] {"ə^l", "ᵊl"},
                    new String[] {"ʲo", "jo"},
                    new String[] {"ʲə", "jə"},
                    new String[] {"e", "A"},
                    new String[] {"ʲ", ""},
                    new String[] {"ɚ", "əɹ"},
                    new String[] {"r", "ɹ"},
                    new String[] {"x", "k"},
                    new String[] {"ç", "k"},
                    new String[] {"ɐ", "ə"},
                    new String[] {"ɬ", "l"},
                    new String[] {"̃", ""});

    /** misaki's e2m for the other espeak languages: ties to single symbols, nothing else. */
    private static final List<String[]> OTHER =
            List.of(
                    new String[] {"a^ɪ", "I"},
                    new String[] {"a^ʊ", "W"},
                    new String[] {"d^z", "ʣ"},
                    new String[] {"d^ʒ", "ʤ"},
                    new String[] {"e^ɪ", "A"},
                    new String[] {"o^ʊ", "O"},
                    new String[] {"ə^ʊ", "Q"},
                    new String[] {"s^s", "S"},
                    new String[] {"t^s", "ʦ"},
                    new String[] {"t^ʃ", "ʧ"},
                    new String[] {"ɔ^ɪ", "Y"});

    private enum Flavor {
        AMERICAN,
        BRITISH,
        OTHER
    }

    private final Flavor flavor;

    private Misaki(Flavor flavor) {
        this.flavor = flavor;
    }

    /** The rewrite for an espeak voice, as {@link Kokoro} names them from the voice pack. */
    static Misaki forLanguage(String language) {
        return new Misaki(
                switch (language) {
                    case "en-us" -> Flavor.AMERICAN;
                    case "en-gb" -> Flavor.BRITISH;
                    default -> Flavor.OTHER;
                });
    }

    @Override
    public String apply(String tiedIpa) {
        return switch (flavor) {
            case AMERICAN -> english(tiedIpa, false);
            case BRITISH -> english(tiedIpa, true);
            case OTHER -> other(tiedIpa);
        };
    }

    private static String english(String ps, boolean british) {
        for (String[] rule : ENGLISH) ps = ps.replace(rule[0], rule[1]);
        ps = SYLLABIC_AFTER.matcher(ps).replaceAll("ᵊ$1").replace(SYLLABIC, "");
        if (british) {
            ps = ps.replace("e^ə", "ɛː").replace("iə", "ɪə").replace("ə^ʊ", "Q");
        } else {
            ps =
                    ps.replace("o^ʊ", "O")
                            .replace("ɜːɹ", "ɜɹ")
                            .replace("ɜː", "ɜɹ")
                            .replace("ɪə", "iə")
                            .replace("ː", "");
        }
        ps = ps.replace("o", "ɔ");
        // Kokoro v1.0: the flap and the glottal stop are written as T and t
        ps = ps.replace("ɾ", "T").replace("ʔ", "t");
        return ps.replace(TIE, "");
    }

    private static String other(String ps) {
        for (String[] rule : OTHER) ps = ps.replace(rule[0], rule[1]);
        return ps.replace(TIE, "").replace("-", "");
    }
}
