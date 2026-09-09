package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.codecs.Espeak;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

final class MisakiTest {

    private static final Misaki US = Misaki.forLanguage("en-us");

    /**
     * misaki's US_VOCAB with v1.0's flap and glottal stop spelled T and t: everything an American
     * Kokoro voice was trained to hear, plus the space between words.
     */
    private static final String AMERICAN_ALPHABET =
            "AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹʃʊʌʒʤʧˈˌθᵊᵻT ";

    @Test
    void diphthongsAndAffricatesBecomeOneSymbol() {
        assertEquals("ʤˈʌʤ", US.apply("d^ʒˈʌd^ʒ"));
        assertEquals("ʧˈɜɹʧ", US.apply("t^ʃˈɜːt^ʃ"));
        assertEquals("dˈA tˈIm ɡˌO hˌW bˈY", US.apply("dˈe^ɪ tˈa^ɪm ɡˌo^ʊ hˌa^ʊ bˈɔ^ɪ"));
    }

    @Test
    void theTieIsWhatTellsAnAffricateFromTwoPhonemes() {
        // "nightshirt" is t + ʃ, "headjoint" is the affricate: espeak ties only the second
        assertEquals("nˈItʃɜɹt", US.apply("nˈa^ɪtʃɜːt"));
        assertEquals("hˈɛdʤYnt", US.apply("hˈɛdd^ʒɔ^ɪnt"));
    }

    @Test
    void americanDropsLengthAndSpellsTheRhoticAndTheFlap() {
        assertEquals("wˈɔTəɹ", US.apply("wˈɔːɾɚ"));
        assertEquals("wˈɜɹld", US.apply("wˈɜːld"));
        assertEquals("ɡɹˈɑl vˌiˈɛm", US.apply("ɡɹˈɑːl vˌiːˈɛm"));
        assertEquals("əbˈWt", US.apply("ɐbˈa^ʊt"));
    }

    @Test
    void syllabicConsonantsGetTheSmallSchwa() {
        assertEquals("bˈʌtn", US.apply("bˈʌʔn̩")); // misaki's own ʔn̩ rule, then ʔ -> t
        assertEquals("lˈɪɾᵊl".replace("ɾ", "T"), US.apply("lˈɪɾə^l"));
        assertEquals("ᵊm", US.apply("m̩"));
    }

    @Test
    void britishKeepsLengthAndHasItsOwnDiphthong() {
        Misaki gb = Misaki.forLanguage("en-gb");
        assertEquals("ɡˌQ", gb.apply("ɡˌə^ʊ"));
        assertEquals("wˈɜːld", gb.apply("wˈɜːld"));
    }

    @Test
    void otherLanguagesOnlyCollapseTies() {
        Misaki es = Misaki.forLanguage("es");
        assertEquals("ʧ ʣ ʦ mˈuʧo", es.apply("t^ʃ d^z t^s mˈut^ʃo"));
        assertEquals("ab", es.apply("a-b"));
        assertEquals("eɪ", es.apply("eɪ"), "no rewrite without a tie, and no e -> A");
    }

    @Test
    void everythingAnAmericanVoiceHearsIsOnItsAlphabet() {
        Espeak espeak =
                Espeak.find().orElseGet(() -> Assumptions.abort("espeak-ng is not installed"));
        String corpus =
                "The quick brown fox jumps over the lazy dog while judging church bells"
                        + " nightshirt roadshow headjoint outshine adjoin midyear"
                        + " button water little better matter city kitten mountain"
                        + " don't isn't we're they've bird word first nurse learn heard"
                        + " oil boy toy choice voice noise about again around banana sofa"
                        + " measure vision usual pleasure garage yes yellow beyond onion million"
                        + " GraalVM thirty six percent nineteen ninety five naïve café Zürich";
        String ipa = US.apply(espeak.ipa(corpus, "en-us", Misaki.TIE));
        List<String> strangers = new ArrayList<>();
        ipa.codePoints()
                .filter(cp -> AMERICAN_ALPHABET.indexOf(cp) < 0)
                .forEach(cp -> strangers.add(Character.toString(cp)));
        assertTrue(strangers.isEmpty(), "not in Kokoro's American alphabet: " + strangers);
        assertTrue(ipa.contains("ʤ") && ipa.contains("ʧ") && ipa.contains("A"), ipa);
    }
}
