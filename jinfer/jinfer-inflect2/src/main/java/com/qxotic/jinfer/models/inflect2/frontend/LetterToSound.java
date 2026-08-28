// Best-effort English letter-to-sound: the last rung of the pronunciation ladder.
package com.qxotic.jinfer.models.inflect2.frontend;

import java.text.Normalizer;
import java.util.Locale;

/**
 * A rule-based guess at a word's pronunciation, in the model's IPA-flavored symbols, for words the
 * lexicon does not know and espeak-ng cannot see (not installed). Ordered grapheme rules, longest
 * match first, plus the silent-e convention and an initial-stress mark. The common irregulars
 * ("though", "rough", "choir") live in the lexicon, so what reaches this table is mostly rare,
 * inflected or invented words, which the regular rules cover well. A guess is never silent: any
 * input with a letter produces symbols, and every symbol it emits is in {@link
 * com.qxotic.jinfer.models.inflect2.Symbols}' table.
 */
final class LetterToSound {

    private LetterToSound() {}

    /**
     * The guess for one lowercase-or-mixed word, stress mark included, or "" for no letters. An
     * accented letter is read as its base letter (café as cafe), never dropped.
     */
    static String guess(String word) {
        String w =
                Normalizer.normalize(word.toLowerCase(Locale.ROOT), Normalizer.Form.NFD)
                        .replaceAll("\\p{M}", "");
        StringBuilder out = new StringBuilder(w.length() * 2);
        int i = 0;
        int n = w.length();
        while (i < n) {
            char c = w.charAt(i);
            if (!isLetter(c)) {
                i++;
                continue;
            }
            int consumed = consonantCluster(w, i, out);
            if (consumed == 0) {
                // y before a vowel is the consonant (yes, beyond); elsewhere it is a vowel
                boolean consonantalY =
                        c == 'y' && i + 1 < n && "aeiou".indexOf(w.charAt(i + 1)) >= 0;
                if (isVowelLetter(c) && !consonantalY) consumed = vowel(w, i, out);
                else consumed = consonant(w, i, out);
            }
            i += consumed;
        }
        return stress(out);
    }

    // ── consonants ────────────────────────────────────────────────────────

    /** Multi-letter consonant patterns; returns the letters consumed (0 = none matched). */
    private static int consonantCluster(String w, int i, StringBuilder out) {
        int n = w.length();
        if (at(w, i, "tion")) return emit(out, "ʃən", 4);
        if (at(w, i, "sion")) return emit(out, "ʒən", 4);
        if (at(w, i, "ture") && i + 4 == n) return emit(out, "ʧɚ", 4);
        if (at(w, i, "sure") && i + 4 == n) return emit(out, "ʒɚ", 4);
        if (at(w, i, "tch")) return emit(out, "ʧ", 3);
        if (at(w, i, "dge")) return emit(out, "ʤ", 3);
        if (at(w, i, "sch")) return emit(out, "ʃ", 3);
        if (at(w, i, "sh")) return emit(out, "ʃ", 2);
        if (at(w, i, "ch")) return emit(out, "ʧ", 2);
        if (at(w, i, "th")) {
            // intervocalic th voices (mother, weather); elsewhere it does not (thin, depth)
            boolean voiced =
                    i > 0
                            && isVowelLetter(w.charAt(i - 1))
                            && i + 2 < n
                            && isVowelLetter(w.charAt(i + 2));
            return emit(out, voiced ? "ð" : "θ", 2);
        }
        if (at(w, i, "ph")) return emit(out, "f", 2);
        if (at(w, i, "wh")) return emit(out, "w", 2);
        if (at(w, i, "ck")) return emit(out, "k", 2);
        if (at(w, i, "qu")) return emit(out, "kw", 2);
        if (at(w, i, "ng")) return emit(out, "ŋ", 2);
        if (at(w, i, "nk")) return emit(out, "ŋk", 2); // n velarizes before k: chunk, bank
        if (at(w, i, "wr")) return emit(out, "ɹ", 2);
        if (at(w, i, "kn")) return emit(out, "n", 2);
        if (at(w, i, "gn")) return emit(out, "n", 2);
        if (at(w, i, "gue") && i + 3 == n) return emit(out, "ɡ", 3);
        if (at(w, i, "gu") && i == 0) return emit(out, "ɡ", 2);
        if (at(w, i, "ps") && i == 0) return emit(out, "s", 2);
        return 0;
    }

    /** One consonant letter; doubles collapse to a single sound. */
    private static int consonant(String w, int i, StringBuilder out) {
        char c = w.charAt(i);
        if (i + 1 < w.length() && w.charAt(i + 1) == c && c != 'h') {
            out.append(consonantSound(w, i));
            return 2; // "ll", "tt", "ss": one sound
        }
        String sound = consonantSound(w, i);
        if (sound != null) out.append(sound);
        return 1;
    }

    private static String consonantSound(String w, int i) {
        char c = w.charAt(i);
        return switch (c) {
            case 'b', 'd', 'f', 'h', 'k', 'l', 'm', 'n', 'p', 's', 't', 'v', 'w', 'z' ->
                    String.valueOf(c);
            case 'c' -> followsFrontVowel(w, i) ? "s" : "k";
            case 'g' -> followsFrontVowel(w, i) ? "ʤ" : "ɡ";
            case 'j' -> "ʤ";
            case 'q' -> "k"; // a bare q, as in "Iraq"
            case 'r' -> "ɹ";
            case 'x' -> "ks";
            case 'y' -> "j"; // consonantal y, as in "yes" or "beyond"
            default -> null; // apostrophes and strays: skip
        };
    }

    /** The next letter is e/i/y: the soft-c/soft-g context. */
    private static boolean followsFrontVowel(String w, int i) {
        if (i + 1 >= w.length()) return false;
        char next = w.charAt(i + 1);
        return next == 'e' || next == 'i' || next == 'y';
    }

    // ── vowels ────────────────────────────────────────────────────────────

    /** The vowel cluster at {@code i}; doubles and digraphs before singles. */
    private static int vowel(String w, int i, StringBuilder out) {
        int n = w.length();
        if (at(w, i, "eigh")) return emit(out, "eɪ", 4);
        if (at(w, i, "igh")) return emit(out, "aɪ", 3);
        if (at(w, i, "ough")) return emit(out, "oʊ", 4);
        if (at(w, i, "augh")) return emit(out, "ɔː", 4);
        if (at(w, i, "ee") || at(w, i, "ea")) return emit(out, "iː", 2);
        if (at(w, i, "ai") || at(w, i, "ay")) return emit(out, "eɪ", 2);
        if (at(w, i, "oa")) return emit(out, "oʊ", 2);
        if (at(w, i, "ow")) return emit(out, "oʊ", 2);
        if (at(w, i, "oo")) return emit(out, "uː", 2);
        if (at(w, i, "ou")) return emit(out, "aʊ", 2);
        if (at(w, i, "oi") || at(w, i, "oy")) return emit(out, "ɔɪ", 2);
        if (at(w, i, "au") || at(w, i, "aw")) return emit(out, "ɔː", 2);
        if (at(w, i, "ew")) return emit(out, "uː", 2);
        if (at(w, i, "ey") && i + 2 == n) return emit(out, "i", 2); // money, valley
        // r-colored vowels, rhotic like espeak's en-us
        if (at(w, i, "ar")) return emit(out, "ɑɹ", 2);
        if (at(w, i, "er") || at(w, i, "ir") || at(w, i, "ur")) return emit(out, "ɜɹ", 2);
        if (at(w, i, "or")) return emit(out, "ɔɹ", 2);

        char c = w.charAt(i);
        if (c == 'e' && i + 1 == n && i > 0 && !isVowelLetter(w.charAt(i - 1))) {
            return 1; // final silent e: it lengthens an earlier vowel, says nothing itself
        }
        if (c == 'y' && i + 1 == n) return emit(out, "i", 1); // happy, city
        if (magicE(w, i)) {
            return emit(
                    out,
                    switch (c) {
                        case 'a' -> "eɪ";
                        case 'e' -> "iː";
                        case 'i' -> "aɪ";
                        case 'o' -> "oʊ";
                        case 'u' -> "uː";
                        default -> null;
                    },
                    1);
        }
        return emit(
                out,
                switch (c) {
                    case 'a' -> "æ";
                    case 'e' -> "ɛ";
                    case 'i' -> "ɪ";
                    case 'o' -> "ɑ";
                    case 'u' -> "ʌ";
                    case 'y' -> "ɪ";
                    default -> null;
                },
                1);
    }

    /** "rate", "site", "code": a vowel, one consonant, a final e lengthens the vowel. */
    private static boolean magicE(String w, int i) {
        int n = w.length();
        return i + 2 < n
                && !isVowelLetter(w.charAt(i + 1))
                && isLetter(w.charAt(i + 1))
                && w.charAt(i + 2) == 'e'
                && i + 3 == n;
    }

    // ── shared ────────────────────────────────────────────────────────────

    /** Initial stress, as espeak would stamp it: before the first syllable's onset. */
    private static String stress(StringBuilder ipa) {
        int vowel = firstVowel(ipa);
        if (vowel < 0) return ipa.toString();
        int onset = vowel;
        while (onset > 0 && !isVowelSymbol(ipa.charAt(onset - 1)) && ipa.charAt(onset - 1) != 'ː')
            onset--;
        ipa.insert(onset, 'ˈ');
        return ipa.toString();
    }

    private static int firstVowel(StringBuilder ipa) {
        for (int i = 0; i < ipa.length(); i++) if (isVowelSymbol(ipa.charAt(i))) return i;
        return -1;
    }

    private static boolean isVowelSymbol(char c) {
        return "aeiouæɛɪɑʌɔəɜʊɐɒ".indexOf(c) >= 0;
    }

    private static boolean at(String w, int i, String s) {
        return w.startsWith(s, i);
    }

    private static int emit(StringBuilder out, String phonemes, int consumed) {
        if (phonemes != null) out.append(phonemes);
        return consumed;
    }

    private static boolean isLetter(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }

    private static boolean isVowelLetter(char c) {
        return "aeiouy".indexOf(c) >= 0;
    }
}
