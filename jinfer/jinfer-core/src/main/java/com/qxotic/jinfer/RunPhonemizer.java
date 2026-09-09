package com.qxotic.jinfer;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * {@link Phonemizer#ipa}: text is cut into punctuation-free runs of words, each run's IPA comes
 * from the grapheme-to-phoneme function, the marks between runs are re-emitted as symbols, and
 * every code point is mapped through the model's table.
 */
final class RunPhonemizer implements Phonemizer {

    private static final int ABSENT = -1;

    /** Code point to id, direct-indexed: the tables are small and this is on every utterance. */
    private final int[] ids;

    private final UnaryOperator<String> ipa;

    RunPhonemizer(List<String> symbols, UnaryOperator<String> ipa) {
        this.ipa = Objects.requireNonNull(ipa, "ipa");
        Objects.requireNonNull(symbols, "symbols");
        int highest = 0;
        for (int id = 0; id < symbols.size(); id++) {
            String symbol = symbols.get(id);
            if (symbol.isEmpty()) continue;
            if (symbol.codePointCount(0, symbol.length()) != 1)
                throw new IllegalArgumentException(
                        "symbol " + id + " is not one Unicode code point: '" + symbol + "'");
            highest = Math.max(highest, symbol.codePointAt(0));
        }
        ids = new int[highest + 1];
        Arrays.fill(ids, ABSENT);
        for (int id = 0; id < symbols.size(); id++) {
            if (symbols.get(id).isEmpty()) continue;
            int codePoint = symbols.get(id).codePointAt(0);
            if (ids[codePoint] == ABSENT) ids[codePoint] = id; // a duplicate keeps its first id
        }
        if (idOf(' ') == ABSENT)
            throw new IllegalArgumentException("symbol table has no space: runs cannot be joined");
    }

    @Override
    public int[] phonemize(String text) {
        Objects.requireNonNull(text, "text");
        var out = new StringBuilder();
        var run = new StringBuilder();
        for (String token : text.split("\\s+")) {
            if (token.isEmpty()) continue;
            int start = wordStart(token);
            int end = wordEnd(token);
            if (start > 0) {
                flush(run, out);
                out.append(token, 0, start).append(' '); // an opening quote, before its word
            }
            if (end > start) run.append(token, start, end).append(' ');
            if (end < token.length()) { // trailing punctuation closes the run
                flush(run, out);
                out.append(token, end, token.length()).append(' ');
            }
        }
        flush(run, out);
        return ids(out.toString().replaceAll("\\s+", " ").trim());
    }

    /** One G2P call for a whole punctuation-free run; the tool separates the words itself. */
    private void flush(StringBuilder run, StringBuilder out) {
        if (run.isEmpty()) return;
        out.append(ipa.apply(run.toString().trim())).append(' ');
        run.setLength(0);
    }

    /** Where the word begins: past any opening punctuation ("\"hello", "(yes"). */
    private static int wordStart(String token) {
        int start = 0;
        while (start < token.length() && !Character.isLetterOrDigit(token.codePointAt(start))) {
            start += Character.charCount(token.codePointAt(start));
        }
        return start == token.length() ? 0 : start; // all punctuation: nothing to split off
    }

    /**
     * Where the word ends and its trailing punctuation begins. An apostrophe after a letter is part
     * of the word ("dogs'", "don't"), not punctuation.
     */
    private static int wordEnd(String token) {
        int end = token.length();
        while (end > 0) {
            int codePoint = token.codePointAt(end - 1);
            if (Character.isLetterOrDigit(codePoint)) break;
            if (codePoint == '\'' && end > 1 && Character.isLetter(token.codePointBefore(end - 1)))
                break;
            end--;
        }
        return end;
    }

    /**
     * Ids for IPA text. A code point the table lacks is dropped, never mapped to the pad, and
     * leaves no trace: the separators around it collapse to one.
     */
    private int[] ids(String ipa) {
        int space = idOf(' ');
        int[] out = new int[ipa.length()];
        int length = 0;
        for (int i = 0; i < ipa.length(); ) {
            int codePoint = ipa.codePointAt(i);
            i += Character.charCount(codePoint);
            int id = idOf(codePoint);
            if (id == ABSENT) continue;
            if (id == space && (length == 0 || out[length - 1] == space)) continue;
            out[length++] = id;
        }
        if (length > 0 && out[length - 1] == space) length--;
        return Arrays.copyOf(out, length);
    }

    private int idOf(int codePoint) {
        return codePoint < ids.length ? ids[codePoint] : ABSENT;
    }
}
