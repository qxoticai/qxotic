// English text normalizer: numbers, dates, times, money, ordinals and acronyms rewritten as the
// words a reader would say. Mirrors inflect_nano_v2_frontend.py's normalize_text().
//
// The steps run in a fixed order and each consumes what it recognizes, so the narrow patterns
// (money, dates, times, phone numbers, versions) must come before the general ones (decimals,
// ordinals, bare numbers). Pure Java, no dependencies.
package com.qxotic.jinfer.models.inflect2.frontend;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.MatchResult;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class TextNormalizer {
    private TextNormalizer() {}

    // ── vocabulary ────────────────────────────────────────────────────────

    private static final String[] ONES = {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen"
    };
    private static final String[] TENS = {
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
    };
    private static final String[] MONTHS = {
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    };

    /** How each letter is said when a word is spelled out. Indexed by {@code letter - 'A'}. */
    private static final String[] LETTER_NAMES =
            ("ay,bee,see,dee,ee,eff,gee,aitch,eye,jay,kay,ell,em,en,"
                            + "oh,pee,cue,ar,ess,tee,you,vee,double you,ex,why,zee")
                    .split(",");

    private static final Map<String, String> ABBREVIATIONS =
            Map.ofEntries(
                    Map.entry("Dr.", "doctor"),
                    Map.entry("Mr.", "mister"),
                    Map.entry("Mrs.", "missus"),
                    Map.entry("Ms.", "miss"),
                    Map.entry("Prof.", "professor"),
                    Map.entry("St.", "saint"),
                    Map.entry("vs.", "versus"),
                    Map.entry("etc.", "et cetera"),
                    Map.entry("e.g.", "for example"),
                    Map.entry("i.e.", "that is"));

    /** Built-in pronunciations for names the letter-by-letter rules would mangle. */
    private static final Map<String, String> OVERRIDES =
            Map.ofEntries(
                    Map.entry("Qwen3", "Qwen three"),
                    Map.entry("Qwen", "Qwen"),
                    Map.entry("PyTorch", "pie torch"),
                    Map.entry("SQLite", "ess cue lite"),
                    Map.entry("USB-C", "you ess bee see"),
                    Map.entry("RTX 3060", "ar tee ex thirty sixty"),
                    Map.entry("RTX 3090", "ar tee ex thirty ninety"),
                    Map.entry("RTX 4090", "ar tee ex forty ninety"),
                    Map.entry("RTX 5080", "ar tee ex fifty eighty"),
                    Map.entry("RTX 5090", "ar tee ex fifty ninety"));

    // ── the pipeline's patterns, in the order they run ────────────────────

    private static final Pattern WHITESPACE = Pattern.compile("\\s+");
    private static final Pattern BRACKETS = Pattern.compile("[()\\[\\]{}]");
    private static final Pattern DOTTED_ACRONYM = Pattern.compile("\\b([A-Z])(?:\\.([A-Z]))+\\.?");
    private static final Pattern LABELLED_NUMBER =
            Pattern.compile(
                    "\\b(apartment|apt\\.?|suite|unit|room|flight|extension|order|invoice|locker"
                            + "|aisle|gate)\\s+([A-Za-z]?\\d{1,4}[A-Za-z]?)\\b",
                    Pattern.CASE_INSENSITIVE);
    private static final Pattern STREET_NUMBER =
            Pattern.compile(
                    "\\b(\\d{3})(?=\\s+(?:North|South|East|West)\\b)", Pattern.CASE_INSENSITIVE);
    private static final Pattern MONEY = Pattern.compile("\\$(\\d[\\d,]*(?:\\.\\d{1,2})?)");
    private static final Pattern DATE =
            Pattern.compile("\\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\\d|3[01])/(20\\d{2}|19\\d{2})\\b");
    private static final Pattern CLOCK_TIME =
            Pattern.compile("\\b(\\d{1,2}):(\\d{2})\\s*([AaPp]\\.?\\s*[Mm]\\.?)?\\b");
    private static final Pattern BARE_TIME =
            Pattern.compile("\\b(\\d{1,2})\\s*([AaPp]\\.?\\s*[Mm]\\.?)\\b");
    private static final Pattern PHONE = Pattern.compile("\\b(\\d{3})-(\\d{4})\\b");
    private static final Pattern VERSION = Pattern.compile("\\b\\d+(?:\\.\\d+){2,}\\b");
    // the integer part may be comma-grouped (1,234.56): NUMBER would otherwise eat the "1,"
    private static final Pattern DECIMAL =
            Pattern.compile("\\b(\\d{1,3}(?:,\\d{3})+|\\d+)\\.(\\d+)\\b");
    private static final Pattern ORDINAL =
            Pattern.compile("\\b(\\d+)(st|nd|rd|th)\\b", Pattern.CASE_INSENSITIVE);
    private static final Pattern NUMBER = Pattern.compile("\\b\\d[\\d,]*\\b");
    private static final Pattern UPPERCASE_ACRONYM = Pattern.compile("\\b[A-Z]{2,}\\b");
    private static final Pattern IDENTIFIER = Pattern.compile("([A-Za-z]?)(\\d+)([A-Za-z]?)");

    private static final Pattern REPEATED_COMMA = Pattern.compile(",(?:\\s*,)+");
    private static final Pattern COMMA_THEN_STOP = Pattern.compile(",\\s*([.!?])");
    private static final Pattern SPACE_BEFORE_PUNCTUATION = Pattern.compile("\\s+([,;:.!?])");
    private static final Pattern PUNCTUATION_RUN_ON = Pattern.compile("([,;:.!?])(?=\\S)");

    /** Digits above this many cannot be an {@code int}, so they get spelled out. */
    private static final int MAX_NUMBER_DIGITS = 9;

    // ── public ────────────────────────────────────────────────────────────

    /** Normalize with the built-in pronunciations only. */
    public static String normalize(String text) {
        return normalize(text, Map.of());
    }

    /**
     * Normalize English text for synthesis. {@code userOverrides} join the built-in pronunciation
     * table and take priority over it. Values should be readable English, e.g. {@code "PyTorch" →
     * "pie torch"}.
     */
    public static String normalize(String text, Map<String, String> userOverrides) {
        text = punctuation(text);
        text = WHITESPACE.matcher(BRACKETS.matcher(text).replaceAll(" ")).replaceAll(" ").trim();

        Map<String, String> pronunciations = new HashMap<>(OVERRIDES);
        pronunciations.putAll(userOverrides);
        for (var entry : pronunciations.entrySet())
            text = replaceWord(text, entry.getKey(), entry.getValue(), true);
        for (var entry : ABBREVIATIONS.entrySet())
            text = replaceWord(text, entry.getKey(), entry.getValue(), false);

        text = DOTTED_ACRONYM.matcher(text).replaceAll(r -> separateLetters(r.group()));
        text =
                LABELLED_NUMBER
                        .matcher(text)
                        .replaceAll(r -> r.group(1) + " " + identifier(r.group(2)));
        text = STREET_NUMBER.matcher(text).replaceAll(r -> digitWords(r.group(1)));
        text = MONEY.matcher(text).replaceAll(TextNormalizer::money);
        text = DATE.matcher(text).replaceAll(TextNormalizer::date);
        text = CLOCK_TIME.matcher(text).replaceAll(TextNormalizer::clockTime);
        text = BARE_TIME.matcher(text).replaceAll(TextNormalizer::bareTime);
        text =
                PHONE.matcher(text)
                        .replaceAll(r -> digitWords(r.group(1)) + ", " + digitWords(r.group(2)));
        text = VERSION.matcher(text).replaceAll(TextNormalizer::version);
        text =
                DECIMAL.matcher(text)
                        .replaceAll(
                                r ->
                                        spokenNumber(r.group(1).replace(",", ""))
                                                + " point "
                                                + digitWords(r.group(2)));
        text = ORDINAL.matcher(text).replaceAll(r -> ordinal(r.group(1)));
        text = NUMBER.matcher(text).replaceAll(TextNormalizer::number);
        text = UPPERCASE_ACRONYM.matcher(text).replaceAll(r -> spellLetters(r.group()));

        return tidy(text);
    }

    // ── steps ─────────────────────────────────────────────────────────────

    /** Curly quotes, dashes and ellipses down to what the symbol table can carry. */
    private static String punctuation(String text) {
        return text.replace('\u2018', '\'')
                .replace('\u2019', '\'')
                .replace('\u201c', '"')
                .replace('\u201d', '"')
                .replace("\u2013", "-")
                .replace("\u2014", ",")
                .replace("\u2026", "...");
    }

    /** Collapse the spacing that the expansions above leave behind. */
    private static String tidy(String text) {
        text = REPEATED_COMMA.matcher(text).replaceAll(",");
        text = COMMA_THEN_STOP.matcher(text).replaceAll("$1");
        text = SPACE_BEFORE_PUNCTUATION.matcher(text).replaceAll("$1");
        text = PUNCTUATION_RUN_ON.matcher(text).replaceAll("$1 ");
        return WHITESPACE.matcher(text).replaceAll(" ").trim();
    }

    /**
     * Replace a literal term where it starts a word. {@code wholeWord} also requires the term to
     * *end* one, which abbreviations must not: "Dr." is followed by a space, not a word boundary.
     * The key is quoted as a literal and so is the replacement, so a {@code $} in either is data.
     */
    private static String replaceWord(
            String text, String term, String replacement, boolean wholeWord) {
        return text.replaceAll(
                "\\b" + Pattern.quote(term) + (wholeWord ? "\\b" : ""),
                Matcher.quoteReplacement(replacement));
    }

    /**
     * "U.S.A." → "U S A". Single letters are left as letters on purpose: the phonemizer already
     * says a lone letter by name, and a later pass only spells out runs of two or more.
     */
    private static String separateLetters(String word) {
        var out = new StringBuilder();
        for (char c : word.toCharArray()) {
            if (c < 'A' || c > 'Z') continue;
            if (!out.isEmpty()) out.append(' ');
            out.append(c);
        }
        return out.toString();
    }

    /** "ABC" → "ay bee see". */
    private static String spellLetters(String word) {
        var out = new StringBuilder();
        for (char c : word.toCharArray()) {
            if (c < 'A' || c > 'Z') continue;
            if (!out.isEmpty()) out.append(' ');
            out.append(LETTER_NAMES[c - 'A']);
        }
        return out.toString();
    }

    /** "42A" → "forty two ay"; a three-digit or leading-zero run is read digit by digit. */
    static String identifier(String token) {
        Matcher matched = IDENTIFIER.matcher(token);
        if (!matched.matches()) return token;
        String prefix = matched.group(1), digits = matched.group(2), suffix = matched.group(3);
        var out = new StringBuilder();
        if (!prefix.isEmpty()) out.append(letterName(prefix)).append(' ');
        if (digits.length() == 3 || digits.startsWith("0"))
            for (int i = 0; i < digits.length(); i++) {
                char digit = digits.charAt(i);
                out.append(digit == '0' && i > 0 ? "oh" : ONES[digit - '0']).append(' ');
            }
        else out.append(spokenNumber(digits)).append(' ');
        if (!suffix.isEmpty()) out.append(letterName(suffix)).append(' ');
        return out.toString().trim();
    }

    private static String letterName(String letter) {
        char c = Character.toUpperCase(letter.charAt(0));
        return c >= 'A' && c <= 'Z' ? LETTER_NAMES[c - 'A'] : letter;
    }

    private static String money(MatchResult match) {
        String[] parts = match.group(1).replace(",", "").split("\\.");
        var out = new StringBuilder(spokenNumber(parts[0]));
        out.append("1".equals(parts[0]) ? " dollar" : " dollars");
        if (parts.length > 1 && !parts[1].isEmpty()) {
            int cents = Integer.parseInt((parts[1] + "00").substring(0, 2));
            if (cents > 0)
                out.append(" and ")
                        .append(numberWords(cents))
                        .append(cents == 1 ? " cent" : " cents");
        }
        return out.toString();
    }

    private static String date(MatchResult match) {
        return MONTHS[Integer.parseInt(match.group(1)) - 1]
                + " "
                + ordinalWords(Integer.parseInt(match.group(2)))
                + " "
                + numberWords(Integer.parseInt(match.group(3)));
    }

    private static String clockTime(MatchResult match) {
        int hour = Integer.parseInt(match.group(1)), minute = Integer.parseInt(match.group(2));
        var out = new StringBuilder(numberWords(hour));
        if (minute == 0) out.append(" o clock");
        else if (minute < 10) out.append(" oh ").append(numberWords(minute));
        else out.append(' ').append(numberWords(minute));
        return out.append(meridiem(match.group(3))).toString();
    }

    private static String bareTime(MatchResult match) {
        return numberWords(Integer.parseInt(match.group(1))) + meridiem(match.group(2));
    }

    /** "p.m." → " p m" - the letters are said, and an absent suffix contributes nothing. */
    private static String meridiem(String suffix) {
        if (suffix == null) return "";
        var out = new StringBuilder();
        for (char c : suffix.toLowerCase().replaceAll("[^a-z]", "").toCharArray())
            out.append(' ').append(c);
        return out.toString();
    }

    private static String version(MatchResult match) {
        var out = new StringBuilder();
        for (String part : match.group().split("\\.")) {
            if (!out.isEmpty()) out.append(" point ");
            out.append(spokenNumber(part));
        }
        return out.toString();
    }

    /**
     * A long run of digits is an identifier, not a quantity - except a year-like 20xx, and except a
     * comma-grouped number: the grouping is the writer saying it is a quantity.
     */
    private static String number(MatchResult match) {
        String written = match.group();
        String digits = written.replace(",", "");
        boolean identifier =
                !written.contains(",") && digits.length() >= 5 && !digits.startsWith("20");
        return identifier ? digitWords(digits) : spokenNumber(digits);
    }

    // ── numbers to words ──────────────────────────────────────────────────

    /** As a quantity when it fits an {@code int}, else digit by digit. */
    static String spokenNumber(String digits) {
        return digits.length() > MAX_NUMBER_DIGITS
                ? digitWords(digits)
                : numberWords(Integer.parseInt(digits));
    }

    /** An ordinal as a quantity when it fits; a longer digit run is an identifier, read out. */
    static String ordinal(String digits) {
        return digits.length() > MAX_NUMBER_DIGITS
                ? digitWords(digits)
                : ordinalWords(Integer.parseInt(digits));
    }

    static String numberWords(int n) {
        if (n < 20) return ONES[n];
        if (n < 100) return TENS[n / 10] + (n % 10 > 0 ? " " + ONES[n % 10] : "");
        if (n < 1_000) {
            int rest = n % 100;
            return ONES[n / 100] + " hundred" + (rest > 0 ? " " + numberWords(rest) : "");
        }
        // every quantity spokenNumber admits (MAX_NUMBER_DIGITS) fits below a billion
        boolean millions = n >= 1_000_000;
        int scale = millions ? 1_000_000 : 1_000;
        int rest = n % scale;
        return numberWords(n / scale)
                + (millions ? " million" : " thousand")
                + (rest > 0 ? " " + numberWords(rest) : "");
    }

    static String ordinalWords(int n) {
        String[] words = numberWords(n).split(" ");
        String last = words[words.length - 1];
        words[words.length - 1] =
                switch (last) {
                    case "one" -> "first";
                    case "two" -> "second";
                    case "three" -> "third";
                    case "five" -> "fifth";
                    case "eight" -> "eighth";
                    case "nine" -> "ninth";
                    case "twelve" -> "twelfth";
                    default ->
                            last.endsWith("y")
                                    ? last.substring(0, last.length() - 1) + "ieth"
                                    : last + "th";
                };
        return String.join(" ", words);
    }

    static String digitWords(String digits) {
        var out = new StringBuilder();
        for (char digit : digits.toCharArray()) {
            if (digit < '0' || digit > '9') continue;
            if (!out.isEmpty()) out.append(' ');
            out.append(ONES[digit - '0']);
        }
        return out.toString();
    }
}
