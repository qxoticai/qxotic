// English text normalizer. Mirrors inflect_nano_v2_frontend.py normalize_text().
// Converts numbers, dates, times, money, ordinals, acronyms into readable English.
// Pure Java — no external dependencies.
package com.qxotic.jinfer.models.inflect2.frontend;

import java.util.*;
import java.util.regex.*;

public final class TextNormalizer {
    private TextNormalizer() {}

    private static final String[] ONES = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                "eighteen", "nineteen"
    };
    private static final String[] TENS = {
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
    };
    private static final String[] MONTHS = {
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    };

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

    private static final Map<String, String> LETTERS = new HashMap<>();

    static {
        for (String[] p :
                new String[][] {
                    {"A", "ay"},
                    {"B", "bee"},
                    {"C", "see"},
                    {"D", "dee"},
                    {"E", "ee"},
                    {"F", "eff"},
                    {"G", "gee"},
                    {"H", "aitch"},
                    {"I", "eye"},
                    {"J", "jay"},
                    {"K", "kay"},
                    {"L", "ell"},
                    {"M", "em"},
                    {"N", "en"},
                    {"O", "oh"},
                    {"P", "pee"},
                    {"Q", "cue"},
                    {"R", "ar"},
                    {"S", "ess"},
                    {"T", "tee"},
                    {"U", "you"},
                    {"V", "vee"},
                    {"W", "double you"},
                    {"X", "ex"},
                    {"Y", "why"},
                    {"Z", "zee"}
                }) LETTERS.put(p[0], p[1]);
    }

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

    // ── public ──────────────────────────────────────────────────────────────

    /** Normalize with built-in word overrides only. */
    public static String normalize(String text) {
        return normalize(text, Map.of());
    }

    /**
     * Normalize English text for synthesis. {@code userOverrides} are applied <em>before</em> the
     * built-in pronunciation table — user entries take priority. Override values should be readable
     * English (e.g. {@code "PyTorch" → "pie torch"}).
     */
    public static String normalize(String text, Map<String, String> userOverrides) {
        // 1. Unicode punctuation normalisation
        text =
                text.replace('\u2018', '\'')
                        .replace('\u2019', '\'')
                        .replace('\u201c', '"')
                        .replace('\u201d', '"')
                        .replace("\u2013", "-")
                        .replace("\u2014", ",")
                        .replace("\u2026", "...");
        text =
                text.replace("(", " ")
                        .replace(")", " ")
                        .replace("[", " ")
                        .replace("]", " ")
                        .replace("{", " ")
                        .replace("}", " ");
        text = text.replaceAll("\\s+", " ").trim();

        // 2. Word overrides (user overrides take priority over built-in)
        Map<String, String> merged = new HashMap<>(OVERRIDES);
        merged.putAll(userOverrides);
        for (var e : merged.entrySet())
            text = text.replaceAll("\\b" + Pattern.quote(e.getKey()) + "\\b", e.getValue());

        // 3. Abbreviations
        for (var e : ABBREVIATIONS.entrySet())
            text = text.replaceAll("\\b" + Pattern.quote(e.getKey()), e.getValue());

        // 4. Acronyms with periods
        text =
                Pattern.compile("\\b([A-Z])(?:\\.([A-Z]))+\\.?")
                        .matcher(text)
                        .replaceAll(
                                r -> {
                                    var sb = new StringBuilder();
                                    for (char c : r.group().toCharArray())
                                        if (c >= 'A' && c <= 'Z') sb.append(c).append(' ');
                                    return sb.toString().trim();
                                });

        // 5. Labeled identifiers: "apt 42A"
        text =
                Pattern.compile(
                                "\\b(apartment|apt\\.?|suite|unit|room|flight|extension|order|invoice|locker|aisle|gate)\\s+([A-Za-z]?\\d{1,4}[A-Za-z]?)\\b",
                                Pattern.CASE_INSENSITIVE)
                        .matcher(text)
                        .replaceAll(r -> r.group(1) + " " + expandIdent(r.group(2)));

        // 6. Street numbers
        text =
                Pattern.compile(
                                "\\b(\\d{3})(?=\\s+(?:North|South|East|West)\\b)",
                                Pattern.CASE_INSENSITIVE)
                        .matcher(text)
                        .replaceAll(r -> digitWords(r.group(1)));

        // 7. Money
        text =
                Pattern.compile("\\$(\\d[\\d,]*(?:\\.\\d{1,2})?)")
                        .matcher(text)
                        .replaceAll(TextNormalizer::money);

        // 8. Dates
        text =
                Pattern.compile(
                                "\\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\\d|3[01])/(20\\d{2}|19\\d{2})\\b")
                        .matcher(text)
                        .replaceAll(TextNormalizer::date);

        // 9. Time
        text =
                Pattern.compile("\\b(\\d{1,2}):(\\d{2})\\s*([AaPp]\\.?\\s*[Mm]\\.?)?\\b")
                        .matcher(text)
                        .replaceAll(TextNormalizer::time);
        text =
                Pattern.compile("\\b(\\d{1,2})\\s*([AaPp]\\.?\\s*[Mm]\\.?)\\b")
                        .matcher(text)
                        .replaceAll(TextNormalizer::bareTime);

        // 10. Phone
        text =
                Pattern.compile("\\b(\\d{3})-(\\d{4})\\b")
                        .matcher(text)
                        .replaceAll(r -> digitWords(r.group(1)) + ", " + digitWords(r.group(2)));

        // 11. Version strings
        text =
                Pattern.compile("\\b\\d+(?:\\.\\d+){2,}\\b")
                        .matcher(text)
                        .replaceAll(
                                r -> {
                                    var sb = new StringBuilder();
                                    for (String p : r.group().split("\\."))
                                        sb.append(numberWords(Integer.parseInt(p)))
                                                .append(" point ");
                                    return sb.substring(0, sb.length() - 7);
                                });

        // 12. Decimals
        text =
                Pattern.compile("\\b(\\d+)\\.(\\d+)\\b")
                        .matcher(text)
                        .replaceAll(
                                r ->
                                        numberWords(Integer.parseInt(r.group(1)))
                                                + " point "
                                                + digitWords(r.group(2)));

        // 13. Ordinals
        text =
                Pattern.compile("\\b(\\d+)(st|nd|rd|th)\\b", Pattern.CASE_INSENSITIVE)
                        .matcher(text)
                        .replaceAll(r -> ordinalWords(Integer.parseInt(r.group(1))));

        // 14. Plain numbers
        text =
                Pattern.compile("\\b\\d[\\d,]*\\b")
                        .matcher(text)
                        .replaceAll(
                                r -> {
                                    String v = r.group().replace(",", "");
                                    if (v.length() >= 5 && !v.startsWith("20"))
                                        return digitWords(v);
                                    return numberWords(Integer.parseInt(v));
                                });

        // 15. Uppercase acronyms
        text =
                Pattern.compile("\\b[A-Z]{2,}\\b")
                        .matcher(text)
                        .replaceAll(
                                r -> {
                                    var sb = new StringBuilder();
                                    for (char c : r.group().toCharArray())
                                        sb.append(
                                                        LETTERS.getOrDefault(
                                                                String.valueOf(c),
                                                                String.valueOf(c)))
                                                .append(' ');
                                    return sb.toString().trim();
                                });

        // 16. Cleanup spacing
        text = text.replaceAll(",(?:\\s*,)+", ",");
        text = text.replaceAll(",\\s*([.!?])", "$1");
        text = text.replaceAll("\\s+([,;:.!?])", "$1");
        text = text.replaceAll("([,;:.!?])(?=\\S)", "$1 ");
        return text.replaceAll("\\s+", " ").trim();
    }

    // ── number converters ──────────────────────────────────────────────────

    static String numberWords(int n) {
        if (n < 20) return ONES[n];
        if (n < 100) return TENS[n / 10] + (n % 10 > 0 ? " " + ONES[n % 10] : "");
        if (n < 1000) {
            int h = n / 100, r = n % 100;
            return ONES[h] + " hundred" + (r > 0 ? " " + numberWords(r) : "");
        }
        if (n < 1_000_000) {
            int th = n / 1000, r = n % 1000;
            return numberWords(th) + " thousand" + (r > 0 ? " " + numberWords(r) : "");
        }
        return digitWords(String.valueOf(n));
    }

    static String ordinalWords(int n) {
        String w = numberWords(n);
        String[] parts = w.split(" ");
        String last = parts[parts.length - 1];
        String replacement =
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
        parts[parts.length - 1] = replacement;
        return String.join(" ", parts);
    }

    static String digitWords(String s) {
        var sb = new StringBuilder();
        for (char c : s.toCharArray()) sb.append(ONES[c - '0']).append(' ');
        return sb.toString().trim();
    }

    static String expandIdent(String token) {
        Matcher m = Pattern.compile("([A-Za-z]?)(\\d+)([A-Za-z]?)").matcher(token);
        if (!m.matches()) return token;
        String prefix = m.group(1), digits = m.group(2), suffix = m.group(3);
        var sb = new StringBuilder();
        if (!prefix.isEmpty()) sb.append(LETTERS.get(prefix.toUpperCase())).append(' ');
        if (digits.length() == 3 || digits.startsWith("0")) {
            for (int i = 0; i < digits.length(); i++) {
                char c = digits.charAt(i);
                sb.append((c == '0' && i > 0) ? "oh " : ONES[c - '0'] + " ");
            }
        } else {
            sb.append(numberWords(Integer.parseInt(digits))).append(' ');
        }
        if (!suffix.isEmpty()) sb.append(LETTERS.get(suffix.toUpperCase())).append(' ');
        return sb.toString().trim();
    }

    // ── regex lambdas ──────────────────────────────────────────────────────

    private static String money(MatchResult r) {
        String raw = r.group(1).replace(",", "");
        String[] parts = raw.split("\\.");
        int dollars = Integer.parseInt(parts[0]);
        var sb =
                new StringBuilder(numberWords(dollars))
                        .append(dollars == 1 ? " dollar" : " dollars");
        if (parts.length > 1 && !parts[1].isEmpty()) {
            int cents = Integer.parseInt((parts[1] + "00").substring(0, 2));
            if (cents > 0)
                sb.append(" and ")
                        .append(numberWords(cents))
                        .append(cents == 1 ? " cent" : " cents");
        }
        return sb.toString();
    }

    private static String date(MatchResult r) {
        return MONTHS[Integer.parseInt(r.group(1)) - 1]
                + " "
                + ordinalWords(Integer.parseInt(r.group(2)))
                + " "
                + numberWords(Integer.parseInt(r.group(3)));
    }

    private static String time(MatchResult r) {
        int hour = Integer.parseInt(r.group(1)), min = Integer.parseInt(r.group(2));
        String suffix = r.group(3) != null ? r.group(3).replace(".", "").toLowerCase() : "";
        var sb = new StringBuilder(numberWords(hour));
        if (min == 0) sb.append(" o clock");
        else if (min < 10) sb.append(" oh ").append(numberWords(min));
        else sb.append(' ').append(numberWords(min));
        if (!suffix.isEmpty()) for (char c : suffix.toCharArray()) sb.append(' ').append(c);
        return sb.toString();
    }

    private static String bareTime(MatchResult r) {
        int hour = Integer.parseInt(r.group(1));
        String suffix = r.group(2).replaceAll("[^A-Za-z]", "").toLowerCase();
        var sb = new StringBuilder(numberWords(hour));
        for (char c : suffix.toCharArray()) sb.append(' ').append(c);
        return sb.toString();
    }
}
