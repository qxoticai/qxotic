package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class SpecialsImpl implements Specials {
    private static final SpecialsImpl NONE = new SpecialsImpl(Set.of(), Map.of(), null);

    private final Set<String> tokens;
    private final Map<String, Integer> specialToId;
    private final Pattern pattern;

    private SpecialsImpl(Set<String> tokens, Map<String, Integer> specialToId, Pattern pattern) {
        this.tokens = tokens;
        this.specialToId = specialToId;
        this.pattern = pattern;
    }

    public static Specials none() {
        return NONE;
    }

    public static Specials compile(Vocabulary vocabulary, Set<String> specials) {
        if (specials.isEmpty()) {
            return NONE;
        }

        List<String> sorted = new ArrayList<>(specials.size());
        for (String special : specials) {
            if (special == null) {
                throw new IllegalArgumentException("special token cannot be null");
            }
            if (special.isEmpty()) {
                throw new IllegalArgumentException("special token cannot be empty");
            }
            if (!vocabulary.contains(special)) {
                throw new IllegalArgumentException(
                        "special token not present in vocabulary: " + special);
            }
            sorted.add(special);
        }
        Collections.sort(sorted);
        validateNoPrefixConflicts(sorted);

        Map<String, Integer> specialToId = new LinkedHashMap<>(sorted.size());
        for (String special : sorted) {
            specialToId.put(special, vocabulary.id(special));
        }
        Set<String> tokens = Collections.unmodifiableSet(new LinkedHashSet<>(sorted));
        Pattern pattern = Pattern.compile(buildAlternationPattern(sorted));
        return new SpecialsImpl(tokens, Collections.unmodifiableMap(specialToId), pattern);
    }

    private static void validateNoPrefixConflicts(List<String> sorted) {
        for (int i = 1; i < sorted.size(); i++) {
            String previous = sorted.get(i - 1);
            String current = sorted.get(i);
            if (current.startsWith(previous)) {
                throw new IllegalArgumentException(
                        "special token prefix conflict: '"
                                + previous
                                + "' is a prefix of '"
                                + current
                                + "'");
            }
        }
    }

    private static String buildAlternationPattern(List<String> specials) {
        // Use a non-capturing group so alternation applies as one unit without creating
        // unnecessary capture groups.
        StringBuilder sb = new StringBuilder();
        sb.append("(?:");
        for (int i = 0; i < specials.size(); i++) {
            if (i != 0) {
                sb.append('|');
            }
            sb.append(Pattern.quote(specials.get(i)));
        }
        sb.append(')');
        return sb.toString();
    }

    @Override
    public Set<String> tokens() {
        return tokens;
    }

    @Override
    public void encodeInto(Tokenizer tokenizer, CharSequence text, IntSequence.Builder out) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(out, "out");

        if (specialToId.isEmpty()) {
            tokenizer.encodeInto(text, out);
            return;
        }

        Matcher matcher = pattern.matcher(text);
        int cursor = 0;
        while (matcher.find()) {
            int start = matcher.start();
            int end = matcher.end();
            if (start > cursor) {
                tokenizer.encodeInto(text, cursor, start, out);
            }
            String matched = matcher.group();
            Integer tokenId = specialToId.get(matched);
            if (tokenId == null) {
                throw new IllegalStateException("Unknown compiled special token: " + matched);
            }
            out.add(tokenId);
            cursor = end;
        }
        if (cursor < text.length()) {
            tokenizer.encodeInto(text, cursor, text.length(), out);
        }
    }
}
