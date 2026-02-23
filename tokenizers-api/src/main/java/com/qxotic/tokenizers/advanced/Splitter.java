package com.qxotic.tokenizers.advanced;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@FunctionalInterface
public interface Splitter {

    List<CharSequence> split(CharSequence text);

    /** Strict default: single chunk, no rewrite. */
    Splitter IDENTITY = List::of;

    /** Strict default: single chunk, no rewrite. */
    static Splitter identity() {
        return IDENTITY;
    }

    static Splitter regex(Pattern pattern) {
        Objects.requireNonNull(pattern, "pattern");
        return text -> {
            List<CharSequence> allMatches = new ArrayList<>();
            Matcher matcher = pattern.matcher(text);
            int lastEnd = 0;

            while (matcher.find()) {
                if (matcher.start() > lastEnd) {
                    allMatches.add(text.subSequence(lastEnd, matcher.start()));
                }
                allMatches.add(text.subSequence(matcher.start(), matcher.end()));
                lastEnd = matcher.end();
            }

            if (lastEnd < text.length()) {
                allMatches.add(text.subSequence(lastEnd, text.length()));
            }

            return allMatches;
        };
    }

    static Splitter regex(String regexPattern) {
        Objects.requireNonNull(regexPattern, "regexPattern");
        return regex(Pattern.compile(regexPattern));
    }

    static Splitter sequence(Splitter... splitters) {
        if (splitters.length == 0) {
            return identity();
        }
        return text -> {
            List<CharSequence> current = List.of(text);
            for (Splitter splitter : splitters) {
                List<CharSequence> next = new ArrayList<>();
                for (CharSequence chunk : current) {
                    next.addAll(splitter.split(chunk));
                }
                current = next;
            }
            return current;
        };
    }
}
