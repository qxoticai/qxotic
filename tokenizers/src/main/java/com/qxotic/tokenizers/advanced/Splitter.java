package com.qxotic.tokenizers.advanced;

import com.qxotic.tokenizers.impl.RegexSplitter;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.regex.Pattern;

@FunctionalInterface
public interface Splitter {

    List<CharSequence> split(CharSequence text);

    /** Strict default: single chunk, no rewrite. */
    static Splitter identity() {
        return List::of;
    }

    static Splitter regex(Pattern pattern) {
        Objects.requireNonNull(pattern, "pattern");
        return RegexSplitter.create(pattern);
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
