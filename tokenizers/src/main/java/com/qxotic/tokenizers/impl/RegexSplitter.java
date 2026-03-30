package com.qxotic.tokenizers.impl;

import com.qxotic.tokenizers.advanced.Splitter;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class RegexSplitter implements Splitter {

    private final Pattern pattern;

    private RegexSplitter(Pattern pattern) {
        this.pattern = pattern;
    }

    public static RegexSplitter create(Pattern pattern) {
        return new RegexSplitter(pattern);
    }

    public static RegexSplitter create(String regexPattern) {
        return new RegexSplitter(Pattern.compile(regexPattern));
    }

    @Override
    public List<CharSequence> split(CharSequence text) {
        return findAll(pattern, text);
    }

    private static List<CharSequence> findAll(Pattern pattern, CharSequence text) {
        List<CharSequence> allMatches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        int lastEnd = 0;

        while (matcher.find()) {
            if (matcher.start() > lastEnd) {
                allMatches.add(text.subSequence(lastEnd, matcher.start()));
            }
            CharSequence subSequence = text.subSequence(matcher.start(), matcher.end());
            assert CharSequence.compare(subSequence, matcher.group()) == 0;
            allMatches.add(subSequence);
            lastEnd = matcher.end();
        }

        if (lastEnd < text.length()) {
            allMatches.add(text.subSequence(lastEnd, text.length()));
        }

        return allMatches;
    }
}
