package com.qxotic.toknroll.testkit;

import com.qxotic.toknroll.Splitter;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class SplitterTestUtils {
    private SplitterTestUtils() {}

    public static List<CharSequence> splitAllToList(Splitter splitter, CharSequence text) {
        Objects.requireNonNull(splitter, "splitter");
        Objects.requireNonNull(text, "text");
        List<CharSequence> chunks = new ArrayList<>();
        splitter.splitAll(
                text,
                (source, startInclusive, endExclusive) ->
                        chunks.add(source.subSequence(startInclusive, endExclusive)));
        return chunks;
    }
}
