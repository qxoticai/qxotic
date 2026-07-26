package com.qxotic.jinfer.chat;

import java.util.List;

/**
 * Sink-side stop-sequence matcher over the reply's CONTENT lane (text-level: stop strings rarely
 * align with token boundaries). Streaming emission holds back longest-stop-minus-one chars so a
 * stop straddling fragments is never partially emitted; the holdback also guarantees a match never
 * starts before already-emitted text, so streamed partials always concatenate to {@link
 * #beforeCut}.
 */
public final class StopSequences {

    private final List<String> stops;
    private final int holdback;
    private final StringBuilder text = new StringBuilder();
    private int emitted;
    private int cut = -1;

    private StopSequences(List<String> stops) {
        this.stops = stops;
        int longest = 0;
        for (String s : stops) longest = Math.max(longest, s.length());
        this.holdback = longest - 1;
    }

    /** null when there is nothing to match - the common case costs nothing. */
    public static StopSequences of(List<String> stops) {
        return stops == null || stops.isEmpty() ? null : new StopSequences(stops);
    }

    /** Feeds a content fragment; returns the chars now safe to emit ("" while held back). */
    public String feed(String fragment) {
        text.append(fragment);
        if (cut < 0) {
            int from = Math.max(0, text.length() - fragment.length() - holdback);
            for (String s : stops) {
                int at = text.indexOf(s, from);
                if (at >= 0 && (cut < 0 || at < cut)) cut = at;
            }
        }
        int safe = cut >= 0 ? cut : Math.max(emitted, text.length() - holdback);
        String out = text.substring(emitted, Math.max(emitted, safe));
        emitted = Math.max(emitted, safe);
        return out;
    }

    /** The held-back tail once generation ends: everything before the cut, nothing after it. */
    public String flush() {
        int end = cut >= 0 ? cut : text.length();
        String out = text.substring(Math.min(emitted, end), end);
        emitted = Math.max(emitted, end);
        return out;
    }

    public boolean hit() {
        return cut >= 0;
    }

    /** The whole content up to the first stop - the text a hit trims the reply to. */
    public String beforeCut() {
        return cut >= 0 ? text.substring(0, cut) : text.toString();
    }
}
