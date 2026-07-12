package com.qxotic.jinfer.llm;

import java.util.List;
import java.util.function.Consumer;

/**
 * String-level stop handling, shared by every text-producing frontend: {@link #apply} truncates a
 * finished text at the first configured stop string; {@link Holdback} does the same for a live
 * stream, holding back any suffix that could still grow into a stop so a stop string is never
 * emitted downstream. Pure string machinery - stops are matched against whatever text the caller
 * assembles (the chat layer's decoder output, or raw completions text).
 */
public final class TextStops {

    private TextStops() {}

    public record Result(String text, boolean stopped) {}

    /** The text truncated at the earliest stop-string occurrence, flagged when one matched. */
    public static Result apply(String text, List<String> stops) {
        int cut = -1;
        for (String stop : stops) {
            int index = text.indexOf(stop);
            if (index >= 0 && (cut < 0 || index < cut)) cut = index;
        }
        return cut >= 0 ? new Result(text.substring(0, cut), true) : new Result(text, false);
    }

    /**
     * Forwards text downstream while holding back any suffix that could grow into a configured stop
     * string; once a stop string appears the text before it is emitted and the consumer goes
     * silent.
     */
    public static final class Holdback implements Consumer<String> {
        private final List<String> stops;
        private final Consumer<String> downstream;
        private final StringBuilder pending = new StringBuilder();
        private boolean stopped;

        public Holdback(List<String> stops, Consumer<String> downstream) {
            this.stops = stops;
            this.downstream = downstream;
        }

        @Override
        public void accept(String text) {
            if (stopped || text.isEmpty()) return;
            pending.append(text);
            Result result = apply(pending.toString(), stops);
            if (result.stopped()) {
                emit(result.text());
                pending.setLength(0);
                stopped = true;
                return;
            }
            int keep = longestStopPrefixSuffix(pending, stops);
            int emitLength = pending.length() - keep;
            if (emitLength > 0) {
                emit(pending.substring(0, emitLength));
                pending.delete(0, emitLength);
            }
        }

        private void emit(String text) {
            if (!text.isEmpty()) downstream.accept(text);
        }

        public void flush() {
            if (!stopped && !pending.isEmpty()) {
                emit(pending.toString());
                pending.setLength(0);
            }
        }

        public boolean stopped() {
            return stopped;
        }

        private static int longestStopPrefixSuffix(StringBuilder text, List<String> stops) {
            int keep = 0;
            String current = text.toString();
            for (String stop : stops) {
                int max = Math.min(stop.length() - 1, current.length());
                for (int len = max; len > keep; len--) {
                    if (current.endsWith(stop.substring(0, len))) {
                        keep = len;
                        break;
                    }
                }
            }
            return keep;
        }
    }
}
