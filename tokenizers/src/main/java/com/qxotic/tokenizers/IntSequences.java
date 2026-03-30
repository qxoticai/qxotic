package com.qxotic.tokenizers;

import java.util.Objects;

/** Utility algorithms for {@link IntSequence}. */
public final class IntSequences {
    private IntSequences() {}

    /** Concatenates all provided sequences in order. */
    public static IntSequence concatAll(IntSequence... sequences) {
        Objects.requireNonNull(sequences, "sequences");
        if (sequences.length == 0) {
            return IntSequence.empty();
        }
        int total = 0;
        for (IntSequence sequence : sequences) {
            total = Math.addExact(total, Objects.requireNonNull(sequence, "sequence").length());
        }
        if (total == 0) {
            return IntSequence.empty();
        }
        int[] merged = new int[total];
        int offset = 0;
        for (IntSequence sequence : sequences) {
            int length = sequence.length();
            sequence.copyTo(merged, offset, length);
            offset += length;
        }
        return IntSequence.wrap(merged);
    }

    /** Concatenates at least two sequences in order. */
    public static IntSequence concat(IntSequence first, IntSequence second, IntSequence... rest) {
        Objects.requireNonNull(first, "first");
        Objects.requireNonNull(second, "second");
        Objects.requireNonNull(rest, "rest");
        IntSequence[] all = new IntSequence[rest.length + 2];
        all[0] = first;
        all[1] = second;
        for (int i = 0; i < rest.length; i++) {
            all[i + 2] = rest[i];
        }
        return concatAll(all);
    }

    /** Compares two sequences for content equality. */
    public static boolean contentEquals(IntSequence first, IntSequence second) {
        if (Objects.requireNonNull(first) == Objects.requireNonNull(second)) {
            return true;
        }
        int length = first.length();
        if (length != second.length()) {
            return false;
        }
        for (int i = 0; i < length; i++) {
            if (first.intAt(i) != second.intAt(i)) {
                return false;
            }
        }
        return true;
    }

    /** Compares two sequences lexicographically. */
    public static int compare(IntSequence first, IntSequence second) {
        if (Objects.requireNonNull(first) == Objects.requireNonNull(second)) {
            return 0;
        }
        int commonLength = Math.min(first.length(), second.length());
        for (int i = 0; i < commonLength; i++) {
            int fi = first.intAt(i);
            int si = second.intAt(i);
            if (fi != si) {
                return Integer.compare(fi, si);
            }
        }
        return Integer.compare(first.length(), second.length());
    }
}
