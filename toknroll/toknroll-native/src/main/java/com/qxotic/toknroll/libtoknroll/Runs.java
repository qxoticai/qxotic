package com.qxotic.toknroll.libtoknroll;

import java.util.Arrays;

/** How a batch is shared out: runs of consecutive texts, one task each. */
final class Runs {
    private Runs() {}

    /**
     * Text indexes, from {@code first} to {@code last}, that cut texts {@code [first, last)} into
     * at most {@code k} runs of about equal bytes, text i spanning {@code [starts[i], starts[i +
     * 1])}. A run is empty only when the range is; empty texts ride along with their neighbours.
     */
    static int[] of(long[] starts, int first, int last, int k) {
        int m = Math.max(1, Math.min(k, last - first));
        long total = starts[last] - starts[first];
        int[] cuts = new int[m + 1];
        cuts[0] = first;
        int r = 1;
        for (int i = first + 1; i < last && r < m; i++) {
            if ((starts[i] - starts[first]) * m >= total * r) cuts[r++] = i; // past share r
        }
        cuts[r] = last;
        return Arrays.copyOf(cuts, r + 1);
    }
}
