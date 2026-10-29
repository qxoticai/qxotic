package com.qxotic.toknroll.libtoknroll;

import java.util.Arrays;

/** How a batch is shared out: runs of consecutive texts, one task each. */
final class Runs {
    private Runs() {}

    /**
     * Text indexes, from 0 to {@code count}, that cut a batch into at most {@code k} runs of about
     * equal bytes, text i spanning {@code [starts[i], starts[i + 1])}. A run is empty only when the
     * batch is; empty texts ride along with their neighbours.
     */
    static int[] of(int[] starts, int count, int k) {
        int m = Math.max(1, Math.min(k, count));
        long total = starts[count];
        int[] cuts = new int[m + 1];
        int r = 1;
        for (int i = 1; i < count && r < m; i++) {
            if ((long) starts[i] * m >= total * r) cuts[r++] = i; // text i starts past share r
        }
        cuts[r] = count;
        return Arrays.copyOf(cuts, r + 1);
    }
}
