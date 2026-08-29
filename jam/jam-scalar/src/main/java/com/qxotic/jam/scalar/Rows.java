package com.qxotic.jam.scalar;

import com.oracle.svm.shared.AlwaysInline;
import java.util.Arrays;

/**
 * A panel of same-length rows, one {@code float[]} each. The tile's loops index every array by the
 * same lane variable, which is what C2 needs to vectorize them (a flat {@code float[]} with per-row
 * offsets is left scalar - measured 6x slower); separate rows also let two panels be padded apart
 * so their stores and loads never false-alias 4 KiB apart in the store queue. Rows are kept and
 * grown on demand; {@link #fit} is the only allocation.
 */
final class Rows {

    private float[][] rows = new float[0][];

    /** The widest length fitted so far: every row holds at least this many. */
    private int len;

    /** Row {@code i}, at least as long as the last {@link #fit}. */
    @AlwaysInline("read per element in the pack and transpose loops")
    float[] row(int i) {
        return rows[i];
    }

    /** All rows, for the tile. */
    float[][] rows() {
        return rows;
    }

    /** Ensures {@code count} rows of at least {@code len} elements. */
    Rows fit(int count, int len) {
        if (rows.length < count || len > this.len) {
            this.len = Math.max(this.len, len);
            float[][] grown = Arrays.copyOf(rows, Math.max(count, rows.length));
            for (int i = 0; i < grown.length; i++)
                if (grown[i] == null || grown[i].length < pad(this.len, i))
                    grown[i] = new float[pad(this.len, i)];
            rows = grown;
        }
        return this;
    }

    /** Zeroes the first {@code len} elements of rows {@code from .. to}. */
    void zero(int from, int to, int len) {
        for (int i = from; i < to; i++) Arrays.fill(rows[i], 0, len, 0f);
    }

    /**
     * Row length for {@code len} used elements: varies with the row index so consecutive rows sit
     * at different offsets modulo 4 KiB.
     */
    private static int pad(int len, int index) {
        return len + 16 + 16 * (index & 3);
    }
}
