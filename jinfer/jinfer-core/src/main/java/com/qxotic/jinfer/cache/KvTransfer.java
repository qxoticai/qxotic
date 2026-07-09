package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.FloatTensor;
import java.lang.foreign.MemorySegment;

/**
 * Direction-flagged bulk copies between resume-state storage and a cache blob. Codecs with a
 * branching layout walk (windowed/recurrent layers) drive save AND restore through one walk built
 * on these, so the blob layout is single-sourced — layout drift between the two mirrored directions
 * is the structural risk in a codec.
 */
public final class KvTransfer {

    private KvTransfer() {}

    /** Tensor elements ↔ blob bytes at the tensor's native encoding; returns the bytes moved. */
    public static long transfer(
            FloatTensor t,
            long elemOff,
            MemorySegment blob,
            long byteOff,
            long elems,
            boolean out) {
        return out
                ? t.copyRawTo(elemOff, blob, byteOff, elems)
                : t.copyRawFrom(blob, byteOff, elemOff, elems);
    }

    /**
     * Fixed-size sliding-window checkpoint: rows {@code [max(0,to-w), to)} live at ring slots
     * {@code pos & (w-1)} (the span wraps at most once, so at most two contiguous runs), padded to
     * {@code w} rows in the blob. {@code elemBytes} is the ring's native element width (2 for F16).
     * Returns the bytes covered — always {@code w * rowElems * elemBytes}, pad included — so save
     * and restore advance identically.
     */
    public static long window(
            FloatTensor ring,
            int to,
            int w,
            long rowElems,
            int elemBytes,
            MemorySegment blob,
            long byteOff,
            boolean out) {
        int lo = Math.max(0, to - w);
        int n = to - lo;
        long off = byteOff;
        int done = 0;
        while (done < n) {
            int slot = (lo + done) & (w - 1);
            int run = Math.min(n - done, w - slot); // stop at the ring edge
            off += transfer(ring, slot * rowElems, blob, off, run * rowElems, out);
            done += run;
        }
        return (long) w * rowElems * elemBytes; // rows moved + pad
    }

    /**
     * Heap {@code float[]} ↔ blob bytes (raw byte copy, native order) — for recurrent state
     * matrices that live as plain arrays; returns the bytes moved.
     */
    public static long transfer(float[] a, MemorySegment blob, long byteOff, boolean out) {
        MemorySegment heap = MemorySegment.ofArray(a);
        long bytes = a.length * 4L;
        if (out) {
            MemorySegment.copy(heap, 0, blob, byteOff, bytes);
        } else {
            MemorySegment.copy(blob, byteOff, heap, 0, bytes);
        }
        return bytes;
    }
}
