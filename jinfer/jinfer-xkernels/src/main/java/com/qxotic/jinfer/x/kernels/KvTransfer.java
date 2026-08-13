package com.qxotic.jinfer.x.kernels;

import com.qxotic.jinfer.x.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;

/**
 * Direction-flagged bulk copies between resume-state storage and a cache blob, ported from
 * jinfer-core {@code KvTransfer}: tensor-side access is a raw {@code MemorySegment.copy} at the
 * view's native encoding (FP16 KV = 2 bytes/element). Codecs with a branching layout walk
 * (windowed/recurrent layers) drive save AND restore through one walk built on these, so the blob
 * layout is single-sourced — layout drift between the two mirrored directions is the structural
 * risk in a codec.
 */
public final class KvTransfer {

    private KvTransfer() {}

    /** View elements ↔ blob bytes at the view's native encoding; returns the bytes moved. */
    public static long transfer(
            MemoryView<MemorySegment> t,
            long elemOff,
            MemorySegment blob,
            long byteOff,
            long elems,
            boolean out) {
        return transfer(t, DataType.FP16, elemOff, blob, byteOff, elems, out);
    }

    /** Dense view elements ↔ blob bytes at the view's native encoding. */
    public static long transfer(
            MemoryView<MemorySegment> t,
            DataType dtype,
            long elemOff,
            MemorySegment blob,
            long byteOff,
            long elems,
            boolean out) {
        Views.requireDense(t, dtype, "state");
        Views.checkAlive(t, "state");
        if (elemOff < 0 || elems < 0 || elemOff > t.shape().size() - elems)
            throw new IndexOutOfBoundsException(
                    "state elements ["
                            + elemOff
                            + ","
                            + Math.addExact(elemOff, elems)
                            + ") out of "
                            + t.shape().size());
        long elemBytes = dtype.byteSize();
        long srcByte = Math.addExact(t.byteOffset(), Math.multiplyExact(elemOff, elemBytes));
        long bytes = Math.multiplyExact(elems, elemBytes);
        if (byteOff < 0 || byteOff > blob.byteSize() - bytes)
            throw new IndexOutOfBoundsException(
                    "blob bytes ["
                            + byteOff
                            + ","
                            + Math.addExact(byteOff, bytes)
                            + ") out of "
                            + blob.byteSize());
        if (out) {
            MemorySegment.copy(t.memory().base(), srcByte, blob, byteOff, bytes);
        } else {
            MemorySegment.copy(blob, byteOff, t.memory().base(), srcByte, bytes);
        }
        return bytes;
    }

    /**
     * Per-position rows of a sliding-window layer, stored through their ring slots {@code pos &
     * (w-1)}: one blob row per position of {@code [from,to)}, walked as contiguous slot runs.
     *
     * <p>Aliasing is safe by construction. A span longer than {@code w} reads a NEWER aliased row
     * for its dead positions, but every row a resume can ever need - the last {@code w} before any
     * block end - was live when its block was saved; and restore replays chains in ascending
     * position order, so aliased slot writes end with the newest (correct) row. Returns the bytes
     * covered: {@code (to-from) * rowElems * 2} (F16).
     */
    public static long ringSpan(
            MemoryView<MemorySegment> ring,
            int from,
            int to,
            int w,
            long rowElems,
            MemorySegment blob,
            long byteOff,
            boolean out) {
        long off = byteOff;
        int done = 0, n = to - from;
        while (done < n) {
            int slot = (from + done) & (w - 1);
            int run = Math.min(n - done, w - slot); // stop at the ring edge
            off += transfer(ring, slot * rowElems, blob, off, run * rowElems, out);
            done += run;
        }
        return off - byteOff;
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
