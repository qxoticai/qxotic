package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.FloatTensor;

import java.lang.foreign.MemorySegment;

/** Direction-flagged bulk copies between resume-state storage and a cache blob. Codecs with a
 *  branching layout walk (windowed/recurrent layers) drive save AND restore through one walk
 *  built on these, so the blob layout is single-sourced — layout drift between the two mirrored
 *  directions is the structural risk in a codec. */
public final class KvTransfer {

    private KvTransfer() {}

    /** Tensor elements ↔ blob bytes at the tensor's native encoding; returns the bytes moved. */
    public static long transfer(FloatTensor t, long elemOff, MemorySegment blob, long byteOff, long elems, boolean out) {
        return out ? t.copyRawTo(elemOff, blob, byteOff, elems)
                   : t.copyRawFrom(blob, byteOff, elemOff, elems);
    }

    /** Heap {@code float[]} ↔ blob bytes (raw byte copy, native order) — for recurrent state
     *  matrices that live as plain arrays; returns the bytes moved. */
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
