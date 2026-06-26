package com.qxotic.jam;

import java.lang.foreign.MemorySegment;

/**
 * jam — fast multithreaded CPU matmul ({@code R = W @ Aᵀ}) with quantized weights.
 *
 * <p>{@code JAM} is the matmul contract. The native ({@code libjam}) implementation is {@link NativeJAM}
 * (via {@link NativeJAM#global()}); other backends (e.g. a Vector API impl) implement this interface.
 * Operands are native {@link MemorySegment}s + byte offsets; an implementation is responsible for its own
 * bounds/liveness handling (the native impl bounds-checks and rejects heap segments).
 */
public interface JAM {

    // ── supported weight dtype tags: numerically identical to GGML's ggml_type ──
    int F32 = 0, F16 = 1, BF16 = 30,
        Q4_0 = 2, Q8_0 = 8,
        Q4_K = 12, Q5_K = 13, Q6_K = 14,
        MXFP4 = 39, NVFP4 = 40;

    // ── jam_status ──
    int OK = 0, EINVAL = 1, EUNSUPPORTED = 2, EBUSY = 3;

    /**
     * {@code R = W @ Aᵀ}. Each operand is a native {@link MemorySegment} + BYTE offset; {@code ldw/lda/ldr}
     * are ELEMENT row strides; {@code wt/at/rt} the operand dtypes ({@code at}, {@code rt} are {@code F32}
     * today). Returns a jam_status ({@link #OK} / {@link #EINVAL} / {@link #EUNSUPPORTED} / {@link #EBUSY}).
     */
    int mm(MemorySegment w, long wOff, int wt, int ldw,
           MemorySegment a, long aOff, int at, int lda,
           MemorySegment r, long rOff, int rt, int ldr,
           int m, int n, int k);

    /** Contiguous shortcut — offsets 0, strides {@code k/k/m}, F32 activations + result. */
    default int mm(MemorySegment w, MemorySegment a, MemorySegment r, int wt, int m, int n, int k) {
        return mm(w, 0, wt, k, a, 0, F32, k, r, 0, F32, m, m, n, k);
    }
}
