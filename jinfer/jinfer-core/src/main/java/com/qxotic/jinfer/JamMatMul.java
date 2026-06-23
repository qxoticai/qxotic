package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jam.JAM;

import java.lang.foreign.MemorySegment;

/**
 * Native backend: passes the operands straight to {@code JAM.mm}. jam carries the (dtype × ISA) dispatch
 * itself and writes token-major output (jinfer's layout), so this is just address + tag plumbing.
 * Offsets are handled in {@code long} — the weight byte address can exceed 2³¹ on large models.
 */
final class JamMatMul implements MatMul {

    static boolean tryLoad() {
        try { Class.forName("com.qxotic.jam.JAM"); return true; }   // triggers libjam load
        catch (Throwable t) {
            System.err.println("jam native library unavailable (" + t + "); using the Java backends.");
            return false;
        }
    }

    @Override
    public boolean mm(FloatTensor w, long wOff, int wStride,
                      FloatTensor a, long aOff, int aStride,
                      FloatTensor c, long cOff, int cStride,
                      int m, int n, int k) {
        GGMLType t = w.type();
        long wBase = ((SegmentFloatTensor) w).baseAddress;
        long aBase = ((SegmentFloatTensor) a).baseAddress;
        long cBase = ((SegmentFloatTensor) c).baseAddress;
        long wa = wBase + (wOff / t.getElementsPerBlock()) * t.getBlockByteSize();
        int st = JAM.mm(wa, jamTag(t), wStride,
                        aBase + aOff * Float.BYTES, JAM.F32, aStride,
                        cBase + cOff * Float.BYTES, JAM.F32, cStride,
                        m, n, k);
        return st == JAM.OK;   // EBUSY / EUNSUPPORTED -> Dispatch falls through to a Java backend
    }

    /** GGMLType -> jam dtype tag (== ggml_type value, but mapped explicitly to stay honest). */
    static int jamTag(GGMLType t) {
        return switch (t) {
            case Q8_0 -> JAM.Q8_0;
            case Q4_0 -> JAM.Q4_0;
            case Q4_K -> JAM.Q4_K;
            case Q5_K -> JAM.Q5_K;
            case Q6_K -> JAM.Q6_K;
            case MXFP4 -> JAM.MXFP4;
            case F16 -> JAM.F16;
            case BF16 -> JAM.BF16;
            case F32 -> JAM.F32;
            default -> throw new IllegalArgumentException("jam has no kernel for " + t);
        };
    }
}
