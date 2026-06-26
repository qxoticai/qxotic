package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jam.JAM;
import com.qxotic.jam.NativeJAM;

import java.lang.foreign.MemorySegment;

/**
 * Native backend: passes the operands to {@code NativeJAM.global().mm}. jam carries the (dtype × ISA) dispatch
 * itself and writes token-major output (jinfer's layout), so this is just segment + offset + tag plumbing.
 * Each operand goes in as {@code (vseg, vbase + relativeByteOffset)} — {@code vseg.address() + vbase} is the
 * tensor's base, so jam reads it zero-copy and bounds-checks against {@code vseg}. Offsets are {@code long}
 * (a weight byte offset can exceed 2³¹ on large models).
 */
final class JamMatMul implements MatMul {

    private final MatMul fallback;   // scalar floor, for a runtime decline (EBUSY contention / unsupported)

    JamMatMul(MatMul fallback) { this.fallback = fallback; }

    static boolean tryLoad() {
        if (Boolean.getBoolean("jinfer.disableJam")) return false;       // force the Java backends (testing)
        try { Class.forName("com.qxotic.jam.NativeJAM"); return true; }  // triggers libjam load (NativeJAM static init)
        catch (Throwable t) {
            System.err.println("jam native library unavailable (" + t + "); using the Java backends.");
            return false;
        }
    }

    @Override
    public void mm(FloatTensor w, long wOff, int wStride,
                   FloatTensor a, long aOff, int aStride,
                   FloatTensor c, long cOff, int cStride,
                   int m, int n, int k) {
        GGMLType t = w.type();
        SegmentFloatTensor sw = (SegmentFloatTensor) w, sa = (SegmentFloatTensor) a, sc = (SegmentFloatTensor) c;
        long wByteOff = sw.vbase + (wOff / t.getElementsPerBlock()) * t.getBlockByteSize();   // block-aware weight offset
        long aByteOff = sa.vbase + aOff * Float.BYTES;
        long cByteOff = sc.vbase + cOff * Float.BYTES;
        int st = NativeJAM.global().mm(sw.vseg, wByteOff, jamTag(t), wStride,
                                       sa.vseg, aByteOff, JAM.F32, aStride,
                                       sc.vseg, cByteOff, JAM.F32, cStride,
                                       m, n, k);
        if (st != JAM.OK) {   // EBUSY (concurrent same-context) / unsupported shape -> the floor finishes it
            fallback.mm(w, wOff, wStride, a, aOff, aStride, c, cOff, cStride, m, n, k);
        }
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
            case NVFP4 -> JAM.NVFP4;
            case F16 -> JAM.F16;
            case BF16 -> JAM.BF16;
            case F32 -> JAM.F32;
            default -> throw new IllegalArgumentException("jam has no kernel for " + t);
        };
    }
}
