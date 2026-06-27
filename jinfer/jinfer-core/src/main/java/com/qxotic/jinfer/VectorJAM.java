package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jam.JAM;

import java.lang.foreign.MemorySegment;

/**
 * Java Vector API implementation of {@link JAM}: jinfer's per-dtype 512-bit register-tiled PREFILL gemm
 * (n &gt; 1), reached through the backend-agnostic JAM contract (native {@link MemorySegment}s + byte
 * offsets). It reconstructs the typed {@code FloatTensor} from each {@code (segment, dtype, offset)} and
 * dispatches to the same tuned kernels {@link VectorMatMul} uses — one source of truth for the kernels.
 *
 * <p><b>Specialization (the point of a Vector backend):</b> the register-tile shape is chosen once from the
 * CPU vector width AND the JIT in play — 4×4 (16 accumulators) on AVX-512 under HotSpot C2 or Graal ≥
 * jvmci-25.1; 3×2 on older Graal that caps vector allocation at zmm0–15; AVX2 2×4; ARM NEON; scalar floor.
 * Override with {@code -Djam.vector.tile} ({@code auto|4x4|3x2|2x8|8x2|1x1|avx256[-RxC]|neon[-RxC]|scalar}).
 *
 * <p>Decode ({@code n == 1}, bandwidth-bound) and non-tileable dtypes return {@link #EUNSUPPORTED} — the
 * caller's scalar floor ({@link ScalarJAM} / {@link ScalarMatMul}) handles them. The weight must be
 * contiguous ({@code ldw == k}); a strided weight also returns {@code EUNSUPPORTED}.
 */
final class VectorJAM implements JAM {

    // ---- specialization: register-tile selection, resolved once from -Djam.vector.tile + CPU width + JIT.
    //      The 7 tensor-class gemm kernels read VectorJAM.TILE_CODE (moved here from FloatTensor). ----
    static final String TILE = System.getProperty("jam.vector.tile",
            System.getProperty("jinfer.Q8_0GemmTile", "auto"));   // legacy name still honored
    /** Constant-foldable codes: 0=3x2,1=3x4,2=4x4,3=2x8,4=8x2,5=1x1,6..9=avx256,10/11=neon,12=scalar. */
    static final int TILE_CODE = switch (TILE) {
        case "auto" -> autoTileCode();
        case "3x2" -> 0; case "4x4" -> 2; case "2x8" -> 3; case "8x2" -> 4; case "1x1" -> 5;
        case "avx256", "avx256-2x4" -> 6; case "avx256-2x3" -> 7; case "avx256-3x4" -> 8; case "avx256-4x3" -> 9;
        case "neon", "neon-4x4" -> 10; case "neon-2x4" -> 11; case "scalar", "java" -> 12;
        default -> 1; // 3x4
    };

    private static int autoTileCode() {
        String arch = System.getProperty("os.arch", "").toLowerCase();
        if (arch.contains("aarch64") || arch.startsWith("arm")) return 10;   // ARM NEON 4x4
        int width = FloatTensor.USE_VECTOR_API ? FloatTensor.VECTOR_BIT_SIZE : 0;
        if (width >= 512) return jitHandlesWideTile() ? 2 /* 4x4 */ : 0 /* 3x2 */;
        if (width >= 256) return 6;   // AVX2 2x4
        return 12;                    // scalar
    }

    // 4x4 needs 16 accumulators in registers: Graal spill-free only from jvmci-25.1; HotSpot C2 spills but
    // they're bandwidth-hidden so 4x4 still wins; unknown VM -> safe 3x2.
    private static boolean jitHandlesWideTile() {
        String version = System.getProperty("java.vm.version", "");
        if (version.contains("jvmci")) {
            var v = java.util.regex.Pattern.compile("jvmci-(\\d+)\\.(\\d+)").matcher(version);
            if (v.find()) {
                int major = Integer.parseInt(v.group(1)), minor = Integer.parseInt(v.group(2));
                return major > 25 || (major == 25 && minor >= 1);
            }
            return false;   // "jvmci-bNN" (25.0-era Graal) caps at zmm0-15
        }
        String name = System.getProperty("java.vm.name", "");
        return name.contains("HotSpot") || name.contains("OpenJDK");
    }

    /** 512-bit prefill tile present — the precondition for every fast path here (JIT-folded constant). */
    static final boolean IS_512 = FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES.vectorBitSize() == 512;

    private static boolean tileable(GGMLType t) {
        return switch (t) {
            case Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, MXFP4, NVFP4 -> true;
            default -> false;   // F16/BF16/F32 -> dot floor
        };
    }

    @Override
    public int mm(MemorySegment w, long wOff, int wt, int ldw,
                  MemorySegment a, long aOff, int at, int lda,
                  MemorySegment r, long rOff, int rt, int ldr,
                  int m, int n, int k) {
        if (n <= 1 || at != F32 || rt != F32 || !IS_512) return EUNSUPPORTED;   // decode/non-F32 -> floor
        GGMLType t = GGMLType.fromId(wt);
        if (t == null || !tileable(t)) return EUNSUPPORTED;
        // weight rows must be contiguous (the kernels assume row stride = k) and a whole number of blocks.
        // (No check on wOff: it's an absolute byte address — block packing is a property of the data, not
        // of where the data sits in memory.)
        if (ldw != k || k % t.getElementsPerBlock() != 0) return EUNSUPPORTED;

        // Reconstruct typed tensors at the operand base (offset baked into the slice; the kernels assume
        // each operand starts at its tensor base, so we pass thisOffset = 0).
        SegmentFloatTensor weight = wrap(t, w.asSlice(wOff), (long) m * k);
        F32FloatTensor x = new F32FloatTensor((long) n * lda, a.asSlice(aOff));
        F32FloatTensor out = new F32FloatTensor((long) n * ldr, r.asSlice(rOff));
        switch (t) {
            case Q8_0 -> Q8_0FloatTensor.vectorGemm512F32((Q8_0FloatTensor) weight, x, out, lda, ldr, n, m, k, 0L);
            case Q4_0 -> Q4_0FloatTensor.vectorGemm512((Q4_0FloatTensor) weight, x, out, lda, ldr, n, m, k, 0L);
            case Q4_K -> Q4_KFloatTensor.vectorGemm512((Q4_KFloatTensor) weight, x, out, lda, ldr, n, m, k, 0L);
            case Q5_K -> Q5_KFloatTensor.vectorGemm512((Q5_KFloatTensor) weight, x, out, lda, ldr, n, m, k, 0L);
            case Q6_K -> Q6_KFloatTensor.vectorGemm512((Q6_KFloatTensor) weight, x, out, lda, ldr, n, m, k, 0L);
            case MXFP4 -> MXFP4FloatTensor.vectorGemmMxfp4((MXFP4FloatTensor) weight, x, out, lda, ldr, n, m, k, 0L);
            case NVFP4 -> NVFP4FloatTensor.vectorGemm512((NVFP4FloatTensor) weight, x, out, lda, ldr, n, m, k, 0L);
            default -> { return EUNSUPPORTED; }
        }
        return OK;
    }

    /** Wrap a native segment as the typed tensor for {@code t} (every JAM weight dtype). Shared with {@link ScalarJAM}. */
    static SegmentFloatTensor wrap(GGMLType t, MemorySegment seg, long size) {
        return switch (t) {
            case Q8_0 -> new Q8_0FloatTensor(size, seg);
            case Q4_0 -> new Q4_0FloatTensor(size, seg);
            case Q4_K -> new Q4_KFloatTensor(size, seg);
            case Q5_K -> new Q5_KFloatTensor(size, seg);
            case Q6_K -> new Q6_KFloatTensor(size, seg);
            case MXFP4 -> new MXFP4FloatTensor(size, seg);
            case NVFP4 -> new NVFP4FloatTensor(size, seg);
            case F16 -> new F16FloatTensor(size, seg);
            case BF16 -> new BF16FloatTensor(size, seg);
            case F32 -> new F32FloatTensor(size, seg);
            default -> throw new IllegalArgumentException("VectorJAM/ScalarJAM: no tensor for " + t);
        };
    }
}
