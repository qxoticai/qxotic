package com.qxotic.jinfer.models.lfm3;

import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;

/**
 * What LFM3's stacks share: the per-head norm and RoPE, the SwiGLU feed-forward, ggml's F16
 * roundings and the 3x3 conv tap layout.
 */
final class Towers {

    private Towers() {}

    /**
     * Per-head RMS norm, then NeoX RoPE at table row {@code row}, over {@code heads} heads of
     * {@code headDim} starting at {@code at}.
     */
    static void headNormRope(
            MemoryView<MemorySegment> t,
            long at,
            int heads,
            int headDim,
            MemoryView<MemorySegment> norm,
            float eps,
            int row,
            MemoryView<MemorySegment> cos,
            MemoryView<MemorySegment> sin) {
        for (int head = 0; head < heads; head++) {
            long offset = at + (long) head * headDim;
            Norms.rmsnormGgml(t, offset, t, offset, norm, headDim, eps);
            RoPE.applyNeox(t, offset, row, cos, sin, headDim / 2);
        }
    }

    /** The SwiGLU feed-forward over {@code rows}: {@code out = down(silu(gate x) * up x)}. */
    static void ffn(
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> up,
            MemoryView<MemorySegment> down,
            MemoryView<MemorySegment> x,
            MemoryView<MemorySegment> hidden,
            MemoryView<MemorySegment> hidden2,
            MemoryView<MemorySegment> out,
            int rows,
            int ffnDim) {
        MatMul.gemm(gate, x, hidden, rows);
        MatMul.gemm(up, x, hidden2, rows);
        Activations.siluMultiplyRows(hidden, hidden2, rows, ffnDim);
        MatMul.gemm(down, hidden, out, rows);
    }

    /** {@code values} rounded to F16 in place: ggml_conv_2d's im2col stores its input so. */
    static float[] roundToHalf(float[] values) {
        for (int i = 0; i < values.length; i++)
            values[i] = Float.float16ToFloat(Float.floatToFloat16(values[i]));
        return values;
    }

    /**
     * {@code t[0, n)} rounded to F16 in place, through {@code half}: ggml's flash attention reads K
     * and V in F16, and F32 values that are F16 keep jinfer's attention on its fast F32 path.
     */
    static void roundToHalf(MemoryView<MemorySegment> t, MemoryView<MemorySegment> half, int n) {
        Convert.f32ToF16(t, 0, half, 0, n);
        Convert.f16ToF32(half, 0, t, 0, n);
    }

    /**
     * 3x3 conv taps from the GGUF's {@code [channels][3][3]} to {@code [3 * 3][channels]}: one
     * tap's channels contiguous, the loop a convolution vectorizes.
     */
    static float[] tapMajor(float[] taps, int channels) {
        float[] out = new float[taps.length];
        for (int c = 0; c < channels; c++)
            for (int k = 0; k < 9; k++) out[k * channels + c] = taps[c * 9 + k];
        return out;
    }
}
