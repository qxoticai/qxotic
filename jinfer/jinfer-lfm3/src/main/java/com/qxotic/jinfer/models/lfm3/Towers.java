package com.qxotic.jinfer.models.lfm3;

import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;

/** What the vision and audio towers share: ggml's F16 roundings and the 3x3 conv tap layout. */
final class Towers {

    private Towers() {}

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
