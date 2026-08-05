package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.kernels.GGMLTensorEntry;
import com.qxotic.jinfer.kernels.ModelLoader;
import java.util.Map;

/**
 * A projection with the mmproj's QAT calibration clamps - the reference's {@code build_mm}: input
 * clamps to {@code [inMin, inMax]} before the matmul, output to {@code [outMin, outMax]} after. One
 * shape for every Gemma 4 tower (the ViT and the audio Conformer carry the same {@code
 * <base>.input_min/max} / {@code output_min/max} scalar tensors; a weight without them clamps
 * nothing). Clamp passes are vectorized ({@code copyTo} + {@code clampInPlace}); the input never
 * mutates - several projections may share one normalized input buffer.
 */
record Clamped(FloatTensor w, float inMin, float inMax, float outMin, float outMax) {

    /** Loads {@code <base>.weight} (size-asserted) plus its optional calibration scalars. */
    static Clamped load(Map<String, GGMLTensorEntry> t, String base, long expectedElements) {
        GGMLTensorEntry e = t.get(base + ".weight");
        if (e == null) {
            throw new IllegalStateException("mmproj tensor missing: " + base + ".weight");
        }
        FloatTensor w = ModelLoader.loadQuantized(e);
        if (w.size() != expectedElements) {
            throw new IllegalStateException(
                    base
                            + ".weight: expected "
                            + expectedElements
                            + " elements, GGUF has "
                            + w.size());
        }
        return new Clamped(
                w,
                scalar(t, base + ".input_min", -Float.MAX_VALUE),
                scalar(t, base + ".input_max", Float.MAX_VALUE),
                scalar(t, base + ".output_min", -Float.MAX_VALUE),
                scalar(t, base + ".output_max", Float.MAX_VALUE));
    }

    private static float scalar(Map<String, GGMLTensorEntry> t, String name, float dflt) {
        GGMLTensorEntry e = t.get(name);
        return e == null ? dflt : ModelLoader.loadQuantized(e).getFloat(0);
    }

    /**
     * build_mm: clamp input into {@code clampTmp} (when clamped), matmul, clamp output in place.
     * {@code clampTmp} must hold {@code rows * inDim} floats and is per-encode scratch - an
     * instance-level buffer would corrupt concurrent encodes.
     */
    void gemm(
            FloatTensor in,
            int inDim,
            FloatTensor out,
            int outDim,
            int rows,
            FloatTensor clampTmp) {
        FloatTensor src = in;
        if (inMin > -Float.MAX_VALUE || inMax < Float.MAX_VALUE) {
            int need = rows * inDim;
            in.copyTo(0, clampTmp, 0, need);
            clampTmp.clampInPlace(0, need, inMin, inMax);
            src = clampTmp;
        }
        w.gemm(src, inDim, out, outDim, rows, outDim, inDim);
        if (outMin > -Float.MAX_VALUE || outMax < Float.MAX_VALUE) {
            out.clampInPlace(0, rows * outDim, outMin, outMax);
        }
    }
}
