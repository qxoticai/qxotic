package com.qxotic.jinfer.x.kernels;

import static com.qxotic.jinfer.x.Segments.F_SPECIES;
import static com.qxotic.jinfer.x.Segments.USE_VECTOR_API;
import static com.qxotic.jinfer.x.Segments.readFloat;
import static com.qxotic.jinfer.x.Segments.writeFloat;

import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Scalar activation functions and fused whole-span operations over dense FP32 views. Scalar tails
 * use the same {@code tanhApprox} twin as vector lanes, so a span applies one monotonic function
 * across the body/tail boundary.
 */
public final class Activations {
    private Activations() {}

    /** Logistic sigmoid {@code 1/(1+e^-x)}. */
    public static float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    /** SiLU / swish {@code x*sigmoid(x)}. */
    public static float silu(float x) {
        return x * sigmoid(x);
    }

    /** Numerically-stable softplus {@code log(1+e^x)}. */
    public static float softplus(float x) {
        if (x > 20f) return x;
        if (x < -20f) return (float) Math.exp(x);
        return (float) Math.log1p(Math.exp(x));
    }

    /** In-place ReLU-squared over {@code n} elements: {@code max(0,x)^2} (Nemotron-H FFN). */
    public static void reluSqr(MemoryView<MemorySegment> t, int off, int n) {
        Ops.reluSqrInPlace(t, off, n);
    }

    private static final float GELU_C = (float) Math.sqrt(2.0 / Math.PI);

    /** tanh-approximation GELU (exact {@code Math.tanh}) — the scalar-fallback oracle. */
    public static float gelu(float x) {
        float inner = GELU_C * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + (float) Math.tanh(inner));
    }

    /**
     * Scalar twin of {@link #geluMultiply}'s vector body — same op order and {@link Ops#tanhApprox}
     * as the lanes, so the vector loop's scalar tail applies the identical approximation.
     */
    private static float geluApprox(float x) {
        float inner = (x * x * x * 0.044715f + x) * GELU_C;
        return x * 0.5f * (1.0f + Ops.tanhApprox(inner));
    }

    /** In-place tanh-approximation GELU over {@code n} FP32 elements. */
    public static void geluInPlace(MemoryView<MemorySegment> value, int offset, int n) {
        Raw raw = Raw.f32(value, "value");
        if (USE_VECTOR_API) {
            VectorSpecies<Float> sp = F_SPECIES;
            int bound = sp.loopBound(n);
            for (int i = 0; i < bound; i += sp.length()) {
                long base = raw.vbase() + (long) (offset + i) * Float.BYTES;
                FloatVector x =
                        FloatVector.fromMemorySegment(
                                sp, raw.vseg(), base, ByteOrder.LITTLE_ENDIAN);
                FloatVector inner = x.mul(x).mul(x).mul(0.044715f).add(x).mul(GELU_C);
                x.mul(0.5f)
                        .mul(Ops.tanhVec(inner).add(1.0f))
                        .intoMemorySegment(raw.vseg(), base, ByteOrder.LITTLE_ENDIAN);
            }
            for (int i = bound; i < n; i++) {
                long base = raw.vbase() + (long) (offset + i) * Float.BYTES;
                writeFloat(raw.vseg(), base, geluApprox(readFloat(raw.vseg(), base)));
            }
            return;
        }
        for (int i = 0; i < n; i++) {
            long base = raw.vbase() + (long) (offset + i) * Float.BYTES;
            writeFloat(raw.vseg(), base, gelu(readFloat(raw.vseg(), base)));
        }
    }

    /**
     * Fused {@code gate[i] = gelu(gate[i]) * up[i]} over {@code n} elements (minimax-rational
     * {@code tanhVec}); mutates {@code gate}. Callers parallelize across rows.
     */
    public static void geluMultiply(
            MemoryView<MemorySegment> gate,
            int gateOff,
            MemoryView<MemorySegment> up,
            int upOff,
            int n) {
        Raw g = Raw.f32(gate, "gate");
        Raw u = Raw.f32(up, "up");
        if (USE_VECTOR_API) {
            VectorSpecies<Float> sp = F_SPECIES;
            int bound = sp.loopBound(n);
            for (int i = 0; i < bound; i += sp.length()) {
                long gb = g.vbase() + (long) (gateOff + i) * Float.BYTES;
                long ub = u.vbase() + (long) (upOff + i) * Float.BYTES;
                FloatVector x =
                        FloatVector.fromMemorySegment(sp, g.vseg(), gb, ByteOrder.LITTLE_ENDIAN);
                FloatVector uv =
                        FloatVector.fromMemorySegment(sp, u.vseg(), ub, ByteOrder.LITTLE_ENDIAN);
                FloatVector inner = x.mul(x).mul(x).mul(0.044715f).add(x).mul(GELU_C);
                FloatVector t = Ops.tanhVec(inner);
                x.mul(0.5f)
                        .mul(t.add(1.0f))
                        .mul(uv)
                        .intoMemorySegment(g.vseg(), gb, ByteOrder.LITTLE_ENDIAN);
            }
            for (int i = bound; i < n; i++) {
                float gv = readFloat(g.vseg(), g.vbase() + (long) (gateOff + i) * Float.BYTES);
                float uv = readFloat(u.vseg(), u.vbase() + (long) (upOff + i) * Float.BYTES);
                writeFloat(
                        g.vseg(),
                        g.vbase() + (long) (gateOff + i) * Float.BYTES,
                        geluApprox(gv) * uv);
            }
            return;
        }
        for (int i = 0; i < n; i++) {
            float gv = readFloat(g.vseg(), g.vbase() + (long) (gateOff + i) * Float.BYTES);
            float uv = readFloat(u.vseg(), u.vbase() + (long) (upOff + i) * Float.BYTES);
            writeFloat(g.vseg(), g.vbase() + (long) (gateOff + i) * Float.BYTES, gelu(gv) * uv);
        }
    }

    /**
     * Exact quick-GELU gate {@code gate = gate / (1 + exp(-1.702 * gate)) * up}; mutates {@code
     * gate}. This intentionally stays scalar because the vision projector requires {@link Math#exp}
     * rather than the tanh approximation used by other fused gates.
     */
    public static void quickGeluMultiply(
            MemoryView<MemorySegment> gate,
            long gateOff,
            MemoryView<MemorySegment> up,
            long upOff,
            int n) {
        Raw g = Raw.f32(gate, "gate");
        Raw u = Raw.f32(up, "up");
        for (int i = 0; i < n; i++) {
            long gb = g.vbase() + (gateOff + i) * Float.BYTES;
            float gv = readFloat(g.vseg(), gb);
            float activated = gv / (1f + (float) Math.exp(-1.702f * gv));
            writeFloat(
                    g.vseg(),
                    gb,
                    activated * readFloat(u.vseg(), u.vbase() + (upOff + i) * Float.BYTES));
        }
    }

    /** {@code out[i] = packed[i] * sigmoid(packed[n + i])} over two contiguous halves. */
    public static void glu(
            MemoryView<MemorySegment> out,
            long outOff,
            MemoryView<MemorySegment> packed,
            long packedOff,
            int n) {
        Raw o = Raw.f32(out, "out");
        Raw x = Raw.f32(packed, "packed");
        for (int i = 0; i < n; i++) {
            float first = readFloat(x.vseg(), x.vbase() + (packedOff + i) * Float.BYTES);
            float second = readFloat(x.vseg(), x.vbase() + (packedOff + n + i) * Float.BYTES);
            writeFloat(
                    o.vseg(),
                    o.vbase() + (outOff + i) * Float.BYTES,
                    first * FastMath.sigmoid(second));
        }
    }

    /**
     * Exact gpt-oss clamped-SwiGLU scalar: {@code quickgelu(min(gate,7)) * (clamp(up,±7) + 1)},
     * where {@code quickgelu(x) = x*sigmoid(1.702x)}. The full-scalar oracle for {@link
     * #clampedSwigluMultiply}.
     */
    public static float clampedSwiglu(float gate, float up) {
        float x = Math.min(gate, 7.0f);
        float y = Math.clamp(up, -7.0f, 7.0f);
        return (float) (x / (1.0 + Math.exp(1.702f * -x)) * (y + 1.0));
    }

    /** Scalar twin of {@link #clampedSwigluMultiply}'s vector body (tanhApprox sigmoid). */
    private static float clampedSwigluApprox(float gate, float up) {
        float x = Math.min(gate, 7.0f);
        float y = Math.clamp(up, -7.0f, 7.0f);
        return x * 0.5f * (1.0f + Ops.tanhApprox(0.851f * x)) * (y + 1.0f);
    }

    /**
     * Fused gpt-oss clamped-SwiGLU {@code gate[i] = clampedSwiglu(gate[i], up[i])} over {@code n}
     * elements, in place on {@code gate} (sigmoid via {@code sigmoid(1.702x) = 0.5(1 +
     * tanh(0.851x))}). Callers parallelize across rows.
     */
    public static void clampedSwigluMultiply(
            MemoryView<MemorySegment> gate,
            int gateOff,
            MemoryView<MemorySegment> up,
            int upOff,
            int n) {
        Raw g = Raw.f32(gate, "gate");
        Raw u = Raw.f32(up, "up");
        if (USE_VECTOR_API) {
            VectorSpecies<Float> sp = F_SPECIES;
            int bound = sp.loopBound(n);
            for (int i = 0; i < bound; i += sp.length()) {
                long gb = g.vbase() + (long) (gateOff + i) * Float.BYTES;
                long ub = u.vbase() + (long) (upOff + i) * Float.BYTES;
                FloatVector x =
                        FloatVector.fromMemorySegment(sp, g.vseg(), gb, ByteOrder.LITTLE_ENDIAN)
                                .min(7.0f);
                FloatVector y =
                        FloatVector.fromMemorySegment(sp, u.vseg(), ub, ByteOrder.LITTLE_ENDIAN)
                                .max(-7.0f)
                                .min(7.0f);
                FloatVector t = Ops.tanhVec(x.mul(0.851f)); // sigmoid(1.702x)
                x.mul(0.5f)
                        .mul(t.add(1.0f))
                        .mul(y.add(1.0f))
                        .intoMemorySegment(g.vseg(), gb, ByteOrder.LITTLE_ENDIAN);
            }
            for (int i = bound; i < n; i++) {
                float gv = readFloat(g.vseg(), g.vbase() + (long) (gateOff + i) * Float.BYTES);
                float uv = readFloat(u.vseg(), u.vbase() + (long) (upOff + i) * Float.BYTES);
                writeFloat(
                        g.vseg(),
                        g.vbase() + (long) (gateOff + i) * Float.BYTES,
                        clampedSwigluApprox(gv, uv));
            }
            return;
        }
        for (int i = 0; i < n; i++) {
            float gv = readFloat(g.vseg(), g.vbase() + (long) (gateOff + i) * Float.BYTES);
            float uv = readFloat(u.vseg(), u.vbase() + (long) (upOff + i) * Float.BYTES);
            writeFloat(
                    g.vseg(),
                    g.vbase() + (long) (gateOff + i) * Float.BYTES,
                    clampedSwiglu(gv, uv));
        }
    }

    /** Fused {@code gate[i] = silu(gate[i]) * up[i]} over {@code n} elements (SwiGLU). */
    public static void siluMultiply(
            MemoryView<MemorySegment> gate,
            int gateOff,
            MemoryView<MemorySegment> up,
            int upOff,
            int n) {
        Ops.siluMultiplyInPlace(gate, gateOff, up, upOff, n);
    }

    /** The WaveNet gate {@code tanh(filter) * sigmoid(gate)} — the scalar-fallback oracle. */
    public static float tanhSigmoid(float filter, float gate) {
        return (float) Math.tanh(filter) * sigmoid(gate);
    }

    /** Scalar twin of {@link #tanhSigmoidGate}'s vector body (tanhApprox for both halves). */
    private static float tanhSigmoidApprox(float filter, float gate) {
        return Ops.tanhApprox(filter) * 0.5f * (1.0f + Ops.tanhApprox(0.5f * gate));
    }

    /**
     * The WaveNet gate over a span: {@code out[i] = tanh(filter[i]) * sigmoid(gate[i])}. One
     * approximation covers both halves ({@code sigmoid(x) = 0.5(1 + tanh(x/2))}), so neither needs
     * a lanewise EXP; ~1.9e-5 is far under Q8_0's ~3.9e-3 quantization noise. {@code filter} and
     * {@code gate} are usually two halves of one view at two offsets. Callers parallelize across
     * rows.
     */
    public static void tanhSigmoidGate(
            MemoryView<MemorySegment> out,
            long outOff,
            MemoryView<MemorySegment> filter,
            long filterOff,
            MemoryView<MemorySegment> gate,
            long gateOff,
            int n) {
        Raw o = Raw.f32(out, "out");
        Raw f = Raw.f32(filter, "filter");
        Raw g = Raw.f32(gate, "gate");
        if (USE_VECTOR_API) {
            VectorSpecies<Float> sp = F_SPECIES;
            int bound = sp.loopBound(n);
            for (int i = 0; i < bound; i += sp.length()) {
                FloatVector x =
                        FloatVector.fromMemorySegment(
                                sp,
                                f.vseg(),
                                f.vbase() + (filterOff + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                FloatVector y =
                        FloatVector.fromMemorySegment(
                                sp,
                                g.vseg(),
                                g.vbase() + (gateOff + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                Ops.tanhVec(x)
                        .mul(Ops.tanhVec(y.mul(0.5f)).add(1.0f).mul(0.5f))
                        .intoMemorySegment(
                                o.vseg(),
                                o.vbase() + (outOff + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
            }
            for (int i = bound; i < n; i++) {
                float fv = readFloat(f.vseg(), f.vbase() + (filterOff + i) * Float.BYTES);
                float gv = readFloat(g.vseg(), g.vbase() + (gateOff + i) * Float.BYTES);
                writeFloat(
                        o.vseg(),
                        o.vbase() + (outOff + i) * Float.BYTES,
                        tanhSigmoidApprox(fv, gv));
            }
            return;
        }
        for (int i = 0; i < n; i++) {
            float fv = readFloat(f.vseg(), f.vbase() + (filterOff + i) * Float.BYTES);
            float gv = readFloat(g.vseg(), g.vbase() + (gateOff + i) * Float.BYTES);
            writeFloat(o.vseg(), o.vbase() + (outOff + i) * Float.BYTES, tanhSigmoid(fv, gv));
        }
    }

    /** In-place logit soft-cap {@code x = cap * tanh(x / cap)} (no-op when {@code cap <= 0}). */
    public static void softcap(MemoryView<MemorySegment> t, int off, int n, float cap) {
        if (cap <= 0f) return;
        Raw r = Raw.f32(t, "t");
        if (USE_VECTOR_API) {
            VectorSpecies<Float> sp = F_SPECIES;
            int bound = sp.loopBound(n);
            float inv = 1.0f / cap;
            for (int i = 0; i < bound; i += sp.length()) {
                long b = r.vbase() + (long) (off + i) * Float.BYTES;
                FloatVector x =
                        FloatVector.fromMemorySegment(sp, r.vseg(), b, ByteOrder.LITTLE_ENDIAN);
                Ops.tanhVec(x.mul(inv))
                        .mul(cap)
                        .intoMemorySegment(r.vseg(), b, ByteOrder.LITTLE_ENDIAN);
            }
            // tail uses tanhApprox (not Math.tanh) so every lane goes through one monotonic
            // function — soft-cap can't reorder logits across the body/tail boundary.
            for (int i = bound; i < n; i++) {
                long b = r.vbase() + (long) (off + i) * Float.BYTES;
                writeFloat(r.vseg(), b, cap * Ops.tanhApprox(readFloat(r.vseg(), b) * inv));
            }
            return;
        }
        for (int i = 0; i < n; i++) {
            long b = r.vbase() + (long) (off + i) * Float.BYTES;
            writeFloat(r.vseg(), b, cap * (float) Math.tanh(readFloat(r.vseg(), b) / cap));
        }
    }
}
