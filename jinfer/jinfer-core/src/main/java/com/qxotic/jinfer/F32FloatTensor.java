// F32FloatTensor split out of Tensors.java so it can be public: the allocatable, writable native
// float tensor used for scratch / KV cache and returned by the model loaders. Consumed by the
// jinfer-gemma4 model port; the quantized subclasses stay package-private in Tensors.java.
package com.qxotic.jinfer;

import com.oracle.svm.shared.AlwaysInline;

import com.qxotic.format.gguf.GGMLType;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;


import java.lang.reflect.Field;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.function.IntConsumer;

public final class F32FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    F32FloatTensor(long numElements, MemorySegment memorySegment) {
        super(numElements, memorySegment);
        this.memorySegment = memorySegment;
    }

    public static F32FloatTensor allocate(int... dims) {
        int numberOfElements = FloatTensor.numberOfElements(dims);
        return new F32FloatTensor(numberOfElements, Arena.ofAuto().allocate((long) numberOfElements * Float.BYTES, 64));
    }

    /** Native copy of a heap float[] (e.g. computed rope frequency tables). */
    public static F32FloatTensor of(float[] values) {
        F32FloatTensor tensor = allocate(values.length);
        MemorySegment.copy(values, 0, tensor.memorySegment, ValueLayout.JAVA_FLOAT_UNALIGNED, 0, values.length);
        return tensor;
    }

    @Override
    public long copyRawTo(long elemOffset, MemorySegment dst, long dstByteOffset, long elemCount) {
        MemorySegment.copy(memorySegment, elemOffset * 4, dst, dstByteOffset, elemCount * 4);
        return elemCount * 4;
    }

    @Override
    public long copyRawFrom(MemorySegment src, long srcByteOffset, long elemOffset, long elemCount) {
        MemorySegment.copy(src, srcByteOffset, memorySegment, elemOffset * 4, elemCount * 4);
        return elemCount * 4;
    }

    @Override
    public float getFloat(long index) {
        // through GLOBAL_SEGMENT (readFloat) so the access inlines on native image (see FloatTensor)
        return readFloat(memorySegment, index * Float.BYTES);
    }

    @Override
    public void setFloat(long index, float value) {
        writeFloat(memorySegment, (long) index * Float.BYTES, value);
    }

    @Override public GGMLType type() { return GGMLType.F32; }

    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromMemorySegment(species, vseg, vbase + (long) index * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (that instanceof F32FloatTensor f32 && USE_VECTOR_API) {
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        }
        if (that instanceof F16FloatTensor) {
            return that.dot(thatOffset, this, thisOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    @Override
    public FloatTensor fillInPlace(long thisOffset, int size, float value) {
        if (USE_VECTOR_API) {
            FloatVector fill = FloatVector.broadcast(F_SPECIES, value);
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                fill.intoMemorySegment(vseg, vbase + (long) (thisOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                setFloat(thisOffset + i, value);
            }
            return this;
        }
        return super.fillInPlace(thisOffset, size, value);
    }

    @Override
    public FloatTensor clampInPlace(long thisOffset, int size, float lo, float hi) {
        if (USE_VECTOR_API) {
            FloatVector vlo = FloatVector.broadcast(F_SPECIES, lo), vhi = FloatVector.broadcast(F_SPECIES, hi);
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                long b = vbase + (long) (thisOffset + i) * Float.BYTES;
                var v = FloatVector.fromMemorySegment(F_SPECIES, vseg, b, ByteOrder.LITTLE_ENDIAN);
                v.max(vlo).min(vhi).intoMemorySegment(vseg, b, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                float v = getFloat(thisOffset + i);
                setFloat(thisOffset + i, v < lo ? lo : v > hi ? hi : v);
            }
            return this;
        }
        return super.clampInPlace(thisOffset, size, lo, hi);
    }

    @Override
    public FloatTensor addInPlace(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (that instanceof F32FloatTensor f32 && USE_VECTOR_API) {
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                var a = FloatVector.fromMemorySegment(F_SPECIES, vseg, vbase + (long) (thisOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                var b = FloatVector.fromMemorySegment(F_SPECIES, f32.vseg, f32.vbase + (long) (thatOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                a.add(b).intoMemorySegment(vseg, vbase + (long) (thisOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                setFloat(thisOffset + i, getFloat(thisOffset + i) + f32.getFloat(thatOffset + i));
            }
            return this;
        }
        return super.addInPlace(thisOffset, that, thatOffset, size);
    }

    @Override
    public FloatTensor saxpyInPlace(long thisOffset, FloatTensor that, long thatOffset, int size, float a) {
        if (that instanceof F16FloatTensor f16 && USE_VECTOR_API) {
            FloatVector va = FloatVector.broadcast(F_SPECIES, a);
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                // f16ToF32Vector inlined by hand for C2 (see F16FloatTensor.vectorDot)
                var bits32 = ShortVector.fromMemorySegment(S_SPECIES_HALF, f16.vseg, f16.vbase + (thatOffset + i) * 2L, ByteOrder.LITTLE_ENDIAN)
                        .castShape(I_SPECIES, 0).reinterpretAsInts();
                var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31);
                FloatVector thatVector = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16)
                        .or(bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13).and(zeroExponentMask))
                        .reinterpretAsFloats();
                FloatVector thisVector = FloatVector.fromMemorySegment(F_SPECIES, vseg, vbase + (long) (thisOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                va.fma(thatVector, thisVector).intoMemorySegment(vseg, vbase + (long) (thisOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                setFloat(thisOffset + i, a * f16.getFloat(thatOffset + i) + getFloat(thisOffset + i));
            }
            return this;
        }
        return super.saxpyInPlace(thisOffset, that, thatOffset, size, a);
    }

    @Override
    public void copyTo(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (that instanceof F32FloatTensor f32) {
            MemorySegment.copy(memorySegment, (long) thisOffset * Float.BYTES,
                    f32.memorySegment, (long) thatOffset * Float.BYTES, (long) size * Float.BYTES);
            return;
        }
        super.copyTo(thisOffset, that, thatOffset, size);
    }

    @Override
    FloatTensor siluMultiplyInPlace(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (that instanceof F32FloatTensor f32 && USE_VECTOR_API) {
            // silu(g)*u, fully vectorized. silu(g)=g*(0.5+0.5*tanh(g/2)) via a Pade(7,7) rational tanh:
            // only mul/add/div (no exp, no integer bit-ops), so it vectorizes on GraalVM/jvmci too (where the
            // lanewise EXP intrinsic is absent and the scalar fallback was ~24% of prefill). ~1e-5 abs error.
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                long thisByte = vbase + (long) (thisOffset + i) * Float.BYTES;
                var g = FloatVector.fromMemorySegment(F_SPECIES, vseg, thisByte, ByteOrder.LITTLE_ENDIAN);
                var u = FloatVector.fromMemorySegment(F_SPECIES, f32.vseg, f32.vbase + (long) (thatOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                // silu(g)*u with siluVec/tanhVec INLINED by hand: the FloatVector temporaries never cross a
                // method boundary, so they scalar-replace into SIMD registers on any JIT — not only those that
                // inline the helpers (a weaker inliner otherwise boxes the escaping return value). Identical
                // math to siluVec(g).mul(u); the helpers stay for GELU / the scalar tail. Keep in sync with tanhVec.
                FloatVector y = g.mul(0.5f).max(-TANH_CUTOFF).min(TANH_CUTOFF);     // tanh input = g/2, clamped
                FloatVector y2 = y.mul(y);
                FloatVector num = FloatVector.broadcast(F_SPECIES, TANH_N0)
                                    .fma(y2, FloatVector.broadcast(F_SPECIES, TANH_N1))
                                    .fma(y2, FloatVector.broadcast(F_SPECIES, TANH_N2)).mul(y2);
                FloatVector den = y2.add(TANH_D0).fma(y2, FloatVector.broadcast(F_SPECIES, TANH_D1));
                FloatVector tanh = num.div(den).fma(y, y);                          // tanh(g/2)
                g.mul(tanh.mul(0.5f).add(0.5f)).mul(u).intoMemorySegment(vseg, thisByte, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                float g = getFloat(thisOffset + i);
                setFloat(thisOffset + i, (float) (g / (1.0 + Math.exp(-g)) * f32.getFloat(thatOffset + i)));
            }
            return this;
        }
        return super.siluMultiplyInPlace(thisOffset, that, thatOffset, size);
    }

    /** Vectorized SiLU g*(0.5+0.5*tanh(g/2)). tanh(y) via njuffa's minimax rational approximation (the
     *  "cutoff" variant): y is clamped to +/-CUTOFF, where tanh has saturated to ~1 (so no output clamp is
     *  needed), then tanh(y) = y + y*num(y^2)/den(y^2). Only mul/add/div/fma -> vectorizes on GraalVM/jvmci
     *  (unlike a lanewise EXP, which Graal does not intrinsify). Source: njuffa, StackOverflow "fast tanhf".
     *  Precision: |error| <= ~1.9e-5 for tanh over all float32; <= 1.1e-4 abs / 4.5e-3 rel for this SiLU over
     *  g in [-40,40] (worst near g~11.5; near-exact for |g|<2). Well under Q8_0's ~3.9e-3 quantization noise. */
    @AlwaysInline("hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    static FloatVector siluVec(FloatVector g) {
        FloatVector tanh = tanhVec(g.mul(0.5f));                     // tanh(g/2)
        return g.mul(tanh.mul(0.5f).add(0.5f));                      // g * sigmoid(g)
    }

    // njuffa minimax-rational tanh coefficients (the "cutoff" variant). One source of truth, shared by
    // tanhVec and the manually-inlined SiLU loop in siluMultiplyInPlace below — keep them in sync.
    static final float TANH_CUTOFF = 5.76110792f;                   // clamp |x| here (tanh ~ ±1 beyond)
    static final float TANH_N0 = -1.60153955e-4f, TANH_N1 = -9.34448242e-1f, TANH_N2 = -2.19176636e+1f;
    static final float TANH_D0 = 29.0915985f,     TANH_D1 = 65.7667847f;

    /** Vectorized tanh(x) via njuffa's minimax rational (the "cutoff" variant): x clamped to +/-CUTOFF
     *  (tanh saturated to ~1 there, so no output clamp), tanh = x + x*num(x^2)/den(x^2). Only mul/add/div/fma,
     *  so it runs fast on GraalVM/jvmci (which does NOT intrinsify lanewise TANH/EXP). Source: njuffa,
     *  StackOverflow "fast tanhf". |error| <= ~1.9e-5 over all float32. Shared by SiLU and Gemma's GELU. */
    @AlwaysInline("hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    static FloatVector tanhVec(FloatVector x) {
        FloatVector y  = x.max(-TANH_CUTOFF).min(TANH_CUTOFF);
        FloatVector y2 = y.mul(y);
        FloatVector num = FloatVector.broadcast(F_SPECIES, TANH_N0)
                            .fma(y2, FloatVector.broadcast(F_SPECIES, TANH_N1))
                            .fma(y2, FloatVector.broadcast(F_SPECIES, TANH_N2)).mul(y2);
        FloatVector den = y2.add(TANH_D0).fma(y2, FloatVector.broadcast(F_SPECIES, TANH_D1));
        return num.div(den).fma(y, y);                              // y + y*num/den
    }

    /** Scalar twin of {@link #tanhVec} — same clamp, constants and fma ops, so a vectorized loop's
     *  scalar remainder applies the identical approximation to its tail lanes (one monotonic function
     *  across the whole span) instead of diverging to {@code Math.tanh}. */
    static float tanhApprox(float x) {
        float y  = Math.max(-TANH_CUTOFF, Math.min(TANH_CUTOFF, x));
        float y2 = y * y;
        float num = Math.fma(Math.fma(TANH_N0, y2, TANH_N1), y2, TANH_N2) * y2;
        float den = Math.fma(y2 + TANH_D0, y2, TANH_D1);
        return Math.fma(num / den, y, y);                          // y + y*num/den
    }

    @Override
    FloatTensor reluSqrInPlace(long thisOffset, int size) {
        if (USE_VECTOR_API) {
            // x = max(0,x)^2, fully vectorized (max + mul only, no scalar setFloat). Nemotron FFN/expert act.
            int upperBound = F_SPECIES.loopBound(size);
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                long byteOff = vbase + (long) (thisOffset + i) * Float.BYTES;
                var r = FloatVector.fromMemorySegment(F_SPECIES, vseg, byteOff, ByteOrder.LITTLE_ENDIAN).max(0f);
                r.mul(r).intoMemorySegment(vseg, byteOff, ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                float r = getFloat(thisOffset + i); r = r > 0f ? r : 0f;
                setFloat(thisOffset + i, r * r);
            }
            return this;
        }
        return super.reluSqrInPlace(thisOffset, size);
    }

    private static float vectorDot(F32FloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            var a = FloatVector.fromMemorySegment(F_SPECIES, thiz.vseg, thiz.vbase + (long) (thisOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            var b = FloatVector.fromMemorySegment(F_SPECIES, that.vseg, that.vbase + (long) (thatOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            val = a.fma(b, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        for (int i = upperBound; i < size; i++) {
            result += thiz.getFloat(thisOffset + i) * that.getFloat(thatOffset + i);
        }
        return result;
    }
}
