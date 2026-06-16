// Shared flash-attention primitives: online-softmax accumulate/normalize over a head vector and
// the per-thread tile scratch. Architecture-agnostic (operate on FloatTensors), used by the
// batched attention of any Model. Vector-API fast paths for an F32 output with an F32 or F16
// value source; scalar fallback otherwise.
package com.llama4j;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import java.nio.ByteOrder;

final class FlashAttention {

    /** Q/K tile sizes for block-tiled prefill (cache-friendly inner loops). */
    static final int Br = 64;
    static final int Bc = 64;

    /** Query-row tile width for the register-tiled QK^T / PV kernels: each key/value vector is
     *  loaded (and F16-decoded) once and reused across QT consecutive query rows. 8 keeps the
     *  accumulators + operand within the 16 zmm registers while doubling the key/value reuse vs 4. */
    static final int QT = 8;

    /** Per-thread scratch: the Br×Bc score tile, per-row running max/sum, and per-block K/V offsets. */
    static final class Buffers {
        final float[] s = new float[Br * Bc];
        final float[] m = new float[Br];
        final double[] l = new double[Br];
        final int[] kvOff = new int[Bc];
    }

    private static final ThreadLocal<Buffers> BUFFERS = ThreadLocal.withInitial(Buffers::new);

    static Buffers buffers() {
        return BUFFERS.get();
    }

    /** out[outOffset, +headSize] *= scale (rescale the running output on a new row max). */
    static void normalize(FloatTensor out, int outOffset, int headSize, float scale) {
        if (out instanceof F32FloatTensor outF32 && FloatTensor.USE_VECTOR_API) {
            FloatVector scaleVector = FloatVector.broadcast(FloatTensor.F_SPECIES, scale);
            int upperBound = FloatTensor.F_SPECIES.loopBound(headSize);
            for (int i = 0; i < upperBound; i += FloatTensor.F_SPECIES.length()) {
                long byteOffset = (long) (outOffset + i) * Float.BYTES;
                FloatVector.fromMemorySegment(FloatTensor.F_SPECIES, outF32.vseg, outF32.vbase + byteOffset, ByteOrder.LITTLE_ENDIAN)
                        .mul(scaleVector).intoMemorySegment(outF32.vseg, outF32.vbase + byteOffset, ByteOrder.LITTLE_ENDIAN);
            }
            for (int i = upperBound; i < headSize; i++) {
                outF32.setFloat(outOffset + i, outF32.getFloat(outOffset + i) * scale);
            }
            return;
        }
        out.mapInPlace(outOffset, headSize, v -> v * scale);
    }

    /** out[outOffset, +headSize] += scale * value[valueOffset, +headSize]. */
    static void accumulate(FloatTensor out, int outOffset, FloatTensor value, int valueOffset, int headSize, float scale) {
        if (out instanceof F32FloatTensor outF32 && FloatTensor.USE_VECTOR_API
                && (value instanceof F32FloatTensor || value instanceof F16FloatTensor)) {
            FloatVector scaleVector = FloatVector.broadcast(FloatTensor.F_SPECIES, scale);
            int upperBound = FloatTensor.F_SPECIES.loopBound(headSize);
            if (value instanceof F32FloatTensor valueF32) {
                for (int d = 0; d < upperBound; d += FloatTensor.F_SPECIES.length()) {
                    long byteOffset = (long) (outOffset + d) * Float.BYTES;
                    FloatVector acc = FloatVector.fromMemorySegment(FloatTensor.F_SPECIES, outF32.vseg, outF32.vbase + byteOffset, ByteOrder.LITTLE_ENDIAN);
                    FloatVector v = FloatVector.fromMemorySegment(FloatTensor.F_SPECIES, valueF32.vseg, valueF32.vbase + (long) (valueOffset + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                    v.fma(scaleVector, acc).intoMemorySegment(outF32.vseg, outF32.vbase + byteOffset, ByteOrder.LITTLE_ENDIAN);
                }
            } else {
                F16FloatTensor f16Value = (F16FloatTensor) value;
                for (int d = 0; d < upperBound; d += FloatTensor.F_SPECIES.length()) {
                    long byteOffset = (long) (outOffset + d) * Float.BYTES;
                    FloatVector acc = FloatVector.fromMemorySegment(FloatTensor.F_SPECIES, outF32.vseg, outF32.vbase + byteOffset, ByteOrder.LITTLE_ENDIAN);
                    var bits32 = ShortVector.fromMemorySegment(FloatTensor.S_SPECIES_HALF, f16Value.vseg, f16Value.vbase + (long) (valueOffset + d) * Float16.BYTES, ByteOrder.LITTLE_ENDIAN)
                            .castShape(FloatTensor.I_SPECIES, 0).reinterpretAsInts();
                    var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31);
                    FloatVector v = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16)
                            .or(bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13).and(zeroExponentMask))
                            .reinterpretAsFloats();
                    v.fma(scaleVector, acc).intoMemorySegment(outF32.vseg, outF32.vbase + byteOffset, ByteOrder.LITTLE_ENDIAN);
                }
            }
            for (int d = upperBound; d < headSize; d++) {
                outF32.setFloat(outOffset + d, outF32.getFloat(outOffset + d) + value.getFloat(valueOffset + d) * scale);
            }
            return;
        }
        for (int d = 0; d < headSize; d++) {
            out.setFloat(outOffset + d, out.getFloat(outOffset + d) + value.getFloat(valueOffset + d) * scale);
        }
    }

    private static FloatVector loadF32(F32FloatTensor t, int off) {
        return FloatVector.fromMemorySegment(FloatTensor.F_SPECIES, t.vseg, t.vbase + (long) off * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
    }

    /** Decode F_SPECIES.length() consecutive F16 values to an F32 vector (IEEE half -> single). */
    private static FloatVector loadF16(F16FloatTensor t, int off) {
        return F16FloatTensor.f16ToF32Vector(t.vseg, t.vbase + (long) off * Float16.BYTES);
    }

    /**
     * Register-tiled QK^T for QT consecutive query rows against a run of {@code nKeys} keys from a
     * single source (F32 batch or F16 cache). Each key vector is loaded once and reused across the QT
     * query rows (contraction over {@code headSize}); each score is multiplied by {@code scale} (1.0
     * when the scale is folded into the query norm). Writes {@code S[(t)*BcRows + runStart+k]} for
     * query row {@code t} in [0,QT) and key {@code k} in [0,nKeys). Key offsets come from {@code kvOff}.
     */
    static void qkTile(F32FloatTensor q, int qBase, int qStride, FloatTensor key,
                       int[] kvOff, int runStart, int nKeys, int headSize, float scale, float[] S, int sRow0, int BcRows) {
        var sp = FloatTensor.F_SPECIES;
        int len = sp.length();
        int bound = sp.loopBound(headSize);
        int qb0 = qBase, qb1 = qBase + qStride, qb2 = qBase + 2 * qStride, qb3 = qBase + 3 * qStride;
        int qb4 = qBase + 4 * qStride, qb5 = qBase + 5 * qStride, qb6 = qBase + 6 * qStride, qb7 = qBase + 7 * qStride;
        boolean f16 = key instanceof F16FloatTensor;
        F32FloatTensor kf32 = f16 ? null : (F32FloatTensor) key;
        F16FloatTensor kf16 = f16 ? (F16FloatTensor) key : null;
        for (int k = 0; k < nKeys; k++) {
            int ko = kvOff[runStart + k];
            FloatVector a0 = FloatVector.zero(sp), a1 = FloatVector.zero(sp), a2 = FloatVector.zero(sp), a3 = FloatVector.zero(sp);
            FloatVector a4 = FloatVector.zero(sp), a5 = FloatVector.zero(sp), a6 = FloatVector.zero(sp), a7 = FloatVector.zero(sp);
            if (f16) {
                for (int d = 0; d < bound; d += len) {
                    FloatVector kv = loadF16(kf16, ko + d);
                    a0 = loadF32(q, qb0 + d).fma(kv, a0);
                    a1 = loadF32(q, qb1 + d).fma(kv, a1);
                    a2 = loadF32(q, qb2 + d).fma(kv, a2);
                    a3 = loadF32(q, qb3 + d).fma(kv, a3);
                    a4 = loadF32(q, qb4 + d).fma(kv, a4);
                    a5 = loadF32(q, qb5 + d).fma(kv, a5);
                    a6 = loadF32(q, qb6 + d).fma(kv, a6);
                    a7 = loadF32(q, qb7 + d).fma(kv, a7);
                }
            } else {
                for (int d = 0; d < bound; d += len) {
                    FloatVector kv = loadF32(kf32, ko + d);
                    a0 = loadF32(q, qb0 + d).fma(kv, a0);
                    a1 = loadF32(q, qb1 + d).fma(kv, a1);
                    a2 = loadF32(q, qb2 + d).fma(kv, a2);
                    a3 = loadF32(q, qb3 + d).fma(kv, a3);
                    a4 = loadF32(q, qb4 + d).fma(kv, a4);
                    a5 = loadF32(q, qb5 + d).fma(kv, a5);
                    a6 = loadF32(q, qb6 + d).fma(kv, a6);
                    a7 = loadF32(q, qb7 + d).fma(kv, a7);
                }
            }
            float s0 = a0.reduceLanes(VectorOperators.ADD), s1 = a1.reduceLanes(VectorOperators.ADD);
            float s2 = a2.reduceLanes(VectorOperators.ADD), s3 = a3.reduceLanes(VectorOperators.ADD);
            float s4 = a4.reduceLanes(VectorOperators.ADD), s5 = a5.reduceLanes(VectorOperators.ADD);
            float s6 = a6.reduceLanes(VectorOperators.ADD), s7 = a7.reduceLanes(VectorOperators.ADD);
            for (int d = bound; d < headSize; d++) {
                float kvf = key.getFloat(ko + d);
                s0 += q.getFloat(qb0 + d) * kvf;
                s1 += q.getFloat(qb1 + d) * kvf;
                s2 += q.getFloat(qb2 + d) * kvf;
                s3 += q.getFloat(qb3 + d) * kvf;
                s4 += q.getFloat(qb4 + d) * kvf;
                s5 += q.getFloat(qb5 + d) * kvf;
                s6 += q.getFloat(qb6 + d) * kvf;
                s7 += q.getFloat(qb7 + d) * kvf;
            }
            int col = runStart + k;
            S[sRow0 + col] = s0 * scale;
            S[sRow0 + BcRows + col] = s1 * scale;
            S[sRow0 + 2 * BcRows + col] = s2 * scale;
            S[sRow0 + 3 * BcRows + col] = s3 * scale;
            S[sRow0 + 4 * BcRows + col] = s4 * scale;
            S[sRow0 + 5 * BcRows + col] = s5 * scale;
            S[sRow0 + 6 * BcRows + col] = s6 * scale;
            S[sRow0 + 7 * BcRows + col] = s7 * scale;
        }
    }

    /**
     * Register-tiled PV for QT consecutive query rows over a run of {@code nKeys} values from a single
     * source (F32 batch or F16 cache). Each value vector is loaded once per chunk and reused across the
     * QT rows; each row's output is read+written once per chunk. Adds {@code sum_k P[t][k]*V_k} into
     * {@code out} for query row {@code t}; probabilities come from {@code P[(t)*BcRows + runStart+k]}.
     */
    static void pvTile(F32FloatTensor out, int oBase, int oStride, FloatTensor value,
                       int[] kvOff, int runStart, int nKeys, int headSize, float[] P, int pRow0, int BcRows) {
        var sp = FloatTensor.F_SPECIES;
        int len = sp.length();
        int bound = sp.loopBound(headSize);
        int ob0 = oBase, ob1 = oBase + oStride, ob2 = oBase + 2 * oStride, ob3 = oBase + 3 * oStride;
        int ob4 = oBase + 4 * oStride, ob5 = oBase + 5 * oStride, ob6 = oBase + 6 * oStride, ob7 = oBase + 7 * oStride;
        int p0 = pRow0, p1 = pRow0 + BcRows, p2 = pRow0 + 2 * BcRows, p3 = pRow0 + 3 * BcRows;
        int p4 = pRow0 + 4 * BcRows, p5 = pRow0 + 5 * BcRows, p6 = pRow0 + 6 * BcRows, p7 = pRow0 + 7 * BcRows;
        boolean f16 = value instanceof F16FloatTensor;
        F32FloatTensor vf32 = f16 ? null : (F32FloatTensor) value;
        F16FloatTensor vf16 = f16 ? (F16FloatTensor) value : null;
        for (int d = 0; d < bound; d += len) {
            FloatVector o0 = loadF32(out, ob0 + d), o1 = loadF32(out, ob1 + d), o2 = loadF32(out, ob2 + d), o3 = loadF32(out, ob3 + d);
            FloatVector o4 = loadF32(out, ob4 + d), o5 = loadF32(out, ob5 + d), o6 = loadF32(out, ob6 + d), o7 = loadF32(out, ob7 + d);
            for (int k = 0; k < nKeys; k++) {
                int col = runStart + k;
                FloatVector v = f16 ? loadF16(vf16, kvOff[col] + d) : loadF32(vf32, kvOff[col] + d);
                o0 = v.fma(FloatVector.broadcast(sp, P[p0 + col]), o0);
                o1 = v.fma(FloatVector.broadcast(sp, P[p1 + col]), o1);
                o2 = v.fma(FloatVector.broadcast(sp, P[p2 + col]), o2);
                o3 = v.fma(FloatVector.broadcast(sp, P[p3 + col]), o3);
                o4 = v.fma(FloatVector.broadcast(sp, P[p4 + col]), o4);
                o5 = v.fma(FloatVector.broadcast(sp, P[p5 + col]), o5);
                o6 = v.fma(FloatVector.broadcast(sp, P[p6 + col]), o6);
                o7 = v.fma(FloatVector.broadcast(sp, P[p7 + col]), o7);
            }
            o0.intoMemorySegment(out.vseg, out.vbase + (long) (ob0 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            o1.intoMemorySegment(out.vseg, out.vbase + (long) (ob1 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            o2.intoMemorySegment(out.vseg, out.vbase + (long) (ob2 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            o3.intoMemorySegment(out.vseg, out.vbase + (long) (ob3 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            o4.intoMemorySegment(out.vseg, out.vbase + (long) (ob4 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            o5.intoMemorySegment(out.vseg, out.vbase + (long) (ob5 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            o6.intoMemorySegment(out.vseg, out.vbase + (long) (ob6 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            o7.intoMemorySegment(out.vseg, out.vbase + (long) (ob7 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
        }
        for (int d = bound; d < headSize; d++) {
            float r0 = out.getFloat(ob0 + d), r1 = out.getFloat(ob1 + d), r2 = out.getFloat(ob2 + d), r3 = out.getFloat(ob3 + d);
            float r4 = out.getFloat(ob4 + d), r5 = out.getFloat(ob5 + d), r6 = out.getFloat(ob6 + d), r7 = out.getFloat(ob7 + d);
            for (int k = 0; k < nKeys; k++) {
                int col = runStart + k;
                float vf = value.getFloat(kvOff[col] + d);
                r0 += P[p0 + col] * vf;
                r1 += P[p1 + col] * vf;
                r2 += P[p2 + col] * vf;
                r3 += P[p3 + col] * vf;
                r4 += P[p4 + col] * vf;
                r5 += P[p5 + col] * vf;
                r6 += P[p6 + col] * vf;
                r7 += P[p7 + col] * vf;
            }
            out.setFloat(ob0 + d, r0);
            out.setFloat(ob1 + d, r1);
            out.setFloat(ob2 + d, r2);
            out.setFloat(ob3 + d, r3);
            out.setFloat(ob4 + d, r4);
            out.setFloat(ob5 + d, r5);
            out.setFloat(ob6 + d, r6);
            out.setFloat(ob7 + d, r7);
        }
    }

    private FlashAttention() {
    }
}
