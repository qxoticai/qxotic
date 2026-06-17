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

    /**
     * Block-tiled causal flash attention over a contiguous full-context F32 KV cache (scale =
     * 1/sqrt(headSize)). Q/output are packed at stride {@code queryDim}, the cache at stride
     * {@code kvDim}; GQA via {@code kvMul}. Online softmax, register-tiled QK/PV, parallel over
     * (head, query block). Writes the attention output into {@code out}. Shared by the plain-causal
     * models (Llama3/Nemotron/Qwen3.5); GptOss keeps its own variant for attention sinks + SWA.
     */
    static void causalPrefill(F32FloatTensor q, F32FloatTensor out, FloatTensor cK, FloatTensor cV,
                              int nHeads, int startPos, int seqLen, int headSize, int kvDim, int queryDim, int kvMul) {
        causalPrefill(q, out, cK, cV, nHeads, startPos, seqLen, headSize, kvDim, queryDim, kvMul,
                1.0f / (float) Math.sqrt(headSize));
    }

    /** As above, with an explicit QK score {@code scale} (Granite uses a custom attention scale
     *  rather than 1/sqrt(headSize)). */
    static void causalPrefill(F32FloatTensor q, F32FloatTensor out, FloatTensor cK, FloatTensor cV,
                              int nHeads, int startPos, int seqLen, int headSize, int kvDim, int queryDim, int kvMul,
                              float scale) {
        boolean vec = FloatTensor.USE_VECTOR_API;   // qkTile/pvTile handle both F32 and F16 caches
        int nQBlocks = (seqLen + Br - 1) / Br;

        Parallel.parallelFor(0, nHeads * nQBlocks, idx -> {
            int h = idx / nQBlocks;
            int qStart = (idx % nQBlocks) * Br;
            Buffers buf = buffers();
            float[] S = buf.s;
            float[] M = buf.m;
            double[] L = buf.l;
            int[] kvOff = buf.kvOff;
            int hHead = h * headSize;
            int kvHeadOffset = (h / kvMul) * headSize;

            int qEnd = Math.min(seqLen, qStart + Br);
            int BrRows = qEnd - qStart;
            for (int i = 0; i < BrRows; i++) {
                M[i] = Float.NEGATIVE_INFINITY;
                L[i] = 0.0;
                out.fillInPlace((qStart + i) * queryDim + hHead, headSize, 0f);
            }

            int blockMaxQ = startPos + qEnd - 1;
            for (int kvStart = 0; kvStart <= blockMaxQ; kvStart += Bc) {
                int kvEnd = Math.min(seqLen + startPos, kvStart + Bc);
                int BcRows = kvEnd - kvStart;
                if (BcRows <= 0) continue;
                for (int j = 0; j < BcRows; j++) {
                    kvOff[j] = (kvStart + j) * kvDim + kvHeadOffset;
                }

                for (int i0 = 0; i0 < BrRows; i0 += QT) {
                    int qr = Math.min(QT, BrRows - i0);
                    int qBase = (qStart + i0) * queryDim + hHead;
                    if (vec && qr == QT) {
                        qkTile(q, qBase, queryDim, cK, kvOff, 0, BcRows, headSize, scale, S, i0 * BcRows, BcRows);
                    } else {
                        for (int t = 0; t < qr; t++) {
                            int qOffset = (qStart + i0 + t) * queryDim + hHead;
                            for (int j = 0; j < BcRows; j++) {
                                S[(i0 + t) * BcRows + j] = q.dot(qOffset, cK, kvOff[j], headSize) * scale;
                            }
                        }
                    }
                }

                for (int i = 0; i < BrRows; i++) {
                    int globalQ = qStart + i + startPos;
                    int rowBase = i * BcRows;
                    for (int j = 0; j < BcRows; j++) {
                        if (kvStart + j > globalQ) S[rowBase + j] = Float.NEGATIVE_INFINITY;
                    }
                }

                for (int i = 0; i < BrRows; i++) {
                    int rowBase = i * BcRows;
                    float blockMax = Float.NEGATIVE_INFINITY;
                    for (int j = 0; j < BcRows; j++) {
                        float sv = S[rowBase + j];
                        if (sv > blockMax) blockMax = sv;
                    }
                    if (blockMax == Float.NEGATIVE_INFINITY) {
                        for (int j = 0; j < BcRows; j++) S[rowBase + j] = 0f;
                        continue;
                    }
                    float rowM = M[i];
                    double rowL = L[i];
                    float newMax = Math.max(rowM, blockMax);
                    if (newMax > rowM) {
                        float rst = (float) Math.exp(rowM - newMax);
                        normalize(out, (qStart + i) * queryDim + hHead, headSize, rst);
                        rowL *= rst;
                        rowM = newMax;
                    }
                    double sum = 0;
                    for (int j = 0; j < BcRows; j++) {
                        float sv = S[rowBase + j];
                        float p = sv == Float.NEGATIVE_INFINITY ? 0f : (float) Math.exp(sv - rowM);
                        S[rowBase + j] = p;
                        sum += p;
                    }
                    M[i] = rowM;
                    L[i] = rowL + sum;
                }

                for (int i0 = 0; i0 < BrRows; i0 += QT) {
                    int qr = Math.min(QT, BrRows - i0);
                    int oBase = (qStart + i0) * queryDim + hHead;
                    if (vec && qr == QT) {
                        pvTile(out, oBase, queryDim, cV, kvOff, 0, BcRows, headSize, S, i0 * BcRows, BcRows);
                    } else {
                        for (int t = 0; t < qr; t++) {
                            int oOffset = (qStart + i0 + t) * queryDim + hHead;
                            int rowBase = (i0 + t) * BcRows;
                            for (int j = 0; j < BcRows; j++) {
                                float p = S[rowBase + j];
                                if (p != 0f) accumulate(out, oOffset, cV, kvOff[j], headSize, p);
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < BrRows; i++) {
                normalize(out, (qStart + i) * queryDim + hHead, headSize, (float) (1.0 / L[i]));
            }
        });
    }

    /**
     * Sliding-window (or full) block-tiled flash attention over a SPLIT KV source: an already-cached
     * prefix (positions {@code < startPos}, stride {@code kvDim}, addressed through the ring) followed
     * by this chunk's freshly-projected K/V (positions {@code >= startPos}, stride {@code batchKvStride}).
     *
     * <p>{@code window <= 0} is unbounded (full causal); {@code window > 0} attends only
     * {@code [q-window+1, q]}. {@code ringMask} (= {@code ringLen-1}, a power-of-two mask, or {@code 0}
     * for a linear cache) maps a cache position to its physical slot — this is the SWA ring: a slot is
     * reused by {@code pos + ringLen}, which is provably out of every future window, so eviction is a
     * plain overwrite with no bookkeeping. {@code sinks} (nullable) is a per-head attention sink: a
     * virtual key with value 0, so it only adds {@code exp(sink-max)} to each row's softmax denominator
     * (folded once at the final normalize) — the "attend to nothing" escape valve that keeps SWA stable
     * as old keys are evicted. Online softmax + register-tiled QK/PV, parallel over (head, query block);
     * {@code scale} is the QK score scale. Used by LFM2.5 (full or ring-SWA, no sinks) and gpt-oss
     * (ring-SWA/full + sinks); the plain-causal single-source models keep {@link #causalPrefill}.
     */
    static void slidingWindowPrefill(FloatTensor q, FloatTensor out, FloatTensor cK, FloatTensor cV,
                                     FloatTensor bK, FloatTensor bV, int nHeads, int startPos, int seqLen,
                                     int headSize, int kvDim, int queryStride, int batchKvStride, int kvMul,
                                     float scale, int window, int ringMask, FloatTensor sinks) {
        // ringMask must be all-ones (ringLen a power of two) for `pos & ringMask` to equal `pos % ringLen`;
        // the authoritative fail-fast is the per-model config check (e.g. Llama.Configuration), at model
        // creation time. This guards future ring adopters against a silently-wrong addressing mask.
        assert ringMask == 0 || (ringMask & (ringMask + 1)) == 0 : "SWA ring length must be a power of two, got mask " + ringMask;
        boolean vec = FloatTensor.USE_VECTOR_API && q instanceof F32FloatTensor && out instanceof F32FloatTensor;
        F32FloatTensor qF32 = vec ? (F32FloatTensor) q : null;
        F32FloatTensor outF32 = vec ? (F32FloatTensor) out : null;
        int attStart = window > 0 ? Math.max(0, startPos - window + 1) : 0;
        int nQBlocks = (seqLen + Br - 1) / Br;

        Parallel.parallelFor(0, nHeads * nQBlocks, idx -> {
            int h = idx / nQBlocks;
            int qStart = (idx % nQBlocks) * Br;
            Buffers buffers = buffers();
            float[] S = buffers.s;
            float[] M = buffers.m;
            double[] L = buffers.l;
            int[] kvOff = buffers.kvOff;
            int hHead = h * headSize;
            int kvHeadOffset = (h / kvMul) * headSize;

            int qEnd = Math.min(seqLen, qStart + Br);
            int BrRows = qEnd - qStart;
            for (int i = 0; i < BrRows; i++) {
                M[i] = Float.NEGATIVE_INFINITY;
                L[i] = 0.0;
                out.fillInPlace((qStart + i) * queryStride + hHead, headSize, 0f);
            }

            int blockMaxQ = startPos + qEnd - 1;
            for (int kvStart = attStart; kvStart <= blockMaxQ; kvStart += Bc) {
                int kvEnd = Math.min(seqLen + startPos, kvStart + Bc);
                int BcRows = kvEnd - kvStart;
                if (BcRows <= 0) continue;

                // cache keys (stride kvDim, ring-addressed) come first, then this chunk's batch keys (stride batchKvStride)
                int cacheCount = Math.max(0, Math.min(BcRows, startPos - kvStart));
                for (int j = 0; j < BcRows; j++) {
                    int kvPos = kvStart + j;
                    kvOff[j] = kvPos < startPos
                            ? (ringMask != 0 ? (kvPos & ringMask) : kvPos) * kvDim + kvHeadOffset
                            : (kvPos - startPos) * batchKvStride + kvHeadOffset;
                }

                for (int i0 = 0; i0 < BrRows; i0 += QT) {
                    int qr = Math.min(QT, BrRows - i0);
                    int qBase = (qStart + i0) * queryStride + hHead;
                    if (vec && qr == QT) {
                        if (cacheCount > 0) qkTile(qF32, qBase, queryStride, cK, kvOff, 0, cacheCount, headSize, scale, S, i0 * BcRows, BcRows);
                        if (cacheCount < BcRows) qkTile(qF32, qBase, queryStride, bK, kvOff, cacheCount, BcRows - cacheCount, headSize, scale, S, i0 * BcRows, BcRows);
                    } else {
                        for (int t = 0; t < qr; t++) {
                            int qOffset = (qStart + i0 + t) * queryStride + hHead;
                            for (int j = 0; j < BcRows; j++) {
                                S[(i0 + t) * BcRows + j] = q.dot(qOffset, j < cacheCount ? cK : bK, kvOff[j], headSize) * scale;
                            }
                        }
                    }
                }

                for (int i = 0; i < BrRows; i++) {
                    int globalQ = qStart + i + startPos;
                    int qAttStart = window > 0 ? Math.max(0, globalQ - window + 1) : 0;
                    int rowBase = i * BcRows;
                    for (int j = 0; j < BcRows; j++) {
                        int kvPos = kvStart + j;
                        if (kvPos > globalQ || kvPos < qAttStart) S[rowBase + j] = Float.NEGATIVE_INFINITY;
                    }
                }

                for (int i = 0; i < BrRows; i++) {
                    int rowBase = i * BcRows;
                    float blockMax = Float.NEGATIVE_INFINITY;
                    for (int j = 0; j < BcRows; j++) {
                        float s = S[rowBase + j];
                        if (s > blockMax) blockMax = s;
                    }
                    if (blockMax == Float.NEGATIVE_INFINITY) {
                        for (int j = 0; j < BcRows; j++) S[rowBase + j] = 0f;
                        continue;
                    }
                    float rowM = M[i];
                    double rowL = L[i];
                    float newMax = Math.max(rowM, blockMax);
                    if (newMax > rowM) {
                        float rst = (float) Math.exp(rowM - newMax);
                        normalize(out, (qStart + i) * queryStride + hHead, headSize, rst);
                        rowL *= rst;
                        rowM = newMax;
                    }
                    double sum = 0;
                    for (int j = 0; j < BcRows; j++) {
                        float s = S[rowBase + j];
                        float p = s == Float.NEGATIVE_INFINITY ? 0f : (float) Math.exp(s - rowM);
                        S[rowBase + j] = p;
                        sum += p;
                    }
                    M[i] = rowM;
                    L[i] = rowL + sum;
                }

                for (int i0 = 0; i0 < BrRows; i0 += QT) {
                    int qr = Math.min(QT, BrRows - i0);
                    int oBase = (qStart + i0) * queryStride + hHead;
                    if (vec && qr == QT) {
                        if (cacheCount > 0) pvTile(outF32, oBase, queryStride, cV, kvOff, 0, cacheCount, headSize, S, i0 * BcRows, BcRows);
                        if (cacheCount < BcRows) pvTile(outF32, oBase, queryStride, bV, kvOff, cacheCount, BcRows - cacheCount, headSize, S, i0 * BcRows, BcRows);
                    } else {
                        for (int t = 0; t < qr; t++) {
                            int oOffset = (qStart + i0 + t) * queryStride + hHead;
                            int rowBase = (i0 + t) * BcRows;
                            for (int j = 0; j < BcRows; j++) {
                                float p = S[rowBase + j];
                                if (p != 0f) accumulate(out, oOffset, j < cacheCount ? cV : bV, kvOff[j], headSize, p);
                            }
                        }
                    }
                }
            }

            if (sinks == null) {
                for (int i = 0; i < BrRows; i++) {
                    normalize(out, (qStart + i) * queryStride + hHead, headSize, (float) (1.0 / L[i]));
                }
            } else {
                // Fold the per-head sink: a virtual key (score=sink, value=0) adds exp(sink-newM) to the
                // denominator only; out_i, currently scaled to running max M[i], rescales by exp(M[i]-newM).
                float sink = sinks.getFloat(h);
                for (int i = 0; i < BrRows; i++) {
                    float newM = Math.max(M[i], sink);
                    float factor = M[i] == Float.NEGATIVE_INFINITY ? 0f : (float) Math.exp(M[i] - newM);
                    double Lf = L[i] * factor + Math.exp(sink - newM);
                    float inv = Lf == 0.0 ? 0f : (float) (factor / Lf);
                    normalize(out, (qStart + i) * queryStride + hHead, headSize, inv);
                }
            }
        });
    }

    private FlashAttention() {
    }
}
