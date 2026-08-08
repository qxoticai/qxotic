package com.qxotic.jinfer;

import com.oracle.svm.shared.AlwaysInline;
import java.lang.foreign.Arena;
import java.nio.ByteOrder;
import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Online-softmax accumulate and normalize over a head vector, plus the per-thread tile scratch.
 * Architecture-agnostic: it operates on {@link FloatTensor}s, so any {@link Model}'s batched
 * attention can use it. Vector paths cover an F32 output over F32 or F16 values; anything else
 * falls back to scalar.
 */
public final class FlashAttention {

    /**
     * True when AOT-compiled into a native image (property is set during the image build; this
     * class initializes at build time there, so the flag folds to a constant either way).
     */
    static final boolean IN_NATIVE_IMAGE =
            System.getProperty("org.graalvm.nativeimage.imagecode") != null;

    /** Q/K tile sizes for block-tiled prefill (cache-friendly inner loops). */
    static final int Br = 64;

    static final int Bc = 64;

    /**
     * Query-row tile width for the register-tiled QK^T / PV kernels: each key/value vector is
     * loaded (and F16-decoded) once and reused across QT consecutive query rows. Kept at 4 (not 8)
     * because Graal CE intrinsifies the Vector API per method only under a bounded op count: 8
     * accumulators + per-key f16 decode overflow that budget and the whole tile compiles to BOXED
     * vectors (~15x slower, measured) — slower than the per-position rolling fallback. 4 live
     * accumulators stay register-resident and intrinsified (same width as the rolling 4x4 tile),
     * still giving 4x key/value decode reuse over the rolling path's 1x.
     */
    static final int QT = 4;

    /**
     * Per-thread scratch: the Br×Bc score tile, per-row running max/sum, and per-block K/V offsets.
     */
    static final class Buffers {
        final float[] s = new float[Br * Bc];
        final float[] m = new float[Br];
        final double[] l = new double[Br];
        final int[] kvOff = new int[Bc];
        F32FloatTensor kDec, vDec; // F16-cache block decode scratch, grown to Bc*headSize

        F32FloatTensor kDec(int capacity) {
            if (kDec == null || kDec.size() < capacity)
                kDec = F32FloatTensor.allocate(Arena.ofAuto(), capacity);
            return kDec;
        }

        F32FloatTensor vDec(int capacity) {
            if (vDec == null || vDec.size() < capacity)
                vDec = F32FloatTensor.allocate(Arena.ofAuto(), capacity);
            return vDec;
        }
    }

    private static final ThreadLocal<Buffers> BUFFERS = ThreadLocal.withInitial(Buffers::new);

    static Buffers buffers() {
        return BUFFERS.get();
    }

    /**
     * Decodes an F16 cache run into F32 scratch ONCE per kv-block. The F16 tile paths converted
     * every key per row-tile (Br/QT times) through the castShape pipeline - the measured 15x
     * cache-leg tax on F16-cache models. Decoding first runs the fast F32 tiles; f16->f32 uses the
     * SAME vector converter (exact for normals, subnormals flushed), so results are bit-identical
     * to the direct F16 tiles.
     */
    static void decodeF16Run(
            F16FloatTensor src, int[] kvOff, int count, int headSize, F32FloatTensor dst) {
        var sp = FloatTensor.F_SPECIES;
        int len = sp.length();
        int bound = sp.loopBound(headSize);
        for (int j = 0; j < count; j++) {
            int so = kvOff[j];
            long dstByte = dst.vbase + (long) j * headSize * Float.BYTES;
            for (int d = 0; d < bound; d += len) {
                loadF16(src, so + d)
                        .intoMemorySegment(
                                dst.vseg,
                                dstByte + (long) d * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
            }
            for (int d = bound; d < headSize; d++) {
                dst.setFloat((long) j * headSize + d, src.getFloat(so + d));
            }
        }
    }

    // ---- the fused softmax pass: vectorized exp over a score row ----------------------------
    // The polynomial, its constants, its scalar mirror and its ACCURACY CONTRACT live in FastMath
    // (gated by ExpAccuracyTest); the vector body is fused inline below because a helper - even
    // @AlwaysInline - stays boxed under the native-image Vector API expansion (measured 9ns vs
    // 0.19ns per element).

    /** Max over {@code S[base, base+n)} - the block row max feeding the online rescale. */
    static float rowMax(float[] S, int base, int n) {
        int j = 0;
        float max = Float.NEGATIVE_INFINITY;
        if (FloatTensor.USE_VECTOR_API) {
            var sp = FloatTensor.F_SPECIES;
            int bound = sp.loopBound(n);
            if (bound > 0) {
                FloatVector acc = FloatVector.broadcast(sp, Float.NEGATIVE_INFINITY);
                for (; j < bound; j += sp.length()) {
                    acc = acc.max(FloatVector.fromArray(sp, S, base + j));
                }
                max = acc.reduceLanes(VectorOperators.MAX);
            }
        }
        for (; j < n; j++) max = Math.max(max, S[base + j]);
        return max;
    }

    /**
     * The exp leg of the online softmax, fused: {@code S[j] = e^(S[j]-max)} in place over the live
     * columns, returning their sum. One vectorized pass replaces the scalar per-score {@code
     * Math.exp} that used to dominate long-context prefill (measured 2.7ns per score; this is
     * ~0.3ns).
     */
    static double expRowInPlace(float[] S, int base, int n, float max) {
        int j = 0;
        double sum = 0;
        if (FloatTensor.USE_VECTOR_API) {
            var sp = FloatTensor.F_SPECIES;
            int len = sp.length();
            int bound = sp.loopBound(n);
            if (bound > 0) {
                // the exp body is fused INLINE: as a helper (even @AlwaysInline) the native-image
                // Vector API expansion phase leaves it boxed and the pass runs at scalar speed -
                // measured 9ns/element vs 0.19ns fused (the same budget trap as pvTile's split)
                FloatVector mv = FloatVector.broadcast(sp, max);
                FloatVector acc = FloatVector.zero(sp);
                FloatVector vLog2e = FloatVector.broadcast(sp, FastMath.EXP_LOG2E);
                FloatVector vMagic = FloatVector.broadcast(sp, FastMath.EXP_MAGIC);
                FloatVector vHi = FloatVector.broadcast(sp, FastMath.EXP_NLN2_HI);
                FloatVector vLo = FloatVector.broadcast(sp, FastMath.EXP_NLN2_LO);
                FloatVector vC6 = FloatVector.broadcast(sp, FastMath.EXP_C6);
                FloatVector vC5 = FloatVector.broadcast(sp, FastMath.EXP_C5);
                FloatVector vC4 = FloatVector.broadcast(sp, FastMath.EXP_C4);
                FloatVector vC3 = FloatVector.broadcast(sp, FastMath.EXP_C3);
                FloatVector vC2 = FloatVector.broadcast(sp, FastMath.EXP_C2);
                FloatVector vOne = FloatVector.broadcast(sp, 1f);
                FloatVector vZero = FloatVector.zero(sp);
                FloatVector vUnder = FloatVector.broadcast(sp, FastMath.EXP_UNDERFLOW);
                for (; j < bound; j += len) {
                    FloatVector x = FloatVector.fromArray(sp, S, base + j).sub(mv);
                    FloatVector xc = x.max(vUnder);
                    FloatVector t = xc.fma(vLog2e, vMagic);
                    FloatVector nn = t.sub(vMagic);
                    FloatVector r = nn.fma(vHi, xc);
                    r = nn.fma(vLo, r);
                    IntVector e =
                            ((IntVector) nn.convert(VectorOperators.F2I, 0))
                                    .add(127)
                                    .lanewise(VectorOperators.LSHL, 23);
                    FloatVector p = vC6.fma(r, vC5);
                    p = p.fma(r, vC4);
                    p = p.fma(r, vC3);
                    p = p.fma(r, vC2);
                    p = p.fma(r, vOne);
                    p = p.fma(r, vOne);
                    p =
                            p.mul(e.reinterpretAsFloats())
                                    .blend(vZero, x.compare(VectorOperators.LT, vUnder));
                    p.intoArray(S, base + j);
                    acc = acc.add(p);
                }
                sum = acc.reduceLanes(VectorOperators.ADD);
            }
        }
        for (; j < n; j++) {
            float p = FastMath.expNeg(S[base + j] - max);
            S[base + j] = p;
            sum += p;
        }
        return sum;
    }

    /** out[outOffset, +headSize] *= scale (rescale the running output on a new row max). */
    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    static void normalize(FloatTensor out, int outOffset, int headSize, float scale) {
        if (out instanceof F32FloatTensor outF32 && FloatTensor.USE_VECTOR_API) {
            FloatVector scaleVector = FloatVector.broadcast(FloatTensor.F_SPECIES, scale);
            int upperBound = FloatTensor.F_SPECIES.loopBound(headSize);
            for (int i = 0; i < upperBound; i += FloatTensor.F_SPECIES.length()) {
                long byteOffset = (long) (outOffset + i) * Float.BYTES;
                FloatVector.fromMemorySegment(
                                FloatTensor.F_SPECIES,
                                outF32.vseg,
                                outF32.vbase + byteOffset,
                                ByteOrder.LITTLE_ENDIAN)
                        .mul(scaleVector)
                        .intoMemorySegment(
                                outF32.vseg, outF32.vbase + byteOffset, ByteOrder.LITTLE_ENDIAN);
            }
            for (int i = upperBound; i < headSize; i++) {
                outF32.setFloat(outOffset + i, outF32.getFloat(outOffset + i) * scale);
            }
            return;
        }
        out.mapInPlace(outOffset, headSize, v -> v * scale);
    }

    /** out[outOffset, +headSize] += scale * value[valueOffset, +headSize]. */
    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    static void accumulate(
            FloatTensor out,
            int outOffset,
            FloatTensor value,
            int valueOffset,
            int headSize,
            float scale) {
        if (out instanceof F32FloatTensor outF32
                && FloatTensor.USE_VECTOR_API
                && (value instanceof F32FloatTensor || value instanceof F16FloatTensor)) {
            FloatVector scaleVector = FloatVector.broadcast(FloatTensor.F_SPECIES, scale);
            int upperBound = FloatTensor.F_SPECIES.loopBound(headSize);
            if (value instanceof F32FloatTensor valueF32) {
                for (int d = 0; d < upperBound; d += FloatTensor.F_SPECIES.length()) {
                    long byteOffset = (long) (outOffset + d) * Float.BYTES;
                    FloatVector acc =
                            FloatVector.fromMemorySegment(
                                    FloatTensor.F_SPECIES,
                                    outF32.vseg,
                                    outF32.vbase + byteOffset,
                                    ByteOrder.LITTLE_ENDIAN);
                    FloatVector v =
                            FloatVector.fromMemorySegment(
                                    FloatTensor.F_SPECIES,
                                    valueF32.vseg,
                                    valueF32.vbase + (long) (valueOffset + d) * Float.BYTES,
                                    ByteOrder.LITTLE_ENDIAN);
                    v.fma(scaleVector, acc)
                            .intoMemorySegment(
                                    outF32.vseg,
                                    outF32.vbase + byteOffset,
                                    ByteOrder.LITTLE_ENDIAN);
                }
            } else {
                F16FloatTensor f16Value = (F16FloatTensor) value;
                for (int d = 0; d < upperBound; d += FloatTensor.F_SPECIES.length()) {
                    long byteOffset = (long) (outOffset + d) * Float.BYTES;
                    FloatVector acc =
                            FloatVector.fromMemorySegment(
                                    FloatTensor.F_SPECIES,
                                    outF32.vseg,
                                    outF32.vbase + byteOffset,
                                    ByteOrder.LITTLE_ENDIAN);
                    var bits32 =
                            ShortVector.fromMemorySegment(
                                            FloatTensor.S_SPECIES_HALF,
                                            f16Value.vseg,
                                            f16Value.vbase
                                                    + (long) (valueOffset + d) * Float16.BYTES,
                                            ByteOrder.LITTLE_ENDIAN)
                                    .castShape(FloatTensor.I_SPECIES, 0)
                                    .reinterpretAsInts();
                    var zeroExponentMask =
                            bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31);
                    FloatVector v =
                            bits32.and(0x8000)
                                    .lanewise(VectorOperators.LSHL, 16)
                                    .or(
                                            bits32.and(0x7FFF)
                                                    .add(0x1C000)
                                                    .lanewise(VectorOperators.LSHL, 13)
                                                    .and(zeroExponentMask))
                                    .reinterpretAsFloats();
                    v.fma(scaleVector, acc)
                            .intoMemorySegment(
                                    outF32.vseg,
                                    outF32.vbase + byteOffset,
                                    ByteOrder.LITTLE_ENDIAN);
                }
            }
            for (int d = upperBound; d < headSize; d++) {
                outF32.setFloat(
                        outOffset + d,
                        outF32.getFloat(outOffset + d) + value.getFloat(valueOffset + d) * scale);
            }
            return;
        }
        for (int d = 0; d < headSize; d++) {
            out.setFloat(
                    outOffset + d,
                    out.getFloat(outOffset + d) + value.getFloat(valueOffset + d) * scale);
        }
    }

    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    private static void pvTileF16(
            F32FloatTensor out,
            F16FloatTensor value,
            int[] kvOff,
            int runStart,
            int nKeys,
            float[] P,
            VectorSpecies<Float> sp,
            int len,
            int bound,
            int ob0,
            int ob1,
            int ob2,
            int ob3,
            int p0,
            int p1,
            int p2,
            int p3) {
        // Native image: the f16->f32 conversion chain fused into the fma accumulation defeats the
        // AOT Vector API expansion for this graph (the whole component stays boxed; the same code
        // expands fine in isolation). Routing the converted vector through a tiny L1-hot scratch
        // array splits the dataflow into two independently-expandable components. JVM JIT expands
        // the fused form fine, so this detour is image-only.
        float[] conv = IN_NATIVE_IMAGE ? new float[len] : null;
        for (int d = 0; d < bound; d += len) {
            FloatVector o0 = loadF32(out, ob0 + d),
                    o1 = loadF32(out, ob1 + d),
                    o2 = loadF32(out, ob2 + d),
                    o3 = loadF32(out, ob3 + d);
            for (int k = 0; k < nKeys; k++) {
                int col = runStart + k;
                FloatVector v;
                if (IN_NATIVE_IMAGE) {
                    loadF16(value, kvOff[col] + d).intoArray(conv, 0);
                    v = FloatVector.fromArray(sp, conv, 0);
                } else {
                    v = loadF16(value, kvOff[col] + d);
                }
                o0 = v.fma(FloatVector.broadcast(sp, P[p0 + col]), o0);
                o1 = v.fma(FloatVector.broadcast(sp, P[p1 + col]), o1);
                o2 = v.fma(FloatVector.broadcast(sp, P[p2 + col]), o2);
                o3 = v.fma(FloatVector.broadcast(sp, P[p3 + col]), o3);
            }
            o0.intoMemorySegment(
                    out.vseg, out.vbase + (long) (ob0 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            o1.intoMemorySegment(
                    out.vseg, out.vbase + (long) (ob1 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            o2.intoMemorySegment(
                    out.vseg, out.vbase + (long) (ob2 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            o3.intoMemorySegment(
                    out.vseg, out.vbase + (long) (ob3 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
        }
    }

    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    private static void pvTileF32(
            F32FloatTensor out,
            F32FloatTensor value,
            int[] kvOff,
            int runStart,
            int nKeys,
            float[] P,
            VectorSpecies<Float> sp,
            int len,
            int bound,
            int ob0,
            int ob1,
            int ob2,
            int ob3,
            int p0,
            int p1,
            int p2,
            int p3) {
        for (int d = 0; d < bound; d += len) {
            FloatVector o0 = loadF32(out, ob0 + d),
                    o1 = loadF32(out, ob1 + d),
                    o2 = loadF32(out, ob2 + d),
                    o3 = loadF32(out, ob3 + d);
            for (int k = 0; k < nKeys; k++) {
                int col = runStart + k;
                FloatVector v = loadF32(value, kvOff[col] + d);
                o0 = v.fma(FloatVector.broadcast(sp, P[p0 + col]), o0);
                o1 = v.fma(FloatVector.broadcast(sp, P[p1 + col]), o1);
                o2 = v.fma(FloatVector.broadcast(sp, P[p2 + col]), o2);
                o3 = v.fma(FloatVector.broadcast(sp, P[p3 + col]), o3);
            }
            o0.intoMemorySegment(
                    out.vseg, out.vbase + (long) (ob0 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            o1.intoMemorySegment(
                    out.vseg, out.vbase + (long) (ob1 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            o2.intoMemorySegment(
                    out.vseg, out.vbase + (long) (ob2 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            o3.intoMemorySegment(
                    out.vseg, out.vbase + (long) (ob3 + d) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
        }
    }

    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    private static FloatVector loadF32(F32FloatTensor t, int off) {
        return FloatVector.fromMemorySegment(
                FloatTensor.F_SPECIES,
                t.vseg,
                t.vbase + (long) off * Float.BYTES,
                ByteOrder.LITTLE_ENDIAN);
    }

    /** Decode F_SPECIES.length() consecutive F16 values to an F32 vector (IEEE half -> single). */
    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    private static FloatVector loadF16(F16FloatTensor t, int off) {
        return F16FloatTensor.f16ToF32Vector(t.vseg, t.vbase + (long) off * Float16.BYTES);
    }

    /**
     * Register-tiled QK^T for QT consecutive query rows against a run of {@code nKeys} keys from a
     * single source (F32 batch or F16 cache). Each key vector is loaded once and reused across the
     * QT query rows (contraction over {@code headSize}); each score is multiplied by {@code scale}
     * (1.0 when the scale is folded into the query norm). Writes {@code S[(t)*BcRows + runStart+k]}
     * for query row {@code t} in [0,QT) and key {@code k} in [0,nKeys). Key offsets come from
     * {@code kvOff}.
     */
    /**
     * Fully-unrolled QK^T for the ubiquitous headSize=64 on 512-bit lanes. The generic path's
     * dimension loop runs only FOUR iterations per key, so compare/branch/addressing bookkeeping
     * rivals its 16 FMAs (perf: the two hottest instructions in the prefill lambda were a cmp and a
     * mov, not FMAs). Here the query vectors are hoisted across the key loop and the key body is
     * branch-free. F32 keys only: the F16 conversion chain would blow the AOT Vector API expansion
     * budget (the pvTileF16 lesson), and the split prefill decodes its F16 cache to F32 scratch
     * before QK anyway. Accumulation order matches the generic path exactly - scores are
     * bit-identical.
     */
    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    private static void qkTile64F32(
            F32FloatTensor q,
            int qb0,
            int qStride,
            F32FloatTensor key,
            int[] kvOff,
            int runStart,
            int nKeys,
            float scale,
            float[] S,
            int sRow0,
            int BcRows) {
        int qb1 = qb0 + qStride, qb2 = qb0 + 2 * qStride, qb3 = qb0 + 3 * qStride;
        FloatVector q00 = loadF32(q, qb0),
                q01 = loadF32(q, qb0 + 16),
                q02 = loadF32(q, qb0 + 32),
                q03 = loadF32(q, qb0 + 48);
        FloatVector q10 = loadF32(q, qb1),
                q11 = loadF32(q, qb1 + 16),
                q12 = loadF32(q, qb1 + 32),
                q13 = loadF32(q, qb1 + 48);
        FloatVector q20 = loadF32(q, qb2),
                q21 = loadF32(q, qb2 + 16),
                q22 = loadF32(q, qb2 + 32),
                q23 = loadF32(q, qb2 + 48);
        FloatVector q30 = loadF32(q, qb3),
                q31 = loadF32(q, qb3 + 16),
                q32 = loadF32(q, qb3 + 32),
                q33 = loadF32(q, qb3 + 48);
        for (int k = 0; k < nKeys; k++) {
            int ko = kvOff[runStart + k];
            FloatVector k0 = loadF32(key, ko),
                    k1 = loadF32(key, ko + 16),
                    k2 = loadF32(key, ko + 32),
                    k3 = loadF32(key, ko + 48);
            FloatVector a0 = q00.mul(k0);
            a0 = q01.fma(k1, a0);
            a0 = q02.fma(k2, a0);
            a0 = q03.fma(k3, a0);
            FloatVector a1 = q10.mul(k0);
            a1 = q11.fma(k1, a1);
            a1 = q12.fma(k2, a1);
            a1 = q13.fma(k3, a1);
            FloatVector a2 = q20.mul(k0);
            a2 = q21.fma(k1, a2);
            a2 = q22.fma(k2, a2);
            a2 = q23.fma(k3, a2);
            FloatVector a3 = q30.mul(k0);
            a3 = q31.fma(k1, a3);
            a3 = q32.fma(k2, a3);
            a3 = q33.fma(k3, a3);
            int col = runStart + k;
            S[sRow0 + col] = a0.reduceLanes(VectorOperators.ADD) * scale;
            S[sRow0 + BcRows + col] = a1.reduceLanes(VectorOperators.ADD) * scale;
            S[sRow0 + 2 * BcRows + col] = a2.reduceLanes(VectorOperators.ADD) * scale;
            S[sRow0 + 3 * BcRows + col] = a3.reduceLanes(VectorOperators.ADD) * scale;
        }
    }

    /**
     * As {@link #qkTile64F32} for any {@code headSize % 64 == 0} (128, 256): the dimension loop
     * runs in 64-float super-chunks with a fully-unrolled body, cutting the per-iteration
     * bookkeeping 4x versus the generic 16-float chunks. Query vectors are NOT hoisted here (4 rows
     * x headSize/16 would exceed the register file past 64) - the loads stay L1-hot. Accumulation
     * order matches the generic path exactly - scores are bit-identical.
     */
    /**
     * ONE query row of the tile's exact math (single accumulator, ascending chunks, reduce, scalar
     * tail) - the partial row-group fallback ({@code qr < QT}). It must be BIT-IDENTICAL to a tile
     * row: rows land in full or partial groups depending on the chunk shape, and a fallback that
     * rounds differently (q.dot's multi-accumulator order) leaks the chunk shape into the scores -
     * caught by Qwen35CacheRun's cached-vs-uncached reply gate when the F16 caches moved to
     * decode-first.
     */
    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    private static void qkRowF32(
            F32FloatTensor q,
            int qOffset,
            F32FloatTensor key,
            int[] kvOff,
            int runStart,
            int nKeys,
            int headSize,
            float scale,
            float[] S,
            int sRowBase) {
        var sp = FloatTensor.F_SPECIES;
        int len = sp.length();
        int bound = sp.loopBound(headSize);
        for (int k = 0; k < nKeys; k++) {
            int ko = kvOff[runStart + k];
            FloatVector a = FloatVector.zero(sp);
            for (int d = 0; d < bound; d += len) {
                a = loadF32(q, qOffset + d).fma(loadF32(key, ko + d), a);
            }
            float s = a.reduceLanes(VectorOperators.ADD);
            for (int d = bound; d < headSize; d++) {
                s += q.getFloat(qOffset + d) * key.getFloat(ko + d);
            }
            S[sRowBase + runStart + k] = s * scale;
        }
    }

    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    private static void qkTileWideF32(
            F32FloatTensor q,
            int qb0,
            int qStride,
            F32FloatTensor key,
            int[] kvOff,
            int runStart,
            int nKeys,
            int headSize,
            float scale,
            float[] S,
            int sRow0,
            int BcRows) {
        int qb1 = qb0 + qStride, qb2 = qb0 + 2 * qStride, qb3 = qb0 + 3 * qStride;
        for (int k = 0; k < nKeys; k++) {
            int ko = kvOff[runStart + k];
            FloatVector a0 = FloatVector.zero(FloatTensor.F_SPECIES),
                    a1 = FloatVector.zero(FloatTensor.F_SPECIES),
                    a2 = FloatVector.zero(FloatTensor.F_SPECIES),
                    a3 = FloatVector.zero(FloatTensor.F_SPECIES);
            for (int d = 0; d < headSize; d += 64) {
                FloatVector k0 = loadF32(key, ko + d),
                        k1 = loadF32(key, ko + d + 16),
                        k2 = loadF32(key, ko + d + 32),
                        k3 = loadF32(key, ko + d + 48);
                a0 = loadF32(q, qb0 + d).fma(k0, a0);
                a0 = loadF32(q, qb0 + d + 16).fma(k1, a0);
                a0 = loadF32(q, qb0 + d + 32).fma(k2, a0);
                a0 = loadF32(q, qb0 + d + 48).fma(k3, a0);
                a1 = loadF32(q, qb1 + d).fma(k0, a1);
                a1 = loadF32(q, qb1 + d + 16).fma(k1, a1);
                a1 = loadF32(q, qb1 + d + 32).fma(k2, a1);
                a1 = loadF32(q, qb1 + d + 48).fma(k3, a1);
                a2 = loadF32(q, qb2 + d).fma(k0, a2);
                a2 = loadF32(q, qb2 + d + 16).fma(k1, a2);
                a2 = loadF32(q, qb2 + d + 32).fma(k2, a2);
                a2 = loadF32(q, qb2 + d + 48).fma(k3, a2);
                a3 = loadF32(q, qb3 + d).fma(k0, a3);
                a3 = loadF32(q, qb3 + d + 16).fma(k1, a3);
                a3 = loadF32(q, qb3 + d + 32).fma(k2, a3);
                a3 = loadF32(q, qb3 + d + 48).fma(k3, a3);
            }
            int col = runStart + k;
            S[sRow0 + col] = a0.reduceLanes(VectorOperators.ADD) * scale;
            S[sRow0 + BcRows + col] = a1.reduceLanes(VectorOperators.ADD) * scale;
            S[sRow0 + 2 * BcRows + col] = a2.reduceLanes(VectorOperators.ADD) * scale;
            S[sRow0 + 3 * BcRows + col] = a3.reduceLanes(VectorOperators.ADD) * scale;
        }
    }

    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    static void qkTile(
            F32FloatTensor q,
            int qBase,
            int qStride,
            FloatTensor key,
            int[] kvOff,
            int runStart,
            int nKeys,
            int headSize,
            float scale,
            float[] S,
            int sRow0,
            int BcRows) {
        var sp = FloatTensor.F_SPECIES;
        int len = sp.length();
        if (len == 16 && key instanceof F32FloatTensor k32) {
            if (headSize == 64) {
                qkTile64F32(
                        q, qBase, qStride, k32, kvOff, runStart, nKeys, scale, S, sRow0, BcRows);
                return;
            }
            if ((headSize & 63) == 0) {
                qkTileWideF32(
                        q, qBase, qStride, k32, kvOff, runStart, nKeys, headSize, scale, S, sRow0,
                        BcRows);
                return;
            }
        }
        int bound = sp.loopBound(headSize);
        int qb0 = qBase,
                qb1 = qBase + qStride,
                qb2 = qBase + 2 * qStride,
                qb3 = qBase + 3 * qStride;
        boolean f16 = key instanceof F16FloatTensor;
        F32FloatTensor kf32 = f16 ? null : (F32FloatTensor) key;
        F16FloatTensor kf16 = f16 ? (F16FloatTensor) key : null;
        for (int k = 0; k < nKeys; k++) {
            int ko = kvOff[runStart + k];
            FloatVector a0 = FloatVector.zero(sp),
                    a1 = FloatVector.zero(sp),
                    a2 = FloatVector.zero(sp),
                    a3 = FloatVector.zero(sp);
            if (f16) {
                for (int d = 0; d < bound; d += len) {
                    FloatVector kv = loadF16(kf16, ko + d);
                    a0 = loadF32(q, qb0 + d).fma(kv, a0);
                    a1 = loadF32(q, qb1 + d).fma(kv, a1);
                    a2 = loadF32(q, qb2 + d).fma(kv, a2);
                    a3 = loadF32(q, qb3 + d).fma(kv, a3);
                }
            } else {
                for (int d = 0; d < bound; d += len) {
                    FloatVector kv = loadF32(kf32, ko + d);
                    a0 = loadF32(q, qb0 + d).fma(kv, a0);
                    a1 = loadF32(q, qb1 + d).fma(kv, a1);
                    a2 = loadF32(q, qb2 + d).fma(kv, a2);
                    a3 = loadF32(q, qb3 + d).fma(kv, a3);
                }
            }
            float s0 = a0.reduceLanes(VectorOperators.ADD),
                    s1 = a1.reduceLanes(VectorOperators.ADD);
            float s2 = a2.reduceLanes(VectorOperators.ADD),
                    s3 = a3.reduceLanes(VectorOperators.ADD);
            for (int d = bound; d < headSize; d++) {
                float kvf = key.getFloat(ko + d);
                s0 += q.getFloat(qb0 + d) * kvf;
                s1 += q.getFloat(qb1 + d) * kvf;
                s2 += q.getFloat(qb2 + d) * kvf;
                s3 += q.getFloat(qb3 + d) * kvf;
            }
            int col = runStart + k;
            S[sRow0 + col] = s0 * scale;
            S[sRow0 + BcRows + col] = s1 * scale;
            S[sRow0 + 2 * BcRows + col] = s2 * scale;
            S[sRow0 + 3 * BcRows + col] = s3 * scale;
        }
    }

    /**
     * Register-tiled PV for QT consecutive query rows over a run of {@code nKeys} values from a
     * single source (F32 batch or F16 cache). Each value vector is loaded once per chunk and reused
     * across the QT rows; each row's output is read+written once per chunk. Adds {@code sum_k
     * P[t][k]*V_k} into {@code out} for query row {@code t}; probabilities come from {@code
     * P[(t)*BcRows + runStart+k]}.
     */
    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    static void pvTile(
            F32FloatTensor out,
            int oBase,
            int oStride,
            FloatTensor value,
            int[] kvOff,
            int runStart,
            int nKeys,
            int headSize,
            float[] P,
            int pRow0,
            int BcRows) {
        var sp = FloatTensor.F_SPECIES;
        int len = sp.length();
        int bound = sp.loopBound(headSize);
        int ob0 = oBase,
                ob1 = oBase + oStride,
                ob2 = oBase + 2 * oStride,
                ob3 = oBase + 3 * oStride;
        int p0 = pRow0, p1 = pRow0 + BcRows, p2 = pRow0 + 2 * BcRows, p3 = pRow0 + 3 * BcRows;
        // The vectorized body is specialized per element type in SEPARATE methods: a per-iteration
        // f16/f32 select feeding the loop-carried o0..o3 phis defeats native-image Vector API
        // expansion, and even as two loops in one body the method exceeds what the AOT expansion
        // phase converts (one loop stayed boxed; measured). One method per element type keeps each
        // compilation unit small enough to expand fully. HotSpot JIT is indifferent to the split.
        if (value instanceof F16FloatTensor vf16) {
            pvTileF16(
                    out, vf16, kvOff, runStart, nKeys, P, sp, len, bound, ob0, ob1, ob2, ob3, p0,
                    p1, p2, p3);
        } else {
            pvTileF32(
                    out,
                    (F32FloatTensor) value,
                    kvOff,
                    runStart,
                    nKeys,
                    P,
                    sp,
                    len,
                    bound,
                    ob0,
                    ob1,
                    ob2,
                    ob3,
                    p0,
                    p1,
                    p2,
                    p3);
        }
        for (int d = bound; d < headSize; d++) {
            float r0 = out.getFloat(ob0 + d),
                    r1 = out.getFloat(ob1 + d),
                    r2 = out.getFloat(ob2 + d),
                    r3 = out.getFloat(ob3 + d);
            for (int k = 0; k < nKeys; k++) {
                int col = runStart + k;
                float vf = value.getFloat(kvOff[col] + d);
                r0 += P[p0 + col] * vf;
                r1 += P[p1 + col] * vf;
                r2 += P[p2 + col] * vf;
                r3 += P[p3 + col] * vf;
            }
            out.setFloat(ob0 + d, r0);
            out.setFloat(ob1 + d, r1);
            out.setFloat(ob2 + d, r2);
            out.setFloat(ob3 + d, r3);
        }
    }

    /**
     * Block-tiled causal flash attention over a contiguous full-context F32 KV cache (scale =
     * 1/sqrt(headSize)). Q/output are packed at stride {@code queryDim}, the cache at stride {@code
     * kvDim}; GQA via {@code kvMul}. Online softmax, register-tiled QK/PV, parallel over (head,
     * query block). Writes the attention output into {@code out}. Shared by the plain-causal models
     * (Llama3/Nemotron/Qwen3.5); GptOss keeps its own variant for attention sinks + SWA.
     */
    public static void causalPrefill(
            F32FloatTensor q,
            F32FloatTensor out,
            FloatTensor cK,
            FloatTensor cV,
            int nHeads,
            int startPos,
            int seqLen,
            int headSize,
            int kvDim,
            int queryDim,
            int kvMul) {
        causalPrefill(
                q,
                out,
                cK,
                cV,
                nHeads,
                startPos,
                seqLen,
                headSize,
                kvDim,
                queryDim,
                kvMul,
                1.0f / (float) Math.sqrt(headSize));
    }

    /**
     * As above, with an explicit QK score {@code scale} (Granite uses a custom attention scale
     * rather than 1/sqrt(headSize)).
     */
    static void causalPrefill(
            F32FloatTensor q,
            F32FloatTensor out,
            FloatTensor cK,
            FloatTensor cV,
            int nHeads,
            int startPos,
            int seqLen,
            int headSize,
            int kvDim,
            int queryDim,
            int kvMul,
            float scale) {
        boolean vec = FloatTensor.USE_VECTOR_API; // qkTile/pvTile handle both F32 and F16 caches
        int nQBlocks = (seqLen + Br - 1) / Br;

        Parallel.parallelFor(
                0,
                nHeads * nQBlocks,
                idx -> {
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
                        // an F16 cache decodes to F32 scratch ONCE per block, exactly like the
                        // split prefill: 16x fewer conversions than the direct-F16 tiles (once
                        // per block instead of once per QT-group), and the F32 tiles unlock the
                        // unrolled qkTile64/Wide specializations. Bit-identical either way - the
                        // decoder is the same converter the direct tiles use.
                        FloatTensor blockK = cK;
                        FloatTensor blockV = cV;
                        if (cK instanceof F16FloatTensor k16) {
                            F32FloatTensor kd = buf.kDec(Bc * headSize);
                            F32FloatTensor vd = buf.vDec(Bc * headSize);
                            decodeF16Run(k16, kvOff, BcRows, headSize, kd);
                            decodeF16Run((F16FloatTensor) cV, kvOff, BcRows, headSize, vd);
                            for (int j = 0; j < BcRows; j++) kvOff[j] = j * headSize;
                            blockK = kd;
                            blockV = vd;
                        }

                        for (int i0 = 0; i0 < BrRows; i0 += QT) {
                            int qr = Math.min(QT, BrRows - i0);
                            int qBase = (qStart + i0) * queryDim + hHead;
                            if (vec && qr == QT) {
                                qkTile(
                                        q,
                                        qBase,
                                        queryDim,
                                        blockK,
                                        kvOff,
                                        0,
                                        BcRows,
                                        headSize,
                                        scale,
                                        S,
                                        i0 * BcRows,
                                        BcRows);
                            } else {
                                for (int t = 0; t < qr; t++) {
                                    int qOffset = (qStart + i0 + t) * queryDim + hHead;
                                    if (vec && blockK instanceof F32FloatTensor bk32) {
                                        // tile-order math: chunk shape must not leak into scores
                                        qkRowF32(
                                                q,
                                                qOffset,
                                                bk32,
                                                kvOff,
                                                0,
                                                BcRows,
                                                headSize,
                                                scale,
                                                S,
                                                (i0 + t) * BcRows);
                                        continue;
                                    }
                                    for (int j = 0; j < BcRows; j++) {
                                        S[(i0 + t) * BcRows + j] =
                                                q.dot(qOffset, blockK, kvOff[j], headSize) * scale;
                                    }
                                }
                            }
                        }

                        for (int i = 0; i < BrRows; i++) {
                            int globalQ = qStart + i + startPos;
                            int rowBase = i * BcRows;
                            // causal masking is a SUFFIX of the row (columns past globalQ):
                            // zeroing it directly replaces the -inf pass, and every live score
                            // stays finite - which is what lets the exp pass vectorize
                            int live = Math.min(BcRows, globalQ - kvStart + 1);
                            if (live <= 0) {
                                Arrays.fill(S, rowBase, rowBase + BcRows, 0f);
                                continue;
                            }
                            Arrays.fill(S, rowBase + live, rowBase + BcRows, 0f);
                            float blockMax = rowMax(S, rowBase, live);
                            float rowM = M[i];
                            double rowL = L[i];
                            float newMax = Math.max(rowM, blockMax);
                            if (newMax > rowM) {
                                float rst = FastMath.expNeg(rowM - newMax);
                                normalize(out, (qStart + i) * queryDim + hHead, headSize, rst);
                                rowL *= rst;
                                rowM = newMax;
                            }
                            M[i] = rowM;
                            L[i] = rowL + expRowInPlace(S, rowBase, live, rowM);
                        }

                        for (int i0 = 0; i0 < BrRows; i0 += QT) {
                            int qr = Math.min(QT, BrRows - i0);
                            int oBase = (qStart + i0) * queryDim + hHead;
                            if (vec && qr == QT) {
                                pvTile(
                                        out,
                                        oBase,
                                        queryDim,
                                        blockV,
                                        kvOff,
                                        0,
                                        BcRows,
                                        headSize,
                                        S,
                                        i0 * BcRows,
                                        BcRows);
                            } else {
                                for (int t = 0; t < qr; t++) {
                                    int oOffset = (qStart + i0 + t) * queryDim + hHead;
                                    int rowBase = (i0 + t) * BcRows;
                                    for (int j = 0; j < BcRows; j++) {
                                        float p = S[rowBase + j];
                                        if (p != 0f)
                                            accumulate(out, oOffset, blockV, kvOff[j], headSize, p);
                                    }
                                }
                            }
                        }
                    }

                    for (int i = 0; i < BrRows; i++) {
                        normalize(
                                out,
                                (qStart + i) * queryDim + hHead,
                                headSize,
                                (float) (1.0 / L[i]));
                    }
                });
    }

    /**
     * Sliding-window (or full) block-tiled flash attention over a SPLIT KV source: an
     * already-cached prefix (positions {@code < startPos}, stride {@code kvDim}, addressed through
     * the ring) followed by this chunk's freshly-projected K/V (positions {@code >= startPos},
     * stride {@code batchKvStride}).
     *
     * <p>{@code window <= 0} is unbounded (full causal); {@code window > 0} attends only {@code
     * [q-window+1, q]}. {@code ringMask} (= {@code ringLen-1}, a power-of-two mask, or {@code 0}
     * for a linear cache) maps a cache position to its physical slot — this is the SWA ring: a slot
     * is reused by {@code pos + ringLen}, which is provably out of every future window, so eviction
     * is a plain overwrite with no bookkeeping. {@code sinks} (nullable) is a per-head attention
     * sink: a virtual key with value 0, so it only adds {@code exp(sink-max)} to each row's softmax
     * denominator (folded once at the final normalize) — the "attend to nothing" escape valve that
     * keeps SWA stable as old keys are evicted. Online softmax + register-tiled QK/PV, parallel
     * over (head, query block); {@code scale} is the QK score scale. Used by LFM2.5 (full or
     * ring-SWA, no sinks) and gpt-oss (ring-SWA/full + sinks); the plain-causal single-source
     * models keep {@link #causalPrefill}.
     */
    public static void slidingWindowPrefill(
            FloatTensor q,
            FloatTensor out,
            FloatTensor cK,
            FloatTensor cV,
            FloatTensor bK,
            FloatTensor bV,
            int nHeads,
            int startPos,
            int seqLen,
            int headSize,
            int kvDim,
            int queryStride,
            int batchKvStride,
            int kvMul,
            float scale,
            int window,
            int ringMask,
            FloatTensor sinks) {
        prefill(
                q,
                out,
                cK,
                cV,
                bK,
                bV,
                nHeads,
                startPos,
                seqLen,
                headSize,
                kvDim,
                queryStride,
                batchKvStride,
                kvMul,
                scale,
                window,
                ringMask,
                sinks,
                false);
    }

    /**
     * Prefix-LM prefill: the chunk attends causally to the cached prefix ({@code startPos} keys)
     * but bidirectionally within itself (every chunk row sees every chunk row). Used for gemma
     * multimodal image blocks, which the decoder attends to non-causally
     * (mtmd_decode_use_non_causal).
     */
    public static void slidingWindowPrefill(
            FloatTensor q,
            FloatTensor out,
            FloatTensor cK,
            FloatTensor cV,
            FloatTensor bK,
            FloatTensor bV,
            int nHeads,
            int startPos,
            int seqLen,
            int headSize,
            int kvDim,
            int queryStride,
            int batchKvStride,
            int kvMul,
            float scale,
            int window,
            int ringMask,
            FloatTensor sinks,
            boolean bidir) {
        prefill(
                q,
                out,
                cK,
                cV,
                bK,
                bV,
                nHeads,
                startPos,
                seqLen,
                headSize,
                kvDim,
                queryStride,
                batchKvStride,
                kvMul,
                scale,
                window,
                ringMask,
                sinks,
                bidir);
    }

    /**
     * Bidirectional (non-causal) full attention over a single K/V source (e.g. a ViT): every query
     * attends to every key. Online-softmax, no materialized score matrix; K/V may be F16. {@code
     * bK}/{@code bV} are the keys/values ([seqLen, kvDim]); {@code stride} is the per-row stride of
     * q/out.
     */
    public static void bidirectionalPrefill(
            FloatTensor q,
            FloatTensor out,
            FloatTensor bK,
            FloatTensor bV,
            int nHeads,
            int seqLen,
            int headSize,
            int kvDim,
            int stride,
            int kvMul,
            float scale) {
        prefill(
                q, out, bK, bV, bK, bV, nHeads, 0, seqLen, headSize, kvDim, stride, kvDim, kvMul,
                scale, 0, 0, null, true);
    }

    private static void prefill(
            FloatTensor q,
            FloatTensor out,
            FloatTensor cK,
            FloatTensor cV,
            FloatTensor bK,
            FloatTensor bV,
            int nHeads,
            int startPos,
            int seqLen,
            int headSize,
            int kvDim,
            int queryStride,
            int batchKvStride,
            int kvMul,
            float scale,
            int window,
            int ringMask,
            FloatTensor sinks,
            boolean bidir) {
        // ringMask must be all-ones (ringLen a power of two) for `pos & ringMask` to equal `pos %
        // ringLen`;
        // the authoritative fail-fast is the per-model config check (e.g. Llama.Configuration), at
        // model
        // creation time. This guards future ring adopters against a silently-wrong addressing mask.
        assert ringMask == 0 || (ringMask & (ringMask + 1)) == 0
                : "SWA ring length must be a power of two, got mask " + ringMask;
        boolean vec =
                FloatTensor.USE_VECTOR_API
                        && q instanceof F32FloatTensor
                        && out instanceof F32FloatTensor;
        F32FloatTensor qF32 = vec ? (F32FloatTensor) q : null;
        F32FloatTensor outF32 = vec ? (F32FloatTensor) out : null;
        int attStart = window > 0 ? Math.max(0, startPos - window + 1) : 0;
        int nQBlocks = (seqLen + Br - 1) / Br;

        Parallel.parallelFor(
                0,
                nHeads * nQBlocks,
                idx -> {
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

                    int blockMaxQ =
                            bidir
                                    ? startPos + seqLen - 1
                                    : startPos + qEnd - 1; // bidir: every query sees all keys
                    for (int kvStart = attStart; kvStart <= blockMaxQ; kvStart += Bc) {
                        int kvEnd = Math.min(seqLen + startPos, kvStart + Bc);
                        int BcRows = kvEnd - kvStart;
                        if (BcRows <= 0) continue;

                        // cache keys (stride kvDim, ring-addressed) come first, then this chunk's
                        // batch keys (stride batchKvStride)
                        int cacheCount = Math.max(0, Math.min(BcRows, startPos - kvStart));
                        for (int j = 0; j < BcRows; j++) {
                            int kvPos = kvStart + j;
                            kvOff[j] =
                                    kvPos < startPos
                                            ? (ringMask != 0 ? (kvPos & ringMask) : kvPos) * kvDim
                                                    + kvHeadOffset
                                            : (kvPos - startPos) * batchKvStride + kvHeadOffset;
                        }
                        FloatTensor cacheK = cK;
                        FloatTensor cacheV = cV;
                        if (cacheCount > 0 && cK instanceof F16FloatTensor k16) {
                            F32FloatTensor kd = buffers.kDec(Bc * headSize);
                            F32FloatTensor vd = buffers.vDec(Bc * headSize);
                            decodeF16Run(k16, kvOff, cacheCount, headSize, kd);
                            decodeF16Run((F16FloatTensor) cV, kvOff, cacheCount, headSize, vd);
                            for (int j = 0; j < cacheCount; j++) kvOff[j] = j * headSize;
                            cacheK = kd;
                            cacheV = vd;
                        }

                        for (int i0 = 0; i0 < BrRows; i0 += QT) {
                            int qr = Math.min(QT, BrRows - i0);
                            int qBase = (qStart + i0) * queryStride + hHead;
                            if (vec && qr == QT) {
                                if (cacheCount > 0)
                                    qkTile(
                                            qF32,
                                            qBase,
                                            queryStride,
                                            cacheK,
                                            kvOff,
                                            0,
                                            cacheCount,
                                            headSize,
                                            scale,
                                            S,
                                            i0 * BcRows,
                                            BcRows);
                                if (cacheCount < BcRows)
                                    qkTile(
                                            qF32,
                                            qBase,
                                            queryStride,
                                            bK,
                                            kvOff,
                                            cacheCount,
                                            BcRows - cacheCount,
                                            headSize,
                                            scale,
                                            S,
                                            i0 * BcRows,
                                            BcRows);
                            } else {
                                for (int t = 0; t < qr; t++) {
                                    int qOffset = (qStart + i0 + t) * queryStride + hHead;
                                    // deliberately NOT qkRowF32: this path's models (lfm2,
                                    // gptoss, llama chunked) hold their behavior gates against
                                    // the historical dot fallback - swapping it flipped LFM2.5's
                                    // borderline multi-turn tool loop. The causal path DOES use
                                    // qkRowF32 (its byte gate demands tile-consistent rows);
                                    // aligning this one belongs to the chunk-shape golden item.
                                    for (int j = 0; j < BcRows; j++) {
                                        S[(i0 + t) * BcRows + j] =
                                                q.dot(
                                                                qOffset,
                                                                j < cacheCount ? cacheK : bK,
                                                                kvOff[j],
                                                                headSize)
                                                        * scale;
                                    }
                                }
                            }
                        }

                        for (int i = 0; i < BrRows; i++) {
                            int globalQ = qStart + i + startPos;
                            int qAttStart = window > 0 ? Math.max(0, globalQ - window + 1) : 0;
                            int rowBase = i * BcRows;
                            // masking is a PREFIX (window) plus a SUFFIX (causal) of the row:
                            // zeroing them directly keeps every live score finite, which is what
                            // lets the exp pass vectorize (see expRowInPlace)
                            int lo = Math.min(BcRows, Math.max(0, qAttStart - kvStart));
                            int hi = bidir ? BcRows : Math.min(BcRows, globalQ - kvStart + 1);
                            if (hi <= lo) {
                                Arrays.fill(S, rowBase, rowBase + BcRows, 0f);
                                continue;
                            }
                            Arrays.fill(S, rowBase, rowBase + lo, 0f);
                            Arrays.fill(S, rowBase + hi, rowBase + BcRows, 0f);
                            float blockMax = rowMax(S, rowBase + lo, hi - lo);
                            float rowM = M[i];
                            double rowL = L[i];
                            float newMax = Math.max(rowM, blockMax);
                            if (newMax > rowM) {
                                float rst = FastMath.expNeg(rowM - newMax);
                                normalize(out, (qStart + i) * queryStride + hHead, headSize, rst);
                                rowL *= rst;
                                rowM = newMax;
                            }
                            M[i] = rowM;
                            L[i] = rowL + expRowInPlace(S, rowBase + lo, hi - lo, rowM);
                        }

                        for (int i0 = 0; i0 < BrRows; i0 += QT) {
                            int qr = Math.min(QT, BrRows - i0);
                            int oBase = (qStart + i0) * queryStride + hHead;
                            if (vec && qr == QT) {
                                if (cacheCount > 0)
                                    pvTile(
                                            outF32,
                                            oBase,
                                            queryStride,
                                            cacheV,
                                            kvOff,
                                            0,
                                            cacheCount,
                                            headSize,
                                            S,
                                            i0 * BcRows,
                                            BcRows);
                                if (cacheCount < BcRows)
                                    pvTile(
                                            outF32,
                                            oBase,
                                            queryStride,
                                            bV,
                                            kvOff,
                                            cacheCount,
                                            BcRows - cacheCount,
                                            headSize,
                                            S,
                                            i0 * BcRows,
                                            BcRows);
                            } else {
                                for (int t = 0; t < qr; t++) {
                                    int oOffset = (qStart + i0 + t) * queryStride + hHead;
                                    int rowBase = (i0 + t) * BcRows;
                                    for (int j = 0; j < BcRows; j++) {
                                        float p = S[rowBase + j];
                                        if (p != 0f)
                                            accumulate(
                                                    out,
                                                    oOffset,
                                                    j < cacheCount ? cacheV : bV,
                                                    kvOff[j],
                                                    headSize,
                                                    p);
                                    }
                                }
                            }
                        }
                    }

                    if (sinks == null) {
                        for (int i = 0; i < BrRows; i++) {
                            normalize(
                                    out,
                                    (qStart + i) * queryStride + hHead,
                                    headSize,
                                    (float) (1.0 / L[i]));
                        }
                    } else {
                        // Fold the per-head sink: a virtual key (score=sink, value=0) adds
                        // exp(sink-newM) to the
                        // denominator only; out_i, currently scaled to running max M[i], rescales
                        // by exp(M[i]-newM).
                        float sink = sinks.getFloat(h);
                        for (int i = 0; i < BrRows; i++) {
                            float newM = Math.max(M[i], sink);
                            float factor =
                                    M[i] == Float.NEGATIVE_INFINITY
                                            ? 0f
                                            : (float) Math.exp(M[i] - newM);
                            double Lf = L[i] * factor + Math.exp(sink - newM);
                            float inv = Lf == 0.0 ? 0f : (float) (factor / Lf);
                            normalize(out, (qStart + i) * queryStride + hHead, headSize, inv);
                        }
                    }
                });
    }

    /**
     * Online-softmax attention for a SINGLE query position (decode), parallel over heads. Streams
     * keys with a running max/sum so the score row is never materialized. The decode counterpart to
     * {@link #slidingWindowPrefill}; shared by every model's {@code seqLen == 1} path.
     *
     * <p>Keys/values for positions {@code [attStart, position)} come from the cache {@code cK/cV},
     * ring-addressed via {@code ringMask} ({@code = window-1} for SWA, {@code 0} for a linear
     * cache). The current token (position {@code position}) is read from the batch buffer {@code
     * bK/bV} at {@code (head/kvMul)*headSize} when {@code bK != null}; when {@code bK == null} the
     * caller has already written the current token into the cache, so it is read from there like
     * any other key. Each score is multiplied by {@code scale}. {@code sinks} (nullable) is a
     * per-head attention sink: a virtual key (score=sink, value 0) that adds {@code exp(sink-max)}
     * to the denominator only. Writes the attention output into {@code out} (heads packed at stride
     * {@code headSize}).
     */
    static void rollingDecode(
            F32FloatTensor q,
            F32FloatTensor out,
            FloatTensor cK,
            FloatTensor cV,
            FloatTensor bK,
            FloatTensor bV,
            int nHeads,
            int position,
            int attStart,
            int headSize,
            int kvDim,
            int kvMul,
            float scale,
            int ringMask,
            FloatTensor sinks) {
        Parallel.parallelFor(
                0,
                nHeads,
                h -> {
                    int kvHeadOffset = (h / kvMul) * headSize;
                    int qOff = h * headSize;
                    out.fillInPlace(qOff, headSize, 0f);
                    float m = Float.NEGATIVE_INFINITY;
                    double l = 0.0;
                    for (int t = attStart; t <= position; t++) {
                        boolean cur = bK != null && t >= position;
                        int off =
                                cur
                                        ? kvHeadOffset
                                        : ((ringMask != 0 ? (t & ringMask) : t) * kvDim
                                                + kvHeadOffset);
                        float score = q.dot(qOff, cur ? bK : cK, off, headSize) * scale;
                        if (score > m) { // new running max -> rescale accumulator
                            float rescale = (float) Math.exp(m - score);
                            FlashAttention.normalize(out, qOff, headSize, rescale);
                            l *= rescale;
                            m = score;
                        }
                        float p = (float) Math.exp(score - m);
                        FlashAttention.accumulate(out, qOff, cur ? bV : cV, off, headSize, p);
                        l += p;
                    }
                    if (sinks == null) {
                        normalize(out, qOff, headSize, (float) (1.0 / l));
                    } else { // fold the per-head sink into the denominator
                        float sink = sinks.getFloat(h);
                        float newM = Math.max(m, sink);
                        float factor =
                                m == Float.NEGATIVE_INFINITY ? 0f : (float) Math.exp(m - newM);
                        double denom = l * factor + Math.exp(sink - newM);
                        normalize(
                                out, qOff, headSize, denom == 0.0 ? 0f : (float) (factor / denom));
                    }
                });
    }

    /**
     * Reusable per-sequence scratch for {@link #flashDecode} partials (one per {@code State}).
     * Draws from the state's arena so it dies with the state - it is state memory, and an ofAuto
     * buffer here survived close() in every family (plus one orphaned copy per lazy regrowth).
     */
    public static final class DecodeScratch {
        private final Arena arena;
        F32FloatTensor o;
        float[] m;
        double[] l;

        public DecodeScratch(Arena arena) {
            this.arena = arena;
        }

        void ensure(int totalPartials, int headSize) {
            int oFloats = totalPartials * headSize;
            if (o == null || o.size() < oFloats) o = F32FloatTensor.allocate(arena, oFloats);
            if (m == null || m.length < totalPartials) m = new float[totalPartials];
            if (l == null || l.length < totalPartials) l = new double[totalPartials];
        }
    }

    /**
     * Flash-decoding for a SINGLE query (decode): same result as {@link #rollingDecode} but the
     * attended range is split into {@code nPartitions} slices computed in parallel — each does its
     * own online softmax into a partial (O, m, l), then the partials are merged per head. Breaking
     * the long serial running-max/sum chain across cores lowers per-token latency at long context
     * (measured +3% at 4k, +10% at 16k on a 32-head model). Below {@link
     * RuntimeFlags#DECODE_BLOCK_SIZE} keys there is nothing to gain, so it falls through to {@link
     * #rollingDecode}. All other arguments (cache ± batch buffer, {@code ringMask}, {@code scale},
     * {@code sinks}) match {@code rollingDecode}.
     */
    public static void flashDecode(
            F32FloatTensor q,
            F32FloatTensor out,
            FloatTensor cK,
            FloatTensor cV,
            FloatTensor bK,
            FloatTensor bV,
            int nHeads,
            int position,
            int attStart,
            int headSize,
            int kvDim,
            int kvMul,
            float scale,
            int ringMask,
            FloatTensor sinks,
            DecodeScratch scratch) {
        int range = position - attStart + 1;
        int nParts =
                Math.max(
                        1,
                        Math.min(
                                RuntimeFlags.DECODE_THREADS,
                                range / RuntimeFlags.DECODE_BLOCK_SIZE + 1));
        if (nParts == 1) {
            rollingDecode(
                    q, out, cK, cV, bK, bV, nHeads, position, attStart, headSize, kvDim, kvMul,
                    scale, ringMask, sinks);
            return;
        }
        int blockSize = range / nParts;
        int totalPartials = nParts * nHeads;
        scratch.ensure(totalPartials, headSize);
        F32FloatTensor pO = scratch.o;
        float[] pM = scratch.m;
        double[] pL = scratch.l;

        // Each (partition, head) does an independent online softmax over its key slice into a
        // partial.
        Parallel.parallelFor(
                0,
                totalPartials,
                task -> {
                    int p = task / nHeads, h = task - p * nHeads;
                    int tStart = attStart + p * blockSize;
                    int tEnd = (p + 1 == nParts) ? position + 1 : attStart + (p + 1) * blockSize;
                    int kvHeadOffset = (h / kvMul) * headSize;
                    int qOff = h * headSize;
                    int oOff = task * headSize;
                    pO.fillInPlace(oOff, headSize, 0f);
                    float m = Float.NEGATIVE_INFINITY;
                    double l = 0.0;
                    for (int t = tStart; t < tEnd; t++) {
                        boolean cur = bK != null && t >= position;
                        int off =
                                cur
                                        ? kvHeadOffset
                                        : ((ringMask != 0 ? (t & ringMask) : t) * kvDim
                                                + kvHeadOffset);
                        float score = q.dot(qOff, cur ? bK : cK, off, headSize) * scale;
                        if (score > m) {
                            float rescale = (float) Math.exp(m - score);
                            normalize(pO, oOff, headSize, rescale);
                            l *= rescale;
                            m = score;
                        }
                        float prob = (float) Math.exp(score - m);
                        accumulate(pO, oOff, cur ? bV : cV, off, headSize, prob);
                        l += prob;
                    }
                    pM[task] = m;
                    pL[task] = l;
                });

        // Merge the partitions' partials per head (two-pass online softmax), folding an optional
        // sink.
        Parallel.parallelFor(
                0,
                nHeads,
                h -> {
                    int qOff = h * headSize;
                    float gM = Float.NEGATIVE_INFINITY;
                    for (int p = 0; p < nParts; p++) gM = Math.max(gM, pM[p * nHeads + h]);
                    out.fillInPlace(qOff, headSize, 0f);
                    double gL = 0.0;
                    for (int p = 0; p < nParts; p++) {
                        int task = p * nHeads + h;
                        float pm = pM[task];
                        if (pm == Float.NEGATIVE_INFINITY) continue;
                        float w = (float) Math.exp(pm - gM);
                        accumulate(out, qOff, pO, task * headSize, headSize, w);
                        gL += pL[task] * w;
                    }
                    if (sinks == null) {
                        normalize(out, qOff, headSize, gL == 0.0 ? 0f : (float) (1.0 / gL));
                    } else {
                        float sink = sinks.getFloat(h);
                        float newM = Math.max(gM, sink);
                        float factor =
                                gM == Float.NEGATIVE_INFINITY ? 0f : (float) Math.exp(gM - newM);
                        double denom = gL * factor + Math.exp(sink - newM);
                        normalize(
                                out, qOff, headSize, denom == 0.0 ? 0f : (float) (factor / denom));
                    }
                });
    }

    private FlashAttention() {}
}
