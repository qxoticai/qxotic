package com.qxotic.jam;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED;

import com.oracle.svm.shared.AlwaysInline;
import java.lang.foreign.MemorySegment;
import java.util.Locale;
import java.util.function.IntConsumer;
import java.util.regex.Pattern;
import java.util.stream.IntStream;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * Shared Vector API support for jam-vector's relocated kernels: the float species, fp16/int8
 * decode, the register-tiling knobs ({@code -Djam.vector.*}), a prefill {@code parallelFor}, and
 * the JVM/CPU-aware tile-shape selection ({@link #TILE_CODE}) that the Q8_0 multi-tile kernel
 * dispatches on.
 */
final class VectorSupport {

    private VectorSupport() {}

    /**
     * A {@code jam.vector.*} knob, read uniformly: {@code -Djam.vector.NAME} wins, else env {@code
     * JAM_VECTOR_NAME} (dots→underscores, upper-cased), else {@code def}. So every tunable below
     * works as a system property OR an environment variable.
     */
    static String jamProp(String name, String def) {
        String v = System.getProperty(name);
        if (v == null) v = System.getenv(name.toUpperCase(Locale.ROOT).replace('.', '_'));
        return v != null ? v : def;
    }

    /** Integer {@link #jamProp}. */
    static int jamPropInt(String name, int def) {
        String v = jamProp(name, null);
        return v != null ? Integer.parseInt(v.trim()) : def;
    }

    /**
     * Float vector species; the JVM's preferred width unless pinned with {@code -Djam.vector.width}
     * / {@code JAM_VECTOR_WIDTH}.
     */
    static final VectorSpecies<Float> F_SPECIES =
            VectorShape.forBitSize(
                            jamPropInt(
                                    "jam.vector.width",
                                    FloatVector.SPECIES_PREFERRED.vectorBitSize()))
                    .withLanes(float.class);

    static final boolean IS_512 = F_SPECIES.vectorBitSize() == 512;

    // ---- width-generic k-quant/FP4 decode: a 16-byte SPECIES_128 chunk fans out into 16/F_LEN F32
    // stores
    //      (1 at 512-bit, 2 at 256, 4 at 128). castShape part p maps input byte-lanes [p*F_LEN,
    // (p+1)*F_LEN). ----
    static final int F_LEN = F_SPECIES.length();
    static final int DECODE_PARTS =
            ByteVector.SPECIES_128.length() / F_LEN; // constant-folded: 1 | 2 | 4

    /**
     * Affine decode of a 16-quant byte chunk into {@code dst[off..]}: value = q*scale + neg, any
     * vector width.
     */
    static void storeAffine(
            ByteVector q, FloatVector scale, FloatVector neg, float[] dst, int off) {
        for (int p = 0; p < DECODE_PARTS; p++)
            ((FloatVector) q.castShape(F_SPECIES, p))
                    .fma(scale, neg)
                    .intoArray(dst, off + p * F_LEN);
    }

    /**
     * Scaled decode of a 16-quant byte chunk into {@code dst[off..]}: value = q*scale, any vector
     * width.
     */
    static void storeScaled(ByteVector q, FloatVector scale, float[] dst, int off) {
        for (int p = 0; p < DECODE_PARTS; p++)
            ((FloatVector) q.castShape(F_SPECIES, p)).mul(scale).intoArray(dst, off + p * F_LEN);
    }

    /**
     * True under native image (build- or run-time init both see the property; constant-folds in the
     * image). The wide 512-bit tiles (4x4/2x8/8x2) crash Graal's AOT backend: their register
     * pressure allocates a VEX-only op to xmm16+ and {@code AMD64Assembler$VexOp.checkVex} fails
     * the build. The narrow 3x2 tile compiles and runs correctly, so it is the native-image shape.
     */
    static final boolean IN_NATIVE_IMAGE =
            System.getProperty("org.graalvm.nativeimage.imagecode") != null;

    /**
     * Wide 512-bit tiles (16+ live vector accumulators) are compilable by the compiler in play:
     * always on the JVM; under native image only with an explicit build-time opt-in ({@code
     * -Djam.vector.wideTiles=true} on the image builder), because stock Graal's AOT backend fails
     * their VEX encoding at xmm16+ - a Graal with 32-ZMM AVX-512 support (GR-13757) compiles them
     * fine. Build-time initialized, so the flag constant-folds into the image.
     */
    static final boolean WIDE_TILES_COMPILABLE =
            !IN_NATIVE_IMAGE || Boolean.getBoolean("jam.vector.wideTiles");

    /** Register-tiling knobs (same defaults as jinfer's GEMM_* tunables). */
    static final int SEQ_TILE = jamPropInt("jam.vector.seqTile", 32);

    static final int SEQ_TILE_QK = jamPropInt("jam.vector.seqTileQk", 8); // k-quants tile narrower
    static final int ROW_TILE = jamPropInt("jam.vector.rowTile", 128);
    static final int THREADS =
            jamPropInt("jam.vector.threads", Runtime.getRuntime().availableProcessors() * 4);

    // ---- Register-tile selection, resolved once from -Djam.vector.tile + CPU width + JIT
    // (relocated from
    //      jinfer's VectorJAM). The Q8_0 kernel reads TILE_CODE; wide tiles need spill-free
    // zmm16-zmm31. ----
    static final String TILE =
            jamProp(
                    "jam.vector.tile",
                    System.getProperty(
                            "jinfer.Q8_0GemmTile",
                            "auto")); // legacy -D name still honored as the default

    /**
     * Constant-foldable codes:
     * 0=3x2,1=3x4,2=4x4,3=2x8,4=8x2,5=1x1,6..9=avx256,10/11=neon,12=scalar.
     */
    static final int TILE_CODE =
            switch (TILE) {
                case "auto" -> autoTileCode();
                case "3x2" -> 0;
                case "4x4" -> 2;
                case "2x8" -> 3;
                case "8x2" -> 4;
                case "1x1" -> 5;
                case "avx256", "avx256-2x4" -> 6;
                case "avx256-2x3" -> 7;
                case "avx256-3x4" -> 8;
                case "avx256-4x3" -> 9;
                case "neon", "neon-4x4" -> 10;
                case "neon-2x4" -> 11;
                case "scalar", "java" -> 12;
                default -> 1; // 3x4
            };

    /**
     * Optimistic "the JIT keeps wide vector tiles (>=16 live vectors) spill-free" — gates only the
     * {@link BandGemm} 4x4 default now (the Q8_0 register tile defaults to the spill-free 3x2
     * regardless; see {@link #autoTileCode}). NOTE: stock C2 and Oracle GraalVM actually allocate
     * only zmm0-15, so this over-reports for BandGemm too; revisit if BandGemm 4x4-vs-3x3 is
     * measured on those JITs.
     */
    static final boolean WIDE_TILE = jitHandlesWideTile();

    private static int autoTileCode() {
        String arch = System.getProperty("os.arch", "").toLowerCase();
        if (arch.contains("aarch64") || arch.startsWith("arm")) return 10; // ARM NEON 4x4
        int width = F_SPECIES.vectorBitSize();
        if (width >= 512) {
            if (IN_NATIVE_IMAGE) {
                // stock AOT: 3x2 (fastest compilable shape; 3x4 spills, 207 vs 289 pp; wide tiles
                // fail
                // VEX encoding at xmm16+). A 32-ZMM Graal (jam.vector.wideTiles=true) takes 4x4.
                return WIDE_TILES_COMPILABLE ? 2 : 0;
            }
            // 3x2 fits entirely in zmm0-15, so it is spill-free on EVERY current JIT. 4x4 needs 32
            // ZMM to
            // avoid spilling, which today only a patched Graal provides: stock HotSpot C2 and
            // Oracle GraalVM
            // allocate only zmm0-15, so 4x4 spills (disassembly: 519 zmm<->stack moves vs 23 for
            // 3x2; 632 vs
            // 706 t/s on Oracle EE Q8_0 prefill). Force 4x4 with -Djam.vector.tile=4x4 on a 32-ZMM
            // build (or
            // on C2, where its ILP hides the spills).
            return 0; // 3x2
        }
        if (width >= 256) return 6; // AVX2 2x4
        return 12; // scalar
    }

    // 4x4 needs 16 accumulators in registers: Graal spill-free only from jvmci-25.1; HotSpot C2
    // spills but
    // they're bandwidth-hidden so 4x4 still wins; unknown VM -> safe 3x2.
    private static boolean jitHandlesWideTile() {
        if (IN_NATIVE_IMAGE)
            return WIDE_TILES_COMPILABLE; // stock Graal AOT fails wide-tile VEX encoding; 32-ZMM
        // builders opt in
        String version = System.getProperty("java.vm.version", "");
        if (version.contains("jvmci")) {
            var v = Pattern.compile("jvmci-(\\d+)\\.(\\d+)").matcher(version);
            if (v.find()) {
                int major = Integer.parseInt(v.group(1)), minor = Integer.parseInt(v.group(2));
                return major > 25 || (major == 25 && minor >= 1);
            }
            return false; // "jvmci-bNN" (25.0-era Graal) caps at zmm0-15
        }
        String name = System.getProperty("java.vm.name", "");
        return name.contains("HotSpot") || name.contains("OpenJDK");
    }

    /**
     * Absolute-address scalar store, for kernels that address the output by raw byte address (the
     * Q8_0 tiles). Mirrors jinfer's {@code FloatTensor.putFloat}: one exact segment type so the
     * access folds.
     */
    static final MemorySegment GLOBAL = makeGlobalSegment();

    private static MemorySegment makeGlobalSegment() {
        try {
            return MemorySegment.NULL.reinterpret(Long.MAX_VALUE);
        } catch (Throwable t) {
            return null;
        }
    }

    static void putFloat(long address, float value) {
        GLOBAL.set(JAVA_FLOAT_UNALIGNED, address, value);
    }

    /**
     * Weight-read routing for the tile kernels: with {@link #GLOBAL}, access {@code
     * (vectorSegment(w), vectorBase(w) + byteOffset)} uses ONE compile-time-constant segment with
     * absolute addresses, exactly as jinfer's FloatTensor. Besides folding the bounds/liveness
     * checks, this is REQUIRED under native image: Graal's Vector API expansion mis-addresses a
     * vector load whose segment base object is a runtime value of merged heap/native types (a null
     * native base is decoded as the compressed-reference heap base, producing a non-canonical
     * address and a GP fault). The constant GLOBAL folds the base to a known null, which compiles
     * to correct absolute addressing.
     */
    static MemorySegment vectorSegment(MemorySegment seg) {
        return GLOBAL != null ? GLOBAL : seg;
    }

    static long vectorBase(MemorySegment seg) {
        return GLOBAL != null ? seg.address() : 0L;
    }

    /** Read the signed int8 at byte offset {@code off} in {@code seg} (Q8_0 quant). */
    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    static byte readByte(MemorySegment seg, long off) {
        return GLOBAL != null
                ? GLOBAL.get(JAVA_BYTE, seg.address() + off)
                : seg.get(JAVA_BYTE, off);
    }

    /**
     * Read the raw IEEE half (16-bit) at byte offset {@code off} in {@code seg} (the block scale).
     */
    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    static short readShort(MemorySegment seg, long off) {
        return GLOBAL != null
                ? GLOBAL.get(JAVA_SHORT_UNALIGNED, seg.address() + off)
                : seg.get(JAVA_SHORT_UNALIGNED, off);
    }

    /** Read the little-endian int32 at byte offset {@code off} (k-quant packed scales). */
    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    static int readInt(MemorySegment seg, long off) {
        return GLOBAL != null
                ? GLOBAL.get(JAVA_INT_UNALIGNED, seg.address() + off)
                : seg.get(JAVA_INT_UNALIGNED, off);
    }

    /** Read the little-endian int64 at byte offset {@code off} (k-quant packed scales). */
    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    static long readLong(MemorySegment seg, long off) {
        return GLOBAL != null
                ? GLOBAL.get(JAVA_LONG_UNALIGNED, seg.address() + off)
                : seg.get(JAVA_LONG_UNALIGNED, off);
    }

    /** Prefill fan-out: vanilla parallel IntStream (measured-best for the compute-bound gemm). */
    static void parallelFor(int from, int to, IntConsumer body) {
        IntStream.range(from, to).parallel().forEach(body);
    }

    /** A contiguous {@code [lo, hi)} slice of work handed to one parallel worker. */
    @FunctionalInterface
    interface ChunkConsumer {
        void accept(int lo, int hi);
    }

    /**
     * Split {@code [0, count)} into {@code min(count, THREADS)} contiguous slices and run each as
     * one parallel task. Unlike {@link #parallelFor}, the body owns a whole slice — so a band
     * kernel can {@link Scratch#acquire} one dequant buffer per worker (not per group) and reuse it
     * across the slice's rows. With {@link Scratch}'s context-owned pool this means no per-{@code
     * mm} allocation, while the buffers stay reachable only through the context (freed when it is
     * GC'd) rather than the old commonPool-rooted, JVM-lifetime ThreadLocal.
     */
    static void parallelChunks(int count, ChunkConsumer body) {
        if (count <= 0) return;
        int chunks = Math.min(count, Math.max(1, THREADS));
        IntStream.range(0, chunks)
                .parallel()
                .forEach(
                        c -> {
                            int lo = (int) ((long) count * c / chunks);
                            int hi = (int) ((long) count * (c + 1) / chunks);
                            if (lo < hi) body.accept(lo, hi);
                        });
    }

    /**
     * Decode the IEEE half at byte offset {@code off} in {@code seg} to float (JDK-exact, as
     * jinfer).
     */
    @AlwaysInline(
            "hot scalar accessor: must inline into kernels (profiled out-of-line on CE native)")
    static float readFloat16(MemorySegment seg, long off) {
        return Float.float16ToFloat(readShort(seg, off));
    }
}
