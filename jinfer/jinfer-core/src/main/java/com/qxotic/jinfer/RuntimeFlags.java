// All jinfer.* properties read at run time (works with -D on the JVM and on a native binary).
package com.qxotic.jinfer;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Every tunable jinfer.* system property read AT RUN TIME, in one place. This class is initialized
 * at run time even in a native image (--initialize-at-run-time in the Makefile), so -Djinfer.x=y
 * behaves identically on the JVM and on a compiled binary.
 *
 * <p>Flags NOT here are deliberately baked into the binary at image build time — they shape
 * compiled code or run at build time by design:
 *
 * <ul>
 *   <li>jinfer.VectorBitSize — vector species selection (FloatTensor)
 *   <li>jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK — GLOBAL_SEGMENT routing (FloatTensor)
 *   <li>jinfer.Q8_0GemmTile — register-tile shape (Java tiled gemm)
 *   <li>jinfer.staticGemm / llama.nativeGemmLib / llama.nativeGemv — backend binding (removed — see
 *       MatMul)
 *   <li>jinfer.PreloadGGUF — model baked into the image heap (AOT)
 *   <li>jinfer.convTile — Convolutions' register tile, which must fold to a constant
 *   <li>jinfer.trace — Trace.ENABLED guards per-layer loops, so it must fold too
 *   <li>jinfer.convProfile — the shape census, which exists to be run on the JVM
 *   <li>jam.vector.wideTiles — jam's 512-bit tiles (VectorSupport)
 * </ul>
 *
 * <p>All of those are passed to the IMAGE BUILD (see the jinfer.* pom properties beside
 * jinfer.PreloadGGUF), never to the binary, where they are silently ignored. A flag that must be
 * settable on the binary belongs here, or its class belongs in the builder's
 * {@code --initialize-at-run-time} list — the two ways to keep a run-time flag actually run-time.
 */
public final class RuntimeFlags {

    // generation / engine
    public static final int MAX_PROMPT_SEQUENCE_LENGTH =
            Integer.getInteger("jinfer.maxPromptSequenceLength", 1024);
    // Default scratch/batch width when a caller creates a state without picking one
    // (Model.newState(ctx)):
    // a prefill of up to this many tokens ingests in a single batch; longer prompts re-chunk by the
    // caller.
    public static final int BATCH_CAPACITY = Integer.getInteger("jinfer.batchCapacity", 512);
    static final int DECODE_BLOCK_SIZE = Integer.getInteger("jinfer.decodeBlockSize", 512);
    // decode runs at physical-core width on a spin-barrier pool (Parallel.onDecodePool / SpinPool):
    // decode is
    // memory-bandwidth bound, so one thread per PHYSICAL core saturates DRAM while a 2nd SMT
    // sibling only
    // contends for the core's load/store ports. -Djinfer.decodeSpin=false forces the plain ForkJoin
    // path.
    static final int DECODE_THREADS =
            Integer.getInteger("jinfer.decodeThreads", physicalCoreCount());
    static final boolean DECODE_SPIN = !"false".equals(System.getProperty("jinfer.decodeSpin"));

    // grammar-constrained decoding (GBNF / response_format json_object)
    public static final boolean GRAMMAR = !"false".equals(System.getProperty("jinfer.grammar"));

    // prompt cache
    public static final boolean PROMPT_CACHE =
            !"false".equals(System.getProperty("jinfer.promptCache"));

    /** Live conversations kept resident for append-only reuse (tier 1); each holds a full state. */
    public static final int SESSIONS = Integer.getInteger("jinfer.sessions", 4);

    public static final long PROMPT_CACHE_BUDGET_BYTES =
            Long.getLong("jinfer.promptCacheMB", 2048L) * (1L << 20);

    /**
     * Best-effort physical-core count for sizing the bandwidth-bound decode pool. Linux reports SMT
     * state via sysfs (SMT on => 2 hardware threads per core => logical/2; off => logical).
     * macOS/Windows have no such file, so we assume 2-way SMT on x86 and none on ARM (Apple Silicon
     * and most ARM cores have no SMT, so logical == physical there). Override with
     * -Djinfer.decodeThreads; read at run time so a native binary detects its host.
     */
    private static int physicalCoreCount() {
        int logical = Runtime.getRuntime().availableProcessors();
        try {
            boolean smtOn =
                    !"0"
                            .equals(
                                    Files.readString(Path.of("/sys/devices/system/cpu/smt/active"))
                                            .trim());
            return smtOn ? Math.max(1, logical / 2) : logical;
        } catch (Exception notLinux) {
            String arch = System.getProperty("os.arch", "");
            boolean noSmt = arch.contains("aarch64") || arch.contains("arm");
            return noSmt ? logical : Math.max(1, logical / 2);
        }
    }

    private RuntimeFlags() {}
}
