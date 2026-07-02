// All jinfer.* properties read at run time (works with -D on the JVM and on a native binary).
package com.qxotic.jinfer;

import com.sun.management.HotSpotDiagnosticMXBean;

import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

/**
 * Every tunable jinfer.* system property read AT RUN TIME, in one place. This class is
 * initialized at run time even in a native image (--initialize-at-run-time in the Makefile),
 * so -Djinfer.x=y behaves identically on the JVM and on a compiled binary.
 *
 * <p>Flags NOT here are deliberately baked into the binary at image build time — they shape
 * compiled code or run at build time by design:
 * <ul>
 *   <li>jinfer.VectorBitSize — vector species selection (FloatTensor)</li>
 *   <li>jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK — GLOBAL_SEGMENT routing (FloatTensor)</li>
 *   <li>jinfer.Q8_0GemmTile — register-tile shape (Java tiled gemm)</li>
 *   <li>jinfer.staticGemm / llama.nativeGemmLib / llama.nativeGemv — backend binding (removed — see MatMul)</li>
 *   <li>jinfer.PreloadGGUF — model baked into the image heap (AOT)</li>
 * </ul>
 */
final class RuntimeFlags {

    // generation / engine
    static final int MAX_PROMPT_SEQUENCE_LENGTH = Integer.getInteger("jinfer.maxPromptSequenceLength", 1024);
    // Default scratch/batch width when a caller creates a state without picking one (Model.newState(ctx)):
    // a prefill of up to this many tokens ingests in a single batch; longer prompts re-chunk by the caller.
    static final int BATCH_CAPACITY = Integer.getInteger("jinfer.batchCapacity", 512);
    static final boolean ROLLING_ATTENTION = !"false".equals(System.getProperty("jinfer.rollingAttention"));
    // Block-tiled flash prefill (FlashAttention.qkTile/pvTile): each f16 KV vector is decoded once and
    // reused across QT query rows, ~1.7x faster than the per-position rolling path at multi-thousand-
    // token depth (and it flattens the prefill-vs-context curve). On by default; disable with
    // -Djinfer.flashAttention=false. NOTE: needs the f16-decode inline hints (see Makefile INLINE_HINTS)
    // — without them the qkTile/pvTile vectors box and flash falls behind rolling.
    static final boolean FLASH_ATTENTION = !"false".equals(System.getProperty("jinfer.flashAttention"));
    static final int FULL_ATTENTION_WINDOW = Integer.getInteger("jinfer.fullAttentionWindow", 0);
    // True when a vector-intrinsifying compiler is in play: the Graal JIT (UseJVMCICompiler) or the
    // SubstrateVM AOT backend (native image). The headSize-64 decode-attention kernel
    // (Llama.rollingAttnCached64) decodes 8 f16 KV vectors per key in ONE method; Graal keeps that in
    // zmm registers, but HotSpot C2's vector box-elimination budget is smaller and it BOXES the f16
    // decode (~78x more boxed-vector ops measured), collapsing decode -50% at long context. When this
    // is false (C2 / plain OpenJDK) the dispatch uses the C2-friendly generic decode path instead
    // (per-key keyCache.dot + hand-inlined V decode), which never boxes. Override with
    // -Djinfer.graalVectorJit=true|false. See Llama.decodeAttentionBlockParallel.
    static final boolean GRAAL_VECTOR_JIT = detectGraalVectorJit();
    static final int DECODE_BLOCK_SIZE = Integer.getInteger("jinfer.decodeBlockSize", 512);
    static final int DECODE_BLOCK_PARALLEL_MIN_RANGE = Integer.getInteger("jinfer.decodeBlockParallelMinRange", 1024);
    static final boolean LAST_ROW_LOGITS = !"false".equals(System.getProperty("jinfer.lastRowLogits"));

    // decode runs at physical-core width on a spin-barrier pool (Parallel.onDecodePool / SpinPool): decode is
    // memory-bandwidth bound, so one thread per PHYSICAL core saturates DRAM while a 2nd SMT sibling only
    // contends for the core's load/store ports. -Djinfer.decodeSpin=false forces the plain ForkJoin path.
    static final int DECODE_THREADS = Integer.getInteger("jinfer.decodeThreads", physicalCoreCount());
    static final boolean DECODE_SPIN = !"false".equals(System.getProperty("jinfer.decodeSpin"));

    // grammar-constrained decoding (GBNF / response_format json_object)
    static final boolean GRAMMAR = !"false".equals(System.getProperty("jinfer.grammar"));

    // prompt cache
    static final boolean PROMPT_CACHE = !"false".equals(System.getProperty("jinfer.promptCache"));
    static final long PROMPT_CACHE_BUDGET_BYTES = Long.getLong("jinfer.promptCacheMB", 2048L) * (1L << 20);
    static final String PROMPT_CACHE_FILE = System.getProperty("jinfer.promptCacheFile"); // mmap backing
    static final int PROMPT_CACHE_BLOCK_TOKENS = Integer.getInteger("jinfer.promptCacheBlockTokens", 512);
    // bx retention: conv-input rows kept so lookups can resume INSIDE cached spans (0 = off)
    static final int PROMPT_CACHE_STRIDE = Integer.getInteger("jinfer.promptCacheStride", 64);
    static final int PROMPT_CACHE_DENSE_TAIL = Integer.getInteger("jinfer.promptCacheDenseTail", 256);
    static final String PROMPT_CACHE_WARM = System.getProperty("jinfer.promptCacheWarm");

    // server
    static final int SERVER_THREADS = Integer.getInteger("jinfer.serverThreads", 16);
    static final int SERVER_QUEUE = Integer.getInteger("jinfer.serverQueue", 4);
    static final long SERVER_MAX_BODY_BYTES = Math.min(Long.getLong("jinfer.serverMaxBodyMB", 32), 1024) << 20;
    static final long SERVER_WRITE_STALL_NANOS = TimeUnit.SECONDS.toNanos(Long.getLong("jinfer.serverWriteTimeout", 30));
    static final int SERVER_MAX_TOKENS = Integer.getInteger("jinfer.serverMaxTokens", 4096); // 0 = no completion-token ceiling
    static final long SERVER_REQUEST_TIMEOUT_NANOS = TimeUnit.SECONDS.toNanos(Long.getLong("jinfer.serverRequestTimeout", 300)); // 0 = no generation deadline

    /** Whether a vector-intrinsifying compiler runs the wide decode kernel without boxing: a native
     *  image (Graal AOT backend) or a JVM with the Graal JIT active (UseJVMCICompiler=true — read as
     *  the live flag, so it correctly reports {@code false} for a GraalVM build started with
     *  -XX:-UseJVMCICompiler, i.e. forced HotSpot C2). Plain OpenJDK/C2 -> false. */
    private static boolean detectGraalVectorJit() {
        String override = System.getProperty("jinfer.graalVectorJit");
        if (override != null) return Boolean.parseBoolean(override);
        if (System.getProperty("org.graalvm.nativeimage.imagecode") != null) return true; // SubstrateVM AOT
        try {
            var bean = ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean.class);
            return bean != null && "true".equals(bean.getVMOption("UseJVMCICompiler").getValue());
        } catch (RuntimeException noSuchOptionOrBean) {
            return false; // conservative: assume C2, use the non-boxing decode path
        }
    }

    /** Best-effort physical-core count for sizing the bandwidth-bound decode pool. Linux reports SMT state
     *  via sysfs (SMT on => 2 hardware threads per core => logical/2; off => logical). macOS/Windows have no
     *  such file, so we assume 2-way SMT on x86 and none on ARM (Apple Silicon and most ARM cores have no
     *  SMT, so logical == physical there). Override with -Djinfer.decodeThreads; read at run time so a native
     *  binary detects its host. */
    private static int physicalCoreCount() {
        int logical = Runtime.getRuntime().availableProcessors();
        try {
            boolean smtOn = !"0".equals(Files.readString(
                    Path.of("/sys/devices/system/cpu/smt/active")).trim());
            return smtOn ? Math.max(1, logical / 2) : logical;
        } catch (Exception notLinux) {
            String arch = System.getProperty("os.arch", "");
            boolean noSmt = arch.contains("aarch64") || arch.contains("arm");
            return noSmt ? logical : Math.max(1, logical / 2);
        }
    }

    private RuntimeFlags() {
    }
}
