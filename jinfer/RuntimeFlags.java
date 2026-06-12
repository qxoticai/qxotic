// All llama.* properties read at run time (works with -D on the JVM and on a native binary).
package com.llama4j;

/**
 * Every tunable llama.* system property read AT RUN TIME, in one place. This class is
 * initialized at run time even in a native image (--initialize-at-run-time in the Makefile),
 * so -Dllama.x=y behaves identically on the JVM and on a compiled binary.
 *
 * <p>Flags NOT here are deliberately baked into the binary at image build time — they shape
 * compiled code or run at build time by design:
 * <ul>
 *   <li>llama.VectorBitSize — vector species selection (FloatTensor)</li>
 *   <li>jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK — GLOBAL_SEGMENT routing (FloatTensor)</li>
 *   <li>llama.Q8_0GemmTile — register-tile shape (JavaKernels)</li>
 *   <li>llama.staticGemm / llama.nativeGemmLib / llama.nativeGemv — backend binding (NativeKernels, Kernels.INSTANCE)</li>
 *   <li>llama.PreloadGGUF — model baked into the image heap (AOT)</li>
 * </ul>
 */
final class RuntimeFlags {

    // generation / engine
    static final int MAX_PROMPT_SEQUENCE_LENGTH = Integer.getInteger("llama.maxPromptSequenceLength", 1024);
    static final boolean ROLLING_ATTENTION = !"false".equals(System.getProperty("llama.rollingAttention"));
    static final boolean FLASH_ATTENTION = Boolean.getBoolean("llama.flashAttention");
    static final int FULL_ATTENTION_WINDOW = Integer.getInteger("llama.fullAttentionWindow", 0);
    static final int DECODE_BLOCK_SIZE = Integer.getInteger("llama.decodeBlockSize", 512);
    static final int DECODE_BLOCK_PARALLEL_MIN_RANGE = Integer.getInteger("llama.decodeBlockParallelMinRange", 1024);
    static final boolean LAST_ROW_LOGITS = !"false".equals(System.getProperty("llama.lastRowLogits"));

    // gemm tiling: the values feed loop bounds, not compiled code shapes — safe to tune at run time
    static final int GEMM_SEQ_TILE = Integer.getInteger("llama.Q8_0GemmSeqTile", 32);
    static final int GEMM_ROW_TILE = Integer.getInteger("llama.Q8_0GemmRowTile", 128);
    static final int GEMM_THREADS = Integer.getInteger("llama.Q8_0GemmThreads",
            Integer.getInteger("llama.Q8_0GemmWorkers", Runtime.getRuntime().availableProcessors() * 4));

    // prompt cache
    static final boolean PROMPT_CACHE = !"false".equals(System.getProperty("llama.promptCache"));
    static final int PROMPT_CACHE_PAGE = Integer.getInteger("llama.promptCachePage", 512); // 256 costs ~20% cold prefill (chunked gemms); 512 measured at ~7%
    static final long PROMPT_CACHE_BUDGET_BYTES = Long.getLong("llama.promptCacheMB", 2048L) * (1L << 20);

    // server
    static final int SERVER_THREADS = Integer.getInteger("llama.serverThreads", 16);
    static final int SERVER_QUEUE = Integer.getInteger("llama.serverQueue", 4);
    static final long SERVER_MAX_BODY_BYTES = Math.min(Long.getLong("llama.serverMaxBodyMB", 32), 1024) << 20;
    static final long SERVER_WRITE_STALL_NANOS = java.util.concurrent.TimeUnit.SECONDS.toNanos(Long.getLong("llama.serverWriteTimeout", 30));

    // thread affinity (needs OpenHFT affinity on the classpath)
    static final boolean AFFINITY = Boolean.getBoolean("llama.Affinity");
    static final boolean AFFINITY_VERBOSE = Boolean.getBoolean("llama.AffinityVerbose");
    static final String AFFINITY_CPUS = System.getProperty("llama.AffinityCpus");

    private RuntimeFlags() {
    }
}
