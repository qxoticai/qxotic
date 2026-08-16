package com.qxotic.jinfer.x.bench;

import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.ContextState;
import com.qxotic.jinfer.x.boundary.Multimodal;
import com.qxotic.jinfer.x.boundary.media.ImageCodec;
import com.qxotic.jinfer.x.cache.PromptCache;
import com.qxotic.jinfer.x.chat.ChatEngine;
import com.qxotic.jinfer.x.chat.Content;
import com.qxotic.jinfer.x.chat.LoadedModel;
import com.qxotic.jinfer.x.chat.Message;
import com.qxotic.jinfer.x.chat.Models;
import com.qxotic.jinfer.x.chat.Role;
import com.qxotic.jinfer.x.llm.Sampler;
import com.qxotic.jinfer.x.llm.Sampling;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;

/**
 * A llama-bench-parity harness for the x tree, driving every port through the generic loader
 * ({@link Models#load} - any architecture on the classpath runs, no per-model code). The
 * llama-bench approach ({@code tools/llama-bench/llama-bench.cpp}), point by point:
 *
 * <ul>
 *   <li>pp512 / tg128 are SEPARATE tests, each with its own context sized exactly to the work
 *       ({@code n_ctx = n_prompt} for pp, {@code n_ctx = n_gen} for tg - llama-bench derives {@code
 *       n_prompt + n_gen + n_depth} per instance, and each instance sets the other count to 0);
 *       {@code --ctx} overrides both
 *   <li>ONE state per test, {@code reset()} before every warmup pass and every timed rep -
 *       llama-bench's {@code llama_memory_clear} per rep on a reused context. No allocation or
 *       page-fault cost inside the timed region
 *   <li>pp: the full prompt ingested in chunks of 512 - llama-bench's compute-graph width ({@code
 *       -ub 512}; its {@code -b 2048} only caps tokens per decode CALL, ggml still computes 512-row
 *       ubatches). One vocab projection for the last token, charged inside the timed region, as
 *       llama_decode does
 *   <li>tg: from a CLEARED (empty) state, {@code n} single-token decodes at positions 0..n-1, all
 *       timed - the first token goes in INSIDE the timed loop, like llama-bench's {@code test_gen}.
 *       Logits projected every step, never argmaxed: llama-bench feeds back {@code rand() %
 *       n_vocab}, this feeds a cheap LCG successor (throughput is content-independent for dense
 *       models; MoE routing sees pseudo-random tokens either way)
 *   <li>threads: physical cores by default ({@code common_cpu_get_num_math}), the SAME count for pp
 *       and tg ({@code llama_set_n_threads(ctx, t, t)}), plus the JAM thread-count guard
 *   <li>reporting: mean ± stddev of per-rep tokens/second - llama-bench's avg_ts / stdev_ts
 * </ul>
 *
 * Deliberate deltas, all JVM-necessitated or immaterial: warmup is ADAPTIVE (full-test passes until
 * throughput settles within 3% over a window of 3, min {@code -w}, max 30) instead of llama-bench's
 * single warmup run - native code needs one pass, the JIT needs several; {@code --no-warmup} skips
 * it. Tokens are synthetic in-range ids, not real BOS/rand. Threadpool priority/polling ({@code
 * poll=50}) has no Java analog. Loading is direct ({@code Models.load}) and never timed.
 *
 * <p>Beyond the parity tests, each model also gets a CAPABILITIES pass through {@link ChatEngine}
 * (the layer where the machinery lives): time to first token, prompt-cache hit latency, MTP draft
 * acceptance and net decode speedup (skipped when the model carries no draft head), and - with
 * {@code --media <file>} and a vision port - projected-media cold versus warm latency. State
 * allocation time and peak RSS (VmHWM) are reported per model.
 *
 * <pre>jinfer-xbench -m model.gguf [-m ...] [-p 512] [-n 128] [-r 5] [-w 2] [--ctx N]
 *                  [--media image.png]</pre>
 */
public final class XJinferBench {

    /**
     * Compute-graph width for prefill chunks - llama-bench's {@code -ub 512}. Pinned here rather
     * than inherited from {@code RuntimeFlags.BATCH_CAPACITY} so an ambient {@code
     * -Djinfer.batchCapacity} cannot silently change what is measured.
     */
    private static final int UBATCH = 512;

    public static void main(String[] args) throws Exception {
        List<String> models = new ArrayList<>();
        Map<String, Path> companions = new java.util.LinkedHashMap<>();
        int p = 512, n = 128, reps = 5, warmup = 2, ctx = 0, threads = 0;
        boolean noWarmup = false;
        Path media = null;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-m", "--model" -> models.add(args[++i]);
                case "--with" -> {
                    // xcli's convention: --with media=<mmproj.gguf> attaches a companion
                    String[] kv = args[++i].split("=", 2);
                    if (kv.length != 2) {
                        System.err.println("--with expects capability=path, got: " + args[i]);
                        System.exit(2);
                    }
                    companions.put(kv[0], Path.of(kv[1]));
                }
                case "-p", "--n-prompt" -> p = Integer.parseInt(args[++i]);
                case "-n", "--n-gen" -> n = Integer.parseInt(args[++i]);
                case "-r", "--repetitions" -> reps = Integer.parseInt(args[++i]);
                case "-w", "--warmup" -> warmup = Integer.parseInt(args[++i]);
                case "--no-warmup" -> noWarmup = true;
                case "--ctx" -> ctx = Integer.parseInt(args[++i]);
                case "-t", "--threads" -> threads = Integer.parseInt(args[++i]);
                case "--media" -> media = Path.of(args[++i]);
                case "-h", "--help" -> {
                    usage(System.out);
                    return;
                }
                default -> {
                    System.err.println("unknown option: " + args[i]);
                    usage(System.err);
                    System.exit(2);
                }
            }
        }
        if (models.isEmpty()) {
            usage(System.err);
            System.exit(2);
        }

        if (threads <= 0) threads = physicalCores();
        System.setProperty("jinfer.decodeThreads", Integer.toString(threads));
        System.setProperty(
                "java.util.concurrent.ForkJoinPool.common.parallelism",
                Integer.toString(Math.max(1, threads - 1))); // +1 for the submitting thread
        int prefillThreads = ForkJoinPool.commonPool().getParallelism() + 1;
        // the x tree's OWN constant - the property is shared with legacy today, but this report
        // must not depend on that coincidence
        int decodeThreads = com.qxotic.jinfer.x.RuntimeFlags.DECODE_THREADS;
        System.err.printf(
                "threads: prefill=%d decode=%d (requested %d)%n",
                prefillThreads, decodeThreads, threads);
        String jamThreads = System.getenv("JAM_NUM_THREADS");
        boolean jamAgrees =
                jamThreads == null
                        ? threads == physicalCores()
                        : jamThreads.trim().equals(Integer.toString(threads));
        if (!jamAgrees) {
            System.err.printf(
                    "WARNING: JAM_NUM_THREADS=%s, so the native gemm backend is NOT at %d threads"
                            + " and pp is not comparable to llama-bench -t %d.%n         Re-run as:"
                            + " JAM_NUM_THREADS=%d jinfer-xbench ...%n",
                    jamThreads == null ? "<unset, jam auto-picks physical cores>" : jamThreads,
                    threads,
                    threads,
                    threads);
        }

        List<Row> rows = new ArrayList<>();
        List<CapRow> caps = new ArrayList<>();
        for (String path : models) {
            System.err.printf("loading %s ...%n", path);
            BenchModel<?> bench = BenchModel.open(Path.of(path), companions);
            String name = name(path);
            // llama-bench: pp and tg are separate tests on separate contexts sized to the work
            // (n_ctx = n_prompt resp. n_gen; --ctx overrides both).
            if (p > 0)
                rows.add(
                        measure(
                                bench,
                                name,
                                prefillThreads,
                                "pp" + p,
                                p,
                                true,
                                ctx != 0 ? ctx : p,
                                warmup,
                                reps,
                                noWarmup));
            if (n > 0)
                rows.add(
                        measure(
                                bench,
                                name,
                                decodeThreads,
                                "tg" + n,
                                n,
                                false,
                                ctx != 0 ? ctx : n,
                                warmup,
                                reps,
                                noWarmup));
            caps.add(capabilities(bench, name, p > 0 ? p : 512, Math.max(n, 16), media));
        }
        printTable(rows);
        printCaps(caps);
    }

    /**
     * The one model wrapper: whatever {@link Models#load} resolves, driven at the raw boundary
     * (state/batch/logits) for the llama-bench-parity tests and through {@link ChatEngine} for the
     * capability metrics. States are owned-arena, GC/Cleaner-freed - allocation never enters a
     * timed region.
     */
    private static final class BenchModel<S extends ContextState> {
        final LoadedModel<S> loaded;
        private S state;

        private BenchModel(LoadedModel<S> loaded) {
            this.loaded = loaded;
        }

        @SuppressWarnings("unchecked")
        static BenchModel<?> open(Path path, Map<String, Path> companions)
                throws java.io.IOException {
            return new BenchModel<>(
                    (LoadedModel<ContextState>) Models.load(path, Arena.ofAuto(), companions));
        }

        int vocab() {
            return loaded.model().configuration().vocabularySize();
        }

        /** Times the allocation; the caller reports it. */
        long newState(int ctx) {
            long t0 = System.nanoTime();
            state = loaded.model().newState(ctx, UBATCH);
            return System.nanoTime() - t0;
        }

        /** llama_memory_clear: KV/conv state back to empty, same buffers. */
        void reset() {
            state.reset();
        }

        void ingestPrefill(int[] prompt) {
            for (Batch b : Batch.prepare(List.of(Batch.prefill(prompt)), state.batchCapacity())) {
                loaded.model().ingest(state, b);
            }
        }

        void ingestStep(int tok) {
            loaded.model().ingest(state, Batch.step(tok));
        }

        float logits0() {
            return Views.getFloat(
                    Views.castToSegmentBacked(loaded.model().logits(state), "logits"), 0, "logits");
        }
    }

    /**
     * Physical cores, the same quantity llama-bench defaults to. Duplicated from RuntimeFlags
     * rather than imported because it must run BEFORE RuntimeFlags is initialized.
     */
    private static int physicalCores() {
        int logical = Runtime.getRuntime().availableProcessors();
        try {
            boolean smt =
                    !"0"
                            .equals(
                                    Files.readString(Path.of("/sys/devices/system/cpu/smt/active"))
                                            .trim());
            return smt ? Math.max(1, logical / 2) : logical;
        } catch (Exception notLinux) {
            String arch = System.getProperty("os.arch", "");
            return arch.contains("aarch64") || arch.contains("arm")
                    ? logical
                    : Math.max(1, logical / 2);
        }
    }

    /**
     * One llama-bench test: ONE state sized to the work, adaptive warmup (llama-bench's single
     * warmup pass is not enough for the JIT), then timed reps. Every pass starts with {@code
     * reset()} - llama-bench's per-rep {@code llama_memory_clear}.
     */
    private static Row measure(
            BenchModel<?> model,
            String name,
            int threads,
            String test,
            int count,
            boolean prefill,
            int ctx,
            int minWarmup,
            int reps,
            boolean noWarmup) {
        int vocab = model.vocab();
        int[] prompt = prefill ? fillerTokens(vocab, count) : null;
        model.newState(ctx);
        System.err.printf("  %-6s state: ctx=%d batch=%d%n", test, ctx, UBATCH);

        if (noWarmup) {
            System.err.printf("  %-6s warmup skipped (--no-warmup)%n", test);
        } else {
            final double TOL = 0.03;
            final int WINDOW = 3, MAX = Math.max(minWarmup, 30);
            double[] recent = new double[WINDOW];
            int passes = 0;
            while (passes < MAX) {
                double t = runOnce(model, prompt, count, prefill, vocab);
                recent[passes % WINDOW] = t;
                passes++;
                System.err.printf("  %-6s [warmup %2d] %8.2f t/s%n", test, passes, t);
                if (passes >= Math.max(minWarmup, WINDOW)) {
                    double lo = Double.MAX_VALUE, hi = 0;
                    for (double v : recent) {
                        lo = Math.min(lo, v);
                        hi = Math.max(hi, v);
                    }
                    if ((hi - lo) / lo < TOL) break;
                }
            }
            System.err.printf("  %-6s stabilized after %d warmup passes%n", test, passes);
        }

        double[] tps = new double[reps];
        for (int i = 0; i < reps; i++) {
            tps[i] = runOnce(model, prompt, count, prefill, vocab);
            System.err.printf("  %-6s [rep    %2d] %8.2f t/s%n", test, i, tps[i]);
        }
        return new Row(name, threads, test, mean(tps), stddev(tps));
    }

    /**
     * One pass, timed exactly as llama-bench times: the clock spans the prompt processing (pp)
     * resp. the full generation loop from an EMPTY state (tg), nothing else. Both include the vocab
     * projection llama_decode computes for the last token of a batch / each step.
     */
    private static double runOnce(
            BenchModel<?> model, int[] prompt, int count, boolean prefill, int vocab) {
        model.reset(); // llama_memory_clear, outside the timed region
        if (prefill) {
            long t0 = System.nanoTime();
            model.ingestPrefill(prompt);
            sink += model.logits0();
            return count / ((System.nanoTime() - t0) / 1e9);
        }
        int tok = nextToken(1, vocab); // llama-bench's BOS-or-rand first token, fed inside the loop
        long t0 = System.nanoTime();
        for (int g = 0; g < count; g++) {
            model.ingestStep(tok);
            // Project to logits (llama_decode does) but do NOT argmax: llama-bench never reads
            // them, it feeds back `rand() % n_vocab`.
            sink += model.logits0();
            tok = nextToken(tok, vocab);
        }
        return count / ((System.nanoTime() - t0) / 1e9);
    }

    /** Consumed so the logits projection cannot be optimized away; never read. */
    private static volatile float sink;

    /** A cheap in-range successor, standing in for llama-bench's {@code rand() % n_vocab}. */
    private static int nextToken(int previous, int vocab) {
        return (previous * 1103515245 + 12345 & 0x7fffffff) % vocab;
    }

    /** Synthetic in-range token ids - throughput is content-independent. */
    private static int[] fillerTokens(int vocab, int count) {
        int[] ids = new int[count];
        for (int i = 0; i < count; i++) ids[i] = (i * 17 + 1) % vocab;
        return ids;
    }

    // ---- capabilities: the engine-level metrics (cache, MTP, media live in ChatEngine) ----

    private record CapRow(
            String model,
            String stateAllocMs,
            String ttftMs,
            String cacheHit,
            String mtp,
            String media,
            String peakRssMb) {}

    private static final Sampling GREEDY = new Sampling(0f, 1f, 0, 0f, null);

    private static CapRow capabilities(
            BenchModel<?> bench, String name, int promptLen, int gen, Path mediaPath)
            throws Exception {
        // state allocation, timed once on a capabilities-sized state (JIT-warm by now)
        long allocNanos = bench.newState(promptLen + gen + 16);
        String allocMs = String.format("%.1f", allocNanos / 1e6);

        String ttft = "n/a", cache = "n/a", mtp = "no draft head", media = "n/a";
        // two engines over ONE loaded model: the cached one and its no-cache control (cache-hit
        // latency only means something against a full re-prefill of the same follow-up)
        PromptCache.Options noCache =
                PromptCache.Options.DEFAULTS.withRetainedSessions(0).withBlockBudget(0);
        try (ChatEngine engine = new ChatEngine(bench.loaded, name, PromptCache.Options.DEFAULTS);
                ChatEngine control = new ChatEngine(bench.loaded, name, noCache)) {
            Sampler greedy = GREEDY.sampler(bench.vocab());
            int[] prompt = fillerTokens(bench.vocab(), promptLen);

            // TTFT: a cold request's promptTime is prefill + first-token projection
            ChatEngine.Completion cold = complete(engine, prompt, greedy, gen);
            ttft = String.format("%.1f", cold.result().promptTime().toNanos() / 1e6);

            // cache hit: TURN 2 - the cache's session law is "the prompt strictly extends a
            // retained stream", so the workload is prompt + reply + a small delta, never a
            // repeated identical prompt (a single-batch prompt cannot full-hit by construction:
            // resume stops one position short, the last token always recomputes)
            int[] followUp = followUp(prompt, cold.result().tokens(), bench.vocab());
            ChatEngine.Completion hit = complete(engine, followUp, greedy, gen);
            if (hit.restoredTokens() > 0) {
                ChatEngine.Completion full = complete(control, followUp, greedy, gen);
                double hitMs = hit.result().promptTime().toNanos() / 1e6;
                double fullMs = full.result().promptTime().toNanos() / 1e6;
                cache =
                        String.format(
                                "%.1f vs %.1f (%.1fx, %d restored)",
                                hitMs, fullMs, fullMs / hitMs, hit.restoredTokens());
            }

            // MTP: decode with and without the draft head; acceptance from the pass itself
            if (engine.speculationReady()) {
                engine.speculationDepth(0);
                ChatEngine.Completion plain = complete(engine, prompt, greedy, gen);
                engine.speculationDepth(4);
                ChatEngine.Completion drafted = complete(engine, prompt, greedy, gen);
                double plainTps = decodeTps(plain);
                double mtpTps = decodeTps(drafted);
                mtp =
                        drafted.speculated()
                                .map(
                                        s ->
                                                String.format(
                                                        "%.0f%% accepted, %.2fx",
                                                        100.0
                                                                * s.accepted()
                                                                / Math.max(1, s.drafted()),
                                                        mtpTps / plainTps))
                                .orElse("n/a");
            }

            // projected media, cold (encoder projection) vs warm (media-cache replay)
            if (mediaPath != null
                    && bench.loaded.model() instanceof Multimodal mm
                    && mm.projector(com.qxotic.jinfer.x.boundary.Media.Image.class).isPresent()) {
                byte[] bytes = Files.readAllBytes(mediaPath);
                var image = ImageCodec.decode(bytes);
                var request = mediaRequest(image, sha256(bytes), gen);
                long coldMedia = promptNanos(engine, request);
                long warmMedia = promptNanos(engine, request);
                media = String.format("%.1f / %.1f", coldMedia / 1e6, warmMedia / 1e6);
            }
        }
        long peak = peakRssBytes();
        return new CapRow(
                name,
                allocMs,
                ttft,
                cache,
                mtp,
                media,
                peak < 0 ? "n/a" : Long.toString(peak >> 20));
    }

    private static ChatEngine.Completion complete(
            ChatEngine engine, int[] prompt, Sampler sampler, int gen) {
        try (ChatEngine.Prepared prepared =
                ChatEngine.Prepared.raw(prompt, sampler, gen, Duration.ZERO, List.of())) {
            return engine.complete(prepared, ChatEngine.ReplySink.NONE);
        }
    }

    /** Turn-2 prompt: the original prompt, the model's own reply, then a short new delta. */
    private static int[] followUp(int[] prompt, int[] reply, int vocab) {
        int[] out = Arrays.copyOf(prompt, prompt.length + reply.length + 8);
        System.arraycopy(reply, 0, out, prompt.length, reply.length);
        for (int i = prompt.length + reply.length; i < out.length; i++) {
            out[i] = (i * 29 + 7) % vocab;
        }
        return out;
    }

    private static double decodeTps(ChatEngine.Completion c) {
        return c.result().completionTokens() / (c.result().decodeTime().toNanos() / 1e9);
    }

    private static ChatEngine.Request mediaRequest(
            com.qxotic.jinfer.x.boundary.Media.Image image, byte[] contentKey, int gen) {
        return new ChatEngine.Request(
                List.of(
                        new Message(
                                Role.USER,
                                List.of(
                                        new Content.Media(image, contentKey),
                                        new Content.Text("Describe this image.", null)))),
                List.of(),
                false,
                gen,
                null,
                Duration.ZERO,
                GREEDY,
                null,
                null,
                List.of(),
                Map.of());
    }

    /** The request's prompt phase through the engine, in nanos (media projection lives there). */
    private static long promptNanos(ChatEngine engine, ChatEngine.Request request) {
        try (ChatEngine.Prepared prepared = engine.prepare(request)) {
            return engine.complete(prepared, ChatEngine.ReplySink.NONE)
                    .result()
                    .promptTime()
                    .toNanos();
        }
    }

    private static byte[] sha256(byte[] source) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(source);
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /** Peak resident set (Linux VmHWM); -1 where /proc does not exist. */
    private static long peakRssBytes() {
        try {
            for (String line : Files.readAllLines(Path.of("/proc/self/status"))) {
                if (line.startsWith("VmHWM:")) {
                    return Long.parseLong(line.substring(6).replace("kB", "").trim()) << 10;
                }
            }
        } catch (Exception unreadable) {
            // not Linux, or /proc restricted - reported as n/a
        }
        return -1;
    }

    private record Row(String model, int threads, String test, double mean, double stddev) {}

    private static void printTable(List<Row> rows) {
        int w = rows.stream().mapToInt(r -> r.model.length()).max().orElse(5);
        w = Math.max(w, "model".length());
        String fmt = "| %-" + w + "s | %7s | %-6s | %16s |%n";
        System.out.printf(fmt, "model", "threads", "test", "t/s");
        System.out.printf(fmt, "-".repeat(w), "------:", "------", "---------------:");
        for (Row r : rows)
            System.out.printf(
                    fmt,
                    r.model,
                    r.threads,
                    r.test,
                    String.format("%.2f ± %.2f", r.mean, r.stddev));
    }

    private static void printCaps(List<CapRow> rows) {
        String[] headers = {
            "model", "state ms", "ttft ms", "cache hit ms", "mtp", "media cold/warm ms", "peak MB"
        };
        int[] w = new int[headers.length];
        for (int i = 0; i < w.length; i++) w[i] = headers[i].length();
        for (CapRow r : rows) {
            String[] cells = {
                r.model, r.stateAllocMs, r.ttftMs, r.cacheHit, r.mtp, r.media, r.peakRssMb
            };
            for (int i = 0; i < w.length; i++) w[i] = Math.max(w[i], cells[i].length());
        }
        StringBuilder fmt = new StringBuilder();
        for (int width : w) fmt.append("| %-").append(width).append("s ");
        fmt.append("|%n");
        System.out.println();
        System.out.printf(fmt.toString(), (Object[]) headers);
        String[] rule = new String[w.length];
        for (int i = 0; i < w.length; i++) rule[i] = "-".repeat(w[i]);
        System.out.printf(fmt.toString(), (Object[]) rule);
        for (CapRow r : rows)
            System.out.printf(
                    fmt.toString(),
                    r.model,
                    r.stateAllocMs,
                    r.ttftMs,
                    r.cacheHit,
                    r.mtp,
                    r.media,
                    r.peakRssMb);
    }

    private static double mean(double[] a) {
        return Arrays.stream(a).average().orElse(0);
    }

    private static double stddev(double[] a) {
        if (a.length < 2) return 0;
        double m = mean(a), s = 0;
        for (double v : a) s += (v - m) * (v - m);
        return Math.sqrt(s / (a.length - 1));
    }

    private static String name(String path) {
        String f = Path.of(path).getFileName().toString();
        return f.endsWith(".gguf") ? f.substring(0, f.length() - 5) : f;
    }

    private static void usage(PrintStream out) {
        out.println(
                """
                jinfer-xbench - llama-bench-parity harness for the x tree (generic loader)

                usage: jinfer-xbench -m <model.gguf> [-m ...] [options]
                  -m, --model <path>      model to benchmark (repeatable; any architecture the
                                          x tree loads)
                  -p, --n-prompt <N>      prefill tokens (default 512; 0 to skip pp)
                  -n, --n-gen <N>         decode tokens  (default 128; 0 to skip tg)
                  -r, --repetitions <N>   timed reps     (default 5)
                  -w, --warmup <N>        min warmup passes; warms adaptively until throughput settles (default 2)
                      --no-warmup         skip warmup runs before benchmarking
                  -t, --threads <N>       pp and tg threads (default physical cores)
                      --ctx <N>           override context size for both tests
                                          (default per test, as llama-bench: p for pp, n for tg)
                      --media <path>      also benchmark projected-media cold vs warm latency
                                          (needs a vision projector: --with media=<mmproj.gguf>)
                      --with <cap=path>   attach a companion file (repeatable; xcli's convention)\
                """);
    }
}
