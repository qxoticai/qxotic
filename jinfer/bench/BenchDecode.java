package com.qxotic.jinfer;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

/**
 * Steady-state decode harness. Runs ONE continuous generation of {@code total} tokens from a
 * {@code depth}-deep context, timestamping every token, then reports the distribution over the
 * post-warmup tail only — so the JIT-settling prefix and the prefill→decode boundary don't pollute the
 * number (which is what makes {@link BenchPrefill}'s per-iteration decode number swing for tiny models).
 *
 * <p>Output: a windowed tok/s curve (throughput vs growing context) plus the steady-state per-token
 * latency distribution (p10/median/p90 + sd%) over [warmup, end). The sd% is the honest jitter readout.
 *
 * <p>Usage: {@code BenchDecode <model.gguf> [depth=8] [total=2000] [warmup=min(600,total/3)] [window=200]}
 */
public final class BenchDecode {
    public static void main(String[] args) throws Exception {
        String path = args[0];
        int depth  = args.length > 1 ? Integer.parseInt(args[1]) : 8;
        int total  = args.length > 2 ? Integer.parseInt(args[2]) : 2000;
        int warmup = args.length > 3 ? Integer.parseInt(args[3]) : Math.min(600, total / 3);
        int window = args.length > 4 ? Integer.parseInt(args[4]) : 200;

        Model model = LegacyModelLoader.loadModel(Path.of(path), depth + total + 16);
        Sampler sampler = Engine.configuredSampler(model, false, 0.0f, 1.0f, 42);   // greedy: deterministic timing

        StringBuilder sb = new StringBuilder();
        while (model.tokenizer().encodeWithSpecialTokens(sb.toString()).size() < depth)
            sb.append("The quick brown fox jumps over the lazy dog. ");
        List<Integer> prompt = new ArrayList<>(model.tokenizer().encodeWithSpecialTokens(sb.toString()).subList(0, depth));

        long[] ts = new long[total + 1];           // ts[i] = wall-clock when decode token i was emitted
        int[] n = {0};
        IntConsumer onTok = t -> { if (n[0] < ts.length) ts[n[0]++] = System.nanoTime(); };
        Engine.Params params = new Engine.Params(sampler, total, 0, new Engine.StopSpec(Set.of(), List.of()), false);

        System.err.printf("model=%s  depth=%d  total=%d  warmup=%d  window=%d  backend=%s%n",
                path.substring(path.lastIndexOf('/') + 1), depth, total, warmup, window,
                System.getProperty("jinfer.kernels", "jam"));

        InferenceState state = model.createNewState();
        Engine.generate(model, state, 0, prompt, params, new Engine.Listener(onTok, null, null, null), GenerationHooks.NONE);
        int got = n[0];
        SpinPool.traceReport("decode", got);   // -Djinfer.spinTrace: per-token dispatch/barrier breakdown
        if (got < warmup + window) {
            System.err.printf("only %d tokens generated (early stop?) — need > warmup+window=%d%n", got, warmup + window);
            return;
        }

        // Windowed throughput: each window is `window` tokens at a roughly fixed (growing) context depth.
        System.err.println("  windowed tok/s (context grows top→bottom — shows decode vs depth):");
        for (int w = 0; w + window < got; w += window) {
            double dtMs = (ts[w + window] - ts[w]) / 1e6;
            System.err.printf("    ctx ~%-6d %7.2f tok/s%s%n", depth + w + window, window / (dtMs / 1000.0),
                    w < warmup ? "   (warmup)" : "");
        }

        // Steady-state per-token latency distribution over [warmup, got): the stable signal.
        int lo = Math.max(1, warmup);
        double[] ms = new double[got - lo];
        for (int i = lo; i < got; i++) ms[i - lo] = (ts[i] - ts[i - 1]) / 1e6;
        double mean = 0; for (double v : ms) mean += v; mean /= ms.length;
        double var = 0; for (double v : ms) var += (v - mean) * (v - mean); double sd = Math.sqrt(var / ms.length);
        double[] sorted = ms.clone(); Arrays.sort(sorted);
        double p10 = sorted[(int) (0.10 * sorted.length)], p50 = sorted[sorted.length / 2], p90 = sorted[(int) (0.90 * sorted.length)];
        System.err.printf("%n=== STEADY-STATE decode, tok [%d,%d), ctx ~%d–%d ===%n", lo, got, depth + lo, depth + got);
        System.err.printf("  per-token ms:  p10 %.3f  median %.3f  p90 %.3f   sd %.3f (%.1f%% of mean)%n",
                p10, p50, p90, sd, 100 * sd / mean);
        System.err.printf("  decode tok/s:  MEDIAN %.2f   (p90-latency floor %.2f, p10-latency ceil %.2f)%n",
                1000 / p50, 1000 / p90, 1000 / p10);
    }
}
