package com.qxotic.jinfer;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Warmed-up prefill benchmark. Loads the model once, then runs prefill many times in the SAME JVM so
 * the JIT warms up; reports prefill tokens/s per iteration plus a summary over the measured window.
 * budget=1 → the engine crosses the prefill/decode boundary, so promptMillis is pure prefill time.
 *
 * Usage: BenchPrefill <model.gguf> [promptTokens=512] [warmup=8] [measure=12]
 */
public final class BenchPrefill {
    public static void main(String[] args) throws Exception {
        String path = args[0];
        int target  = args.length > 1 ? Integer.parseInt(args[1]) : 512;
        int warmup  = args.length > 2 ? Integer.parseInt(args[2]) : 8;
        int measure = args.length > 3 ? Integer.parseInt(args[3]) : 12;
        int gen     = args.length > 4 ? Integer.parseInt(args[4]) : 0;   // >0 -> measure DECODE tok/s

        Model model = LegacyModelLoader.loadModel(Path.of(path), 4096);
        Sampler sampler = Engine.configuredSampler(model, false, 0.0f, 1.0f, 42);

        // EXACTLY `target` raw tokens (no chat wrapping) — matches llama.cpp's pp<N> token count.
        StringBuilder sb = new StringBuilder();
        while (model.tokenizer().encodeWithSpecialTokens(sb.toString()).size() < target)
            sb.append("The quick brown fox jumps over the lazy dog. ");
        List<Integer> all = model.tokenizer().encodeWithSpecialTokens(sb.toString());
        List<Integer> promptTokens = new ArrayList<>(all.subList(0, target));
        int nPrompt = promptTokens.size();

        System.err.printf("model=%s  prompt=%d tokens  backend=%s  warmup=%d measure=%d%n",
                path.substring(path.lastIndexOf('/') + 1), nPrompt,
                System.getProperty("jinfer.kernels", "jam"), warmup, measure);

        Engine.Params params = new Engine.Params(sampler, gen > 0 ? gen : 1, 0,
                new Engine.StopSpec(Set.of(), List.of()), false);
        Engine.Listener noop = new Engine.Listener(t -> {}, null, null, null);
        String what = gen > 0 ? "decode" : "prefill";

        double[] tps = new double[warmup + measure];
        for (int i = 0; i < warmup + measure; i++) {
            InferenceState state = model.createNewState();
            Engine.GenerationResult r = Engine.generate(model, state, 0, promptTokens, params, noop, GenerationHooks.NONE);
            tps[i] = gen > 0 ? gen / (r.predictedMillis() / 1000.0) : nPrompt / (r.promptMillis() / 1000.0);
            System.err.printf("  [%-7s %2d] %s %7.2f tok/s%n", i < warmup ? "warmup" : "measure", i, what, tps[i]);
        }

        double[] m = Arrays.copyOfRange(tps, warmup, warmup + measure);
        Arrays.sort(m);
        double sum = 0; for (double v : m) sum += v;
        System.err.printf("%n=== %s (warmed) over %d iters: mean %.2f  median %.2f  min %.2f  max %.2f tok/s ===%n",
                what, measure, sum / measure, m[measure / 2], m[0], m[measure - 1]);
    }
}
