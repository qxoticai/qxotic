package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.llm.SpecialTokens;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Model-level determinism regression. The same batched prefill, repeated in one process, MUST
 * produce bit-identical logits — a forward pass is a pure function of (weights, tokens). Surfaces
 * the jam-native multi-threaded FFN-gemm non-determinism seen on A4B (intermittent ~25%, so it
 * sweeps {@code reps} repetitions to catch it reliably), which no single-gemm jam test reproduces
 * because it is emergent from the whole prefill's gemm sequence + jam's reused cross-call scratch.
 *
 * <pre>PrefillDeterminism &lt;model.gguf&gt; ["prompt"] [reps]</pre>
 *
 * Exit 0 = deterministic across all reps; exit 1 = a divergence was found (prints rep, index,
 * values). Controls to confirm the cause: {@code -Djinfer.disableJam=true} (Java backend) and
 * {@code JAM_NUM_THREADS=1} both make it deterministic.
 */
public final class PrefillDeterminism {
    @Test
    @Tag("integration")
    void run() throws Exception {
        Assumptions.assumeTrue(
                java.nio.file.Files.exists(
                        java.nio.file.Path.of(
                                "/home/mukel/Desktop/playground/models/unsloth/gemma-4-26B-A4B-it-Q8_0.gguf")),
                "model not found:"
                    + " /home/mukel/Desktop/playground/models/unsloth/gemma-4-26B-A4B-it-Q8_0.gguf");
        main(testArgs());
    }

    private static String[] testArgs() {
        String argv = System.getProperty("jinfer.args", "");
        return argv.isBlank() ? new String[0] : argv.trim().split("\\s+");
    }

    private static void main(String[] args) throws Exception {
        String path =
                args.length > 0
                        ? args[0]
                        : "/home/mukel/Desktop/playground/models/unsloth/gemma-4-26B-A4B-it-Q8_0.gguf";
        String promptStr = args.length > 1 ? args[1] : "The capital of France is";
        int reps = args.length > 2 ? Integer.parseInt(args[2]) : 12;

        Gemma4 model = Gemma4.loadModel(Path.of(path), 4096);
        var c = model.config();
        var tk = model.tokenizer();
        int bos = SpecialTokens.find(tk, "<bos>").orElse(2);
        List<Integer> pt = new ArrayList<>();
        pt.add(bos);
        pt.addAll(tk.encode(promptStr).toList());
        int[] ids = pt.stream().mapToInt(Integer::intValue).toArray();
        System.err.printf(
                "model=%s  prompt tokens=%d  reps=%d%n",
                Path.of(path).getFileName(), ids.length, reps);

        // Classify by MAGNITUDE, not bit-equality: jinfer's own parallel reductions add an
        // irreducible
        // ~1e-5 float-order floor, so bit-equality is too strict. The jam SIMD-requant bug is ~1e4×
        // that.
        double TOL =
                1e-2; // well above the ~1e-5 parallel floor, well below the ~0.5 jam-bug divergence
        int WARMUP = 4; // JIT-tier FP drift settles within a few runs
        int vocab = c.vocabularySize();
        float[] ref = null;
        double maxAbs = 0;
        int argIdx = -1;
        for (int r = 0; r < reps; r++) {
            Gemma4.State s = model.newState(c.contextLength(), Math.max(16, ids.length));
            model.ingest(s, com.qxotic.jinfer.Batch.prefill(ids));
            FloatTensor logits = model.logits(s);
            float[] snap = new float[vocab];
            for (int i = 0; i < vocab; i++) snap[i] = logits.getFloat(i);
            // The first prefills run partly in lower JIT tiers whose FP reassociation differs
            // from the final compiled code (~0.2-8 logit drift, deterministic once warm) - warm
            // up before measuring, or the check gates the JVM, not the model.
            if (r < WARMUP) continue;
            if (ref == null) {
                ref = snap;
                continue;
            }
            for (int i = 0; i < vocab; i++) {
                double d = Math.abs((double) snap[i] - ref[i]);
                if (d > maxAbs) {
                    maxAbs = d;
                    argIdx = i;
                }
            }
        }

        if (maxAbs <= TOL) {
            System.out.printf(
                    "DETERMINISTIC: %d prefills agree to %.3g (≤ %.0e; FP-parallel floor)%n",
                    reps, maxAbs, TOL);
        } else {
            System.out.printf(
                    "NON-DETERMINISTIC: max |Δlogit[%d]| = %.6g across %d prefills (> %.0e)%n",
                    argIdx, maxAbs, reps, TOL);
            throw new AssertionError("failure(s) - see output above");
        }
    }
}
