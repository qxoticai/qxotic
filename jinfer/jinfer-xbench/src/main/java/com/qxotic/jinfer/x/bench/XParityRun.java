package com.qxotic.jinfer.x.bench;

import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.kernels.Ops;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

/**
 * Greedy token-id parity of old Lfm2 vs the x port IN THE SAME JVM WITH WHATEVER MATMUL BACKENDS
 * THE CLASSPATH CARRIES — the JAM-arm gate XLfm2Test cannot be (surefire strips jam from the test
 * classpath so it only ever proves floor-vs-floor). Run twice:
 *
 * <pre>
 *   java ... XParityRun -m model.gguf [n]                          # both trees on native jam
 *   java ... -Djinfer.disableJam=true XParityRun -m model.gguf [n] # both trees on the Java floor
 * </pre>
 *
 * Both runs must print PARITY OK: the first proves the x JAM arm issues the same native gemms the
 * old Dispatch does (same libjam, same kernels — diffs should be floor-level tiny), the second
 * reproduces XLfm2Test's floor gate outside surefire. Prompt is synthetic in-range ids, so the
 * tokenizer never enters the comparison.
 *
 * <p>MoE CAVEAT (LFM2-8B-A1B): on MoE checkpoints only the {@code -Djinfer.disableJam=true} run is
 * a correctness gate. Discrete top-k expert routing amplifies tiny accumulation-order differences
 * between the native and Java backends into different expert sets, which then legitimately diverge
 * the logits (~1e-1 max-abs, argmax flips) — and the divergence is symmetric: old-jam vs old-floor
 * disagrees exactly as x-jam vs x-floor does, with x landing on the floor-consistent side. A jam-on
 * mismatch here means "backend mix", not "x bug"; the floor run (and XLfm2Test's MoE leg) carries
 * the correctness signal.
 */
public final class XParityRun {

    public static void main(String[] args) throws Exception {
        Path model = null;
        int n = 64;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-m", "--model" -> model = Path.of(args[++i]);
                default -> n = Integer.parseInt(args[i]);
            }
        }
        if (model == null) {
            System.err.println("usage: XParityRun -m model.gguf [n]");
            System.exit(2);
        }

        var om = com.qxotic.jinfer.models.lfm2.Lfm2.loadModel(model, Arena.ofAuto());
        var xm = com.qxotic.jinfer.x.models.lfm2.Lfm2.loadModel(model, Arena.ofAuto());
        int vocab = om.config().vocabularySize();
        int[] prompt = new int[32];
        for (int i = 0; i < prompt.length; i++) prompt[i] = (i * 17 + 1) % vocab;
        int ctx = prompt.length + n + 16;

        int[] oldIds;
        try (var os = om.newState(ctx, Math.max(16, prompt.length))) {
            om.ingest(os, com.qxotic.jinfer.Batch.prefill(prompt));
            oldIds = new int[n];
            int tok = om.logits(os).argmax();
            for (int g = 0; g < n; g++) {
                oldIds[g] = tok;
                om.ingest(os, com.qxotic.jinfer.Batch.step(tok));
                tok = om.logits(os).argmax();
            }
        }
        int[] xIds;
        try (var xs = xm.newState(ctx, Math.max(16, prompt.length))) {
            xm.ingest(xs, Batch.prefill(prompt));
            xIds = new int[n];
            MemoryView<MemorySegment> logits = Views.castToSegmentBacked(xm.logits(xs), "logits");
            int tok = Ops.argmax(logits, 0, vocab);
            for (int g = 0; g < n; g++) {
                xIds[g] = tok;
                xm.ingest(xs, Batch.step(tok));
                logits = Views.castToSegmentBacked(xm.logits(xs), "logits");
                tok = Ops.argmax(logits, 0, vocab);
            }
        }
        // token parity is the strict gate; per-step logits closeness is implied by 64 equal
        // argmaxes over a 128k vocab.
        int divergent = -1;
        for (int g = 0; g < n; g++) {
            if (oldIds[g] != xIds[g]) {
                divergent = g;
                break;
            }
        }
        if (divergent < 0) {
            System.out.printf("PARITY OK (%d greedy tokens identical)%n", n);
        } else {
            System.out.printf(
                    "PARITY FAIL at step %d: old=%d x=%d%n",
                    divergent, oldIds[divergent], xIds[divergent]);
            System.exit(1);
        }
    }
}
