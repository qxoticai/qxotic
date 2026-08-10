package com.qxotic.jinfer.x.bench;

import com.qxotic.jinfer.x.Ops;
import com.qxotic.jinfer.x.boundary.Batch;
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
        try (var xs = xm.newState(ctx, Math.max(16, prompt.length), Arena.ofAuto())) {
            xs.enter();
            try {
                xm.forward(xs, Batch.prefill(prompt));
                xIds = new int[n];
                @SuppressWarnings("unchecked")
                MemoryView<MemorySegment> logits = (MemoryView<MemorySegment>) xm.head(xs, 0);
                int tok = Ops.argmax(logits, 0, vocab);
                for (int g = 0; g < n; g++) {
                    xIds[g] = tok;
                    xm.forward(xs, Batch.step(tok));
                    logits = (MemoryView<MemorySegment>) xm.head(xs, 0);
                    tok = Ops.argmax(logits, 0, vocab);
                }
            } finally {
                xs.exit();
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
