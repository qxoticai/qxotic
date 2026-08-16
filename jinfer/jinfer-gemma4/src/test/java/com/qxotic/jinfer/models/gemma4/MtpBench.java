// MTP benchmark: plain greedy vs speculative decode at depths 1..3, across prompt types
// (predictable list / code / prose), production engine config (jam). Reports decode tok/s,
// acceptance rate, tokens-per-forward, and the honest verdict of where MTP pays.
//   mvn test -pl jinfer-gemma4 -Dtest=MtpBench -Dsurefire.excludedGroups= \
//       [-Djinfer.args="128 3 1,2,3"]   (maxTokens reps depths)
package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.boundary.Batch;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.llm.Generator.Constraints;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.llm.SpeculativeDecoding.SpeculationResult;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.toknroll.Tokenizer;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class MtpBench {

    private static final String MODEL_REF = "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0";
    private static final String SIDECAR_REF =
            "hf.co/unsloth/gemma-4-E2B-it-GGUF/mtp-gemma-4-E2B-it.gguf";

    record Case(String name, String prompt) {}

    @Test
    @Tag("bench")
    void run() throws Exception {
        String argv = System.getProperty("jinfer.args", "");
        main(argv.isBlank() ? new String[0] : argv.trim().split("\\s+"));
    }

    /** Direct entry for a hand-built classpath that includes the desired JAM backend. */
    public static void main(String[] args) throws Exception {
        int maxTokens = args.length > 0 ? Integer.parseInt(args[0]) : 128;
        int reps = args.length > 1 ? Integer.parseInt(args[1]) : 3;
        int[] depths = args.length > 2 ? depths(args[2]) : new int[] {1, 2, 3};

        Path model = TestModels.require(MODEL_REF);
        Path sidecar = TestModels.require(SIDECAR_REF);
        Gemma4 m = Gemma4.loadWithMtp(model, sidecar, Arena.ofAuto());
        Tokenizer tk = m.tokenizer();
        int bos = SpecialTokens.find(tk, "<bos>").orElse(2);
        Set<Integer> stops = Gemma4MtpIdentityTest.stopTokens(tk);
        int vocab = m.configuration().vocabularySize();

        Case[] cases = {
            new Case(
                    "list (predictable)",
                    "List the numbers from one to fifty as words, comma separated:"),
            new Case(
                    "code",
                    "Write a complete Python function that parses a CSV line, handling quoted"
                            + " fields:\n"),
            new Case(
                    "prose",
                    "Write a vivid, original short story opening set in a lighthouse during a"
                            + " storm."),
        };

        System.out.printf(
                "%-20s %-9s %8s %8s %10s %10s %9s%n",
                "case", "mode", "tokens", "tok/s", "accept", "tok/fwd", "speedup");
        for (Case c : cases) {
            int[] ids = Gemma4MtpIdentityTest.withBos(bos, tk.encode(c.prompt).toList());

            // plain greedy baseline (best of reps)
            double plainBest = 0;
            int plainCount = 0;
            for (int r = 0; r < reps; r++) {
                try (Gemma4.State s = m.newState(4096, Math.max(16, ids.length))) {
                    m.ingest(s, Batch.prefill(ids));
                    long t0 = System.nanoTime();
                    int n = 0;
                    int tok =
                            Ops.argmax(Views.castToSegmentBacked(m.logits(s), "logits"), 0, vocab);
                    while (n < maxTokens && !stops.contains(tok)) {
                        m.ingest(s, Batch.step(tok));
                        tok =
                                Ops.argmax(
                                        Views.castToSegmentBacked(m.logits(s), "logits"), 0, vocab);
                        n++;
                    }
                    double tps = n / ((System.nanoTime() - t0) / 1e9);
                    plainBest = Math.max(plainBest, tps);
                    plainCount = n;
                }
            }
            System.out.printf(
                    "%-20s %-9s %8d %8.1f %10s %10s %9s%n",
                    c.name, "plain", plainCount, plainBest, "-", "-", "1.00x");

            for (int depth : depths) {
                double best = 0;
                SpeculationResult last = null;
                for (int r = 0; r < reps; r++) {
                    try (Gemma4.State s = m.newState(4096, Math.max(16, ids.length))) {
                        m.ingest(s, Batch.prefill(ids));
                        long t0 = System.nanoTime();
                        SpeculationResult res =
                                m.speculate(
                                        s,
                                        Sampler.ARGMAX,
                                        new Constraints(maxTokens, Duration.ZERO, stops),
                                        depth,
                                        null);
                        double tps = res.emitted().length() / ((System.nanoTime() - t0) / 1e9);
                        best = Math.max(best, tps);
                        last = res;
                    }
                }
                double acc = last.drafted() == 0 ? 0 : (double) last.accepted() / last.drafted();
                double tpf = (double) last.emitted().length() / last.forwards();
                System.out.printf(
                        "%-20s %-9s %8d %8.1f %9.0f%% %10.2f %8.2fx%n",
                        c.name,
                        "spec d=" + depth,
                        last.emitted().length(),
                        best,
                        100 * acc,
                        tpf,
                        best / plainBest);
            }
        }
    }

    private static int[] depths(String csv) {
        String[] parts = csv.split(",");
        int[] depths = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            depths[i] = Integer.parseInt(parts[i].trim());
        }
        return depths;
    }
}
