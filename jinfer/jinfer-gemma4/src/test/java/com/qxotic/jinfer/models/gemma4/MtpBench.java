// Stage-4 MTP benchmark: plain greedy vs speculative decode at depths 1..3, across prompt types
// (predictable list / code / prose), production engine config (jam). Reports decode tok/s,
// acceptance rate, tokens-per-forward, and the honest verdict of where MTP pays.
//   java ... com.qxotic.jinfer.models.gemma4.MtpBench [maxTokens] [reps]
package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Batch;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

public final class MtpBench {

    record Case(String name, String prompt) {}

    public static void main(String[] args) throws Exception {
        int maxTokens = args.length > 0 ? Integer.parseInt(args[0]) : 128;
        int reps = args.length > 1 ? Integer.parseInt(args[1]) : 3;
        Path model =
                Path.of("/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf");
        Path sidecar =
                Path.of("/home/mukel/Desktop/playground/models/unsloth/mtp-gemma-4-E2B-it.gguf");
        if (!Files.exists(model) || !Files.exists(sidecar)) {
            System.out.println("MtpBench: model/sidecar absent, skipping");
            return;
        }
        Gemma4 m = Gemma4.loadModel(model, 4096, sidecar);
        var tk = m.tokenizer();
        int bos = tk.getSpecialTokens().getOrDefault("<bos>", 2);
        Set<Integer> stops = m.stopTokens();
        int vocab = m.config().vocabularySize();

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
            int[] ids = withBos(bos, tk.encode(c.prompt).toList());

            // plain greedy baseline (best of reps)
            double plainBest = 0;
            int plainCount = 0;
            for (int r = 0; r < reps; r++) {
                Gemma4.State s = m.newState(4096, Math.max(16, ids.length));
                m.ingest(s, Batch.prefill(ids));
                long t0 = System.nanoTime();
                int n = 0;
                int tok = m.logits(s, 0).argmax(0, vocab);
                while (n < maxTokens && !stops.contains(tok)) {
                    m.ingest(s, Batch.step(tok));
                    tok = m.logits(s, 0).argmax(0, vocab);
                    n++;
                }
                double tps = n / ((System.nanoTime() - t0) / 1e9);
                plainBest = Math.max(plainBest, tps);
                plainCount = n;
            }
            System.out.printf(
                    "%-20s %-9s %8d %8.1f %10s %10s %9s%n",
                    c.name, "plain", plainCount, plainBest, "-", "-", "1.00x");

            for (int depth : new int[] {1, 2, 3}) {
                double best = 0;
                Gemma4Speculative.Result last = null;
                for (int r = 0; r < reps; r++) {
                    Gemma4.State s = m.newState(4096, Math.max(16, ids.length));
                    m.ingest(s, Batch.prefill(ids));
                    long t0 = System.nanoTime();
                    Gemma4Speculative.Result res =
                            Gemma4Speculative.generate(m, s, maxTokens, stops, depth);
                    double tps = res.tokens().size() / ((System.nanoTime() - t0) / 1e9);
                    best = Math.max(best, tps);
                    last = res;
                }
                double acc = last.drafted() == 0 ? 0 : (double) last.accepted() / last.drafted();
                double tpf = (double) last.tokens().size() / last.forwards();
                System.out.printf(
                        "%-20s %-9s %8d %8.1f %9.0f%% %10.2f %8.2fx%n",
                        c.name,
                        "spec d=" + depth,
                        last.tokens().size(),
                        best,
                        100 * acc,
                        tpf,
                        best / plainBest);
            }
        }
    }

    static int[] withBos(int bos, List<Integer> enc) {
        int[] ids = new int[enc.size() + 1];
        ids[0] = bos;
        for (int i = 0; i < enc.size(); i++) ids[i + 1] = enc.get(i);
        return ids;
    }
}
