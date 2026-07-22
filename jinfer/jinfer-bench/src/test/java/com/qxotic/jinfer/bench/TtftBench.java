// TTFT benchmark for the prompt cache: time-to-first-token of a follow-up request over a long
// cached history, vs a cold full prefill - the numbers compared against llama.cpp's caching.
//   warm:           java ... TtftBench warm <model.gguf> [reps]
//   frozen-compile: java ... TtftBench frozen-compile <model.gguf> <out.jkv>
//   frozen-serve:   java ... TtftBench frozen-serve <model.gguf> <in.jkv> (fresh JVM:
// cold-from-disk TTFT)
//   dump-text:      java ... TtftBench dump-text <dir>   (story + question, shared with llama.cpp)
package com.qxotic.jinfer.bench;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.cache.CacheStore;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.cache.StateCodec;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.models.gemma4.Gemma4;
import com.qxotic.jinfer.models.lfm2.Lfm2;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

public final class TtftBench {

    // ~2500-token history: story turn + a generated assistant reply. Same text shared with
    // llama.cpp.
    static String story() {
        StringBuilder s =
                new StringBuilder(
                        "Remember this codeword: PELICAN. Summarize the following notes.\n");
        for (int i = 0; i < 90; i++) {
            s.append("Entry ")
                    .append(i)
                    .append(": the expedition logged river depth, canopy density, ")
                    .append("and soil acidity at station ")
                    .append(i)
                    .append(
                            "; readings were nominal and the weather held clear through the"
                                    + " afternoon.\n");
        }
        return s.toString();
    }

    static final String QUESTION = "What was the codeword? One word only.";
    static final int REPLY_BUDGET = 40;

    @Test
    @Tag("bench")
    void run() throws Exception {
        Assumptions.assumeTrue(
                !System.getProperty("jinfer.args", "").isBlank(),
                "set -Djinfer.args=\"...\" to run this tool");
        main(testArgs());
    }

    private static String[] testArgs() {
        String argv = System.getProperty("jinfer.args", "");
        return argv.isBlank() ? new String[0] : argv.trim().split("\\s+");
    }

    private static void main(String[] args) throws Exception {
        switch (args[0]) {
            case "dump-text" -> {
                Files.writeString(Path.of(args[1], "story.txt"), story());
                Files.writeString(Path.of(args[1], "question.txt"), QUESTION);
                System.out.println("wrote story.txt + question.txt");
            }
            case "warm" ->
                    warm(
                            load(args[1]),
                            Path.of(args[1]),
                            args.length > 2 ? Integer.parseInt(args[2]) : 5);
            case "frozen-compile" ->
                    frozenCompile(load(args[1]), Path.of(args[1]), Path.of(args[2]));
            case "frozen-serve" -> frozenServe(load(args[1]), Path.of(args[1]), Path.of(args[2]));
            default -> throw new IllegalArgumentException(args[0]);
        }
    }

    /** A loaded model paired with its per-turn template (the cache benches drive turns). */
    record Bench<S extends RuntimeState>(LoadedModel<S> model, TurnTemplate tpl) {}

    static Bench<?> load(String path) throws Exception {
        long t0 = System.nanoTime();
        String lower = path.toLowerCase();
        Bench<?> b;
        if (lower.contains("gemma")) {
            var m = Gemma4.loadModel(Path.of(path), 4096);
            b = new Bench<>(m.loaded(), m.turnTemplate().orElseThrow());
        } else {
            var m = Lfm2.loadModel(Path.of(path), 4096);
            b = new Bench<>(m.loaded(), m.turnTemplate().orElseThrow());
        }
        System.err.printf("model load: %.0f ms%n", (System.nanoTime() - t0) / 1e6);
        return b;
    }

    /**
     * Builds the cached history (story turn + generated reply), then benches: cold full-prefill
     * TTFT vs warm resume TTFT for the follow-up question.
     */
    static <S extends RuntimeState> void warm(Bench<S> bench, Path gguf, int reps) {
        LoadedModel<S> model = bench.model();
        TurnTemplate tpl = bench.tpl();
        StateCodec<S> codec = model.codec();
        PromptCache<S> cache =
                new PromptCache<>(codec, CacheStore.inMemory(), 8L << 30, model.seed());

        // history: [start][user: story][genPrompt] -> reply -> closeTurn, all committed
        CachedSession<S> a =
                CachedSession.resume(
                        model.model(), cache, model.model().newState(4096, 512), new long[0]);
        a.ingest(concat(tpl.conversationStart(), tpl.encodeTurn(Message.user(story()))));
        String reply = decode(model, a, tpl, REPLY_BUDGET);
        long[] history = a.fingerprints();
        System.err.println("history: " + history.length + " positions; reply: " + reply.strip());

        List<Batch> followUp =
                concat(tpl.encodeTurn(Message.user(QUESTION)), tpl.generationPrompt(true));
        int[] historyIds = new int[history.length];
        for (int i = 0; i < history.length; i++) historyIds[i] = (int) history[i];

        // tier 1: live pooled sessions, pre-resumed OUTSIDE the timing (a SessionPool holds them
        // resident between requests) - the timed leg is pure append-only delta ingest + first
        // token.
        List<CachedSession<S>> live = new ArrayList<>();
        for (int r = 0; r < reps; r++) {
            CachedSession<S> s =
                    CachedSession.resume(
                            model.model(), cache, model.model().newState(4096, 512), history);
            if (s.position() != history.length)
                throw new IllegalStateException("live resume " + s.position());
            live.add(s);
        }

        double[] cold = new double[reps], warmMs = new double[reps], tier1 = new double[reps];
        String firstTok = "";
        for (int r = 0; r < reps; r++) {
            // cold: full prefill of history + follow-up
            S s1 = model.model().newState(4096, 512);
            long t0 = System.nanoTime();
            for (Batch b : Batch.prepare(concat(List.of(Batch.prefill(historyIds)), followUp), 512))
                model.model().ingest(s1, b);
            int tok1 = model.model().logits(s1).argmax();
            cold[r] = (System.nanoTime() - t0) / 1e6;

            // warm: cache resume + follow-up only
            S s2 = model.model().newState(4096, 512);
            long t1 = System.nanoTime();
            CachedSession<S> b2 = CachedSession.resume(model.model(), cache, s2, history);
            if (b2.position() != history.length)
                throw new IllegalStateException("resume " + b2.position());
            b2.ingest(followUp);
            int tok2 = model.model().logits(s2).argmax();
            warmMs[r] = (System.nanoTime() - t1) / 1e6;
            if (tok1 != tok2)
                System.err.println(
                        "NOTE: first tokens differ (cold " + tok1 + " vs warm " + tok2 + ")");
            firstTok = model.tokenizer().decode(new int[] {tok2});

            // tier 1: append-only on the live session (llama.cpp in-place slot equivalent)
            CachedSession<S> p = live.get(r);
            long t2 = System.nanoTime();
            p.ingest(followUp);
            int tok3 = model.model().logits(p.state()).argmax();
            tier1[r] = (System.nanoTime() - t2) / 1e6;
            if (tok3 != tok2)
                System.err.println(
                        "NOTE: tier-1 first token differs (" + tok3 + " vs " + tok2 + ")");
        }
        int delta = Batch.tokenIds(followUp).length;
        System.out.printf(
                "history=%d deltaTokens=%d firstTok=%s%n", history.length, delta, firstTok.strip());
        System.out.printf("cold        TTFT: best %.1f ms  mean %.1f ms%n", best(cold), mean(cold));
        System.out.printf(
                "tier2 warm  TTFT: best %.1f ms  mean %.1f ms  (block restore + delta)%n",
                best(warmMs), mean(warmMs));
        System.out.printf(
                "tier1 pool  TTFT: best %.1f ms  mean %.1f ms  (append-only on the live session)%n",
                best(tier1), mean(tier1));
    }

    /**
     * Freezes the STATIC prefix (conversationStart + the story turn - fully deterministic, no
     * generated tokens) so the serve side can rebuild the exact fingerprints from text.
     */
    static <S extends RuntimeState> void frozenCompile(Bench<S> bench, Path gguf, Path out)
            throws Exception {
        LoadedModel<S> model = bench.model();
        TurnTemplate tpl = bench.tpl();
        StateCodec<S> codec = model.codec();
        List<Batch> prefix = concat(tpl.conversationStart(), tpl.encodeTurn(Message.user(story())));
        PromptCache<S> build =
                new PromptCache<>(codec, CacheStore.inMemory(), Long.MAX_VALUE, model.seed());
        CachedSession<S> s =
                CachedSession.resume(
                        model.model(), build, model.model().newState(4096, 512), new long[0]);
        s.ingest(prefix);
        build.freeze(out);
        System.out.printf(
                "frozen %d positions -> %s (%.1f MB)%n", s.length(), out, Files.size(out) / 1e6);
    }

    /** Fresh-JVM serve: open + restore + follow-up prefill + first token. */
    static <S extends RuntimeState> void frozenServe(Bench<S> bench, Path gguf, Path file)
            throws Exception {
        LoadedModel<S> model = bench.model();
        TurnTemplate tpl = bench.tpl();
        StateCodec<S> codec = model.codec();
        List<Batch> followUp =
                concat(tpl.encodeTurn(Message.user(QUESTION)), tpl.generationPrompt(true));

        // JIT warmup on a throwaway state (a running server / native image has warm code; without
        // this the first forward pays ~2.5s of compilation, which is JVM startup, not cache cost)
        S w = model.model().newState(4096, 512);
        int[] warmIds = new int[256];
        java.util.Arrays.fill(warmIds, 5);
        for (int i = 0; i < 2; i++) {
            S ws = model.model().newState(4096, 512);
            model.model().ingest(ws, Batch.prefill(warmIds));
            model.model().logits(ws).argmax();
        }

        long t0 = System.nanoTime();
        com.qxotic.jinfer.cache.FrozenBlocks frozen =
                com.qxotic.jinfer.cache.FrozenBlocks.open(file, model.seed());
        double openMs = (System.nanoTime() - t0) / 1e6;

        // request fingerprints: re-derived from the same static text (deterministic template)
        S state = model.model().newState(4096, 512);
        long t1 = System.nanoTime();
        int[] ids =
                Batch.tokenIds(
                        concat(tpl.conversationStart(), tpl.encodeTurn(Message.user(story()))));
        long[] fp = new long[ids.length];
        for (int i = 0; i < ids.length; i++) fp[i] = ids[i];
        int restored = frozen.serve(model.model(), codec, model.seed(), state, fp).position();
        if (restored == 0) throw new IllegalStateException("frozen restore missed");
        for (Batch b : Batch.prepare(followUp, 512)) model.model().ingest(state, b);
        int tok = model.model().logits(state).argmax();
        double ttft = (System.nanoTime() - t1) / 1e6;
        System.out.printf(
                "frozen open: %.1f ms;  first-serve TTFT: %.1f ms;  restored=%d  firstTok=%s%n",
                openMs, ttft, restored, model.tokenizer().decode(new int[] {tok}).strip());
        // steady state (long-running server): re-serve in-process
        for (int r = 0; r < 3; r++) {
            S s2 = model.model().newState(4096, 512);
            long t2 = System.nanoTime();
            com.qxotic.jinfer.cache.FrozenBlocks f2 =
                    com.qxotic.jinfer.cache.FrozenBlocks.open(file, model.seed());
            int n2 = f2.serve(model.model(), codec, model.seed(), s2, fp).position();
            for (Batch b : Batch.prepare(followUp, 512)) model.model().ingest(s2, b);
            model.model().logits(s2).argmax();
            System.out.printf(
                    "steady serve TTFT (open+restore+followUp+token): %.1f ms (restored %d)%n",
                    (System.nanoTime() - t2) / 1e6, n2);
        }
    }

    static <S extends RuntimeState> String decode(
            LoadedModel<S> model, CachedSession<S> s, TurnTemplate tpl, int max) {
        s.ingest(tpl.generationPrompt(true));
        Set<Integer> stops = model.stopTokens();
        StringBuilder out = new StringBuilder();
        int tok = model.model().logits(s.state()).argmax();
        for (int n = 0; n < max && !stops.contains(tok); n++) {
            out.append(model.tokenizer().decode(new int[] {tok}));
            s.step(tok);
            tok = model.model().logits(s.state()).argmax();
        }
        s.ingest(tpl.closeTurn());
        return out.toString();
    }

    static List<Batch> concat(List<Batch>... groups) {
        List<Batch> out = new ArrayList<>();
        for (List<Batch> g : groups) out.addAll(g);
        return out;
    }

    static double best(double[] a) {
        double m = a[0];
        for (double v : a) m = Math.min(m, v);
        return m;
    }

    static double mean(double[] a) {
        double s = 0;
        for (double v : a) s += v;
        return s / a.length;
    }
}
