// TTFT benchmark for the prompt cache: time-to-first-token of a follow-up request over a long
// cached history, vs a cold full prefill - the numbers compared against llama.cpp's caching.
//   warm:           java ... TtftBench warm <model.gguf> [reps]
//   sealed-compile: java ... TtftBench sealed-compile <model.gguf> <out.jkv>
//   sealed-serve:   java ... TtftBench sealed-serve <model.gguf> <in.jkv> (fresh JVM: cold-from-disk TTFT)
//   dump-text:      java ... TtftBench dump-text <dir>   (story + question, shared with llama.cpp)
package com.qxotic.jinfer;

import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.KvCodec;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.cache.SealedPrompt;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.llm.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class TtftBench {

    // ~2500-token history: story turn + a generated assistant reply. Same text shared with llama.cpp.
    static String story() {
        StringBuilder s = new StringBuilder("Remember this codeword: PELICAN. Summarize the following notes.\n");
        for (int i = 0; i < 90; i++) {
            s.append("Entry ").append(i).append(": the expedition logged river depth, canopy density, ")
                    .append("and soil acidity at station ").append(i)
                    .append("; readings were nominal and the weather held clear through the afternoon.\n");
        }
        return s.toString();
    }

    static final String QUESTION = "What was the codeword? One word only.";
    static final int REPLY_BUDGET = 40;

    public static void main(String[] args) throws Exception {
        switch (args[0]) {
            case "dump-text" -> {
                Files.writeString(Path.of(args[1], "story.txt"), story());
                Files.writeString(Path.of(args[1], "question.txt"), QUESTION);
                System.out.println("wrote story.txt + question.txt");
            }
            case "warm" -> warm(load(args[1]), Path.of(args[1]), args.length > 2 ? Integer.parseInt(args[2]) : 5);
            case "sealed-compile" -> sealedCompile(load(args[1]), Path.of(args[1]), Path.of(args[2]));
            case "sealed-serve" -> sealedServe(load(args[1]), Path.of(args[1]), Path.of(args[2]));
            default -> throw new IllegalArgumentException(args[0]);
        }
    }

    static LanguageModel<?, ?, ?> load(String path) throws Exception {
        long t0 = System.nanoTime();
        String lower = path.toLowerCase();
        LanguageModel<?, ?, ?> m = lower.contains("gemma") ? Gemma4.loadModel(Path.of(path), 4096)
                : Lfm2.loadModel(Path.of(path), 4096);
        System.err.printf("model load: %.0f ms%n", (System.nanoTime() - t0) / 1e6);
        return m;
    }

    /** Builds the cached history (story turn + generated reply), then benches: cold full-prefill
     *  TTFT vs warm resume TTFT for the follow-up question. */
    static <S extends RuntimeState> void warm(LanguageModel<?, ?, S> model, Path gguf, int reps) {
        TurnTemplate tpl = model.turnTemplate().orElseThrow();
        KvCodec<S> codec = model.kvCodec().orElseThrow();
        PromptCache<S> cache = new PromptCache<>(codec, CacheStore.inMemory(), 8L << 30, PromptCache.modelSeed(gguf));

        // history: [start][user: story][genPrompt] -> reply -> closeTurn, all committed
        CachedSession<S> a = CachedSession.resume(model, cache, model.newState(4096, 512), new long[0]);
        a.ingest(concat(tpl.conversationStart(), tpl.encodeTurn(Message.user(story()))));
        String reply = decode(model, a, tpl, REPLY_BUDGET);
        long[] history = a.fingerprints();
        System.err.println("history: " + history.length + " positions; reply: " + reply.strip());

        List<Batch> followUp = concat(tpl.encodeTurn(Message.user(QUESTION)), tpl.generationPrompt(true));
        int[] historyIds = new int[history.length];
        for (int i = 0; i < history.length; i++) historyIds[i] = (int) history[i];

        // tier 1: live pooled sessions, pre-resumed OUTSIDE the timing (a SessionPool holds them
        // resident between requests) - the timed leg is pure append-only delta ingest + first token.
        List<CachedSession<S>> live = new ArrayList<>();
        for (int r = 0; r < reps; r++) {
            CachedSession<S> s = CachedSession.resume(model, cache, model.newState(4096, 512), history);
            if (s.position() != history.length) throw new IllegalStateException("live resume " + s.position());
            live.add(s);
        }

        double[] cold = new double[reps], warmMs = new double[reps], tier1 = new double[reps];
        String firstTok = "";
        for (int r = 0; r < reps; r++) {
            // cold: full prefill of history + follow-up
            S s1 = model.newState(4096, 512);
            long t0 = System.nanoTime();
            for (Batch b : Batch.prepare(concat(List.of(Batch.prefill(historyIds)), followUp), 512)) model.ingest(s1, b);
            int tok1 = model.logits(s1).argmax();
            cold[r] = (System.nanoTime() - t0) / 1e6;

            // warm: cache resume + follow-up only
            S s2 = model.newState(4096, 512);
            long t1 = System.nanoTime();
            CachedSession<S> b2 = CachedSession.resume(model, cache, s2, history);
            if (b2.position() != history.length) throw new IllegalStateException("resume " + b2.position());
            b2.ingest(followUp);
            int tok2 = model.logits(s2).argmax();
            warmMs[r] = (System.nanoTime() - t1) / 1e6;
            if (tok1 != tok2) System.err.println("NOTE: first tokens differ (cold " + tok1 + " vs warm " + tok2 + ")");
            firstTok = model.tokenizer().decode(tok2);

            // tier 1: append-only on the live session (llama.cpp in-place slot equivalent)
            CachedSession<S> p = live.get(r);
            long t2 = System.nanoTime();
            p.ingest(followUp);
            int tok3 = model.logits(p.state()).argmax();
            tier1[r] = (System.nanoTime() - t2) / 1e6;
            if (tok3 != tok2) System.err.println("NOTE: tier-1 first token differs (" + tok3 + " vs " + tok2 + ")");
        }
        int delta = Batch.tokenIds(followUp).length;
        System.out.printf("history=%d deltaTokens=%d firstTok=%s%n", history.length, delta, firstTok.strip());
        System.out.printf("cold        TTFT: best %.1f ms  mean %.1f ms%n", best(cold), mean(cold));
        System.out.printf("tier2 warm  TTFT: best %.1f ms  mean %.1f ms  (block restore + delta)%n", best(warmMs), mean(warmMs));
        System.out.printf("tier1 pool  TTFT: best %.1f ms  mean %.1f ms  (append-only on the live session)%n", best(tier1), mean(tier1));
    }

    /** Seals the STATIC prefix (conversationStart + the story turn - fully deterministic, no
     *  generated tokens) so the serve side can rebuild the exact fingerprints from text. */
    static <S extends RuntimeState> void sealedCompile(LanguageModel<?, ?, S> model, Path gguf, Path out) throws Exception {
        TurnTemplate tpl = model.turnTemplate().orElseThrow();
        KvCodec<S> codec = model.kvCodec().orElseThrow();
        List<Batch> prefix = concat(tpl.conversationStart(), tpl.encodeTurn(Message.user(story())));
        int[] ids = Batch.tokenIds(prefix);
        long[] fp = new long[ids.length];
        for (int i = 0; i < ids.length; i++) fp[i] = ids[i];
        S state = model.newState(4096, 512);
        for (Batch b : Batch.prepare(prefix, 512)) model.ingest(state, b);
        SealedPrompt.compile(out, "ttft-bench", codec, state, fp, PromptCache.modelSeed(gguf));
        System.out.printf("sealed %d positions -> %s (%.1f MB)%n", fp.length, out, Files.size(out) / 1e6);
    }

    /** Fresh-JVM serve: open + tryRestore + follow-up prefill + first token. */
    static <S extends RuntimeState> void sealedServe(LanguageModel<?, ?, S> model, Path gguf, Path file) throws Exception {
        TurnTemplate tpl = model.turnTemplate().orElseThrow();
        KvCodec<S> codec = model.kvCodec().orElseThrow();
        List<Batch> followUp = concat(tpl.encodeTurn(Message.user(QUESTION)), tpl.generationPrompt(true));

        // JIT warmup on a throwaway state (a running server / native image has warm code; without
        // this the first forward pays ~2.5s of compilation, which is JVM startup, not cache cost)
        S w = model.newState(4096, 512);
        int[] warmIds = new int[256];
        java.util.Arrays.fill(warmIds, 5);
        for (int i = 0; i < 2; i++) {
            S ws = model.newState(4096, 512);
            model.ingest(ws, Batch.prefill(warmIds));
            model.logits(ws).argmax();
        }

        long t0 = System.nanoTime();
        SealedPrompt sealed = SealedPrompt.open(file, PromptCache.modelSeed(gguf));
        double openMs = (System.nanoTime() - t0) / 1e6;

        // request fingerprints: re-derived from the same static text (deterministic template)
        S state = model.newState(4096, 512);
        long t1 = System.nanoTime();
        int[] ids = Batch.tokenIds(concat(tpl.conversationStart(), tpl.encodeTurn(Message.user(story()))));
        long[] fp = new long[ids.length];
        for (int i = 0; i < ids.length; i++) fp[i] = ids[i];
        int restored = sealed.tryRestore(state, codec, fp);
        if (restored == 0) throw new IllegalStateException("sealed restore missed");
        for (Batch b : Batch.prepare(followUp, 512)) model.ingest(state, b);
        int tok = model.logits(state).argmax();
        double ttft = (System.nanoTime() - t1) / 1e6;
        System.out.printf("sealed open: %.1f ms;  first-serve TTFT: %.1f ms;  restored=%d  firstTok=%s%n",
                openMs, ttft, restored, model.tokenizer().decode(tok).strip());
        // steady state (long-running server): re-serve in-process
        for (int r = 0; r < 3; r++) {
            S s2 = model.newState(4096, 512);
            long t2 = System.nanoTime();
            SealedPrompt sp2 = SealedPrompt.open(file, PromptCache.modelSeed(gguf));
            int n2 = sp2.tryRestore(s2, codec, fp);
            for (Batch b : Batch.prepare(followUp, 512)) model.ingest(s2, b);
            model.logits(s2).argmax();
            System.out.printf("steady serve TTFT (open+restore+followUp+token): %.1f ms (restored %d)%n",
                    (System.nanoTime() - t2) / 1e6, n2);
        }
    }

    static <S extends RuntimeState> String decode(LanguageModel<?, ?, S> model, CachedSession<S> s, TurnTemplate tpl, int max) {
        s.ingest(tpl.generationPrompt(true));
        Set<Integer> stops = model.stopTokens();
        StringBuilder out = new StringBuilder();
        int tok = model.logits(s.state()).argmax();
        for (int n = 0; n < max && !stops.contains(tok); n++) {
            out.append(model.tokenizer().decode(tok));
            s.step(tok);
            tok = model.logits(s.state()).argmax();
        }
        s.ingest(tpl.closeTurn());
        return out.toString();
    }

    static List<Batch> concat(List<Batch>... groups) {
        List<Batch> out = new ArrayList<>();
        for (List<Batch> g : groups) out.addAll(g);
        return out;
    }

    static double best(double[] a) { double m = a[0]; for (double v : a) m = Math.min(m, v); return m; }
    static double mean(double[] a) { double s = 0; for (double v : a) s += v; return s / a.length; }
}
