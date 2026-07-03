// The PromptCache validation scenario, shared by every model's CacheRun: (1) a cold state resumes
// the whole cached conversation, (2) cached and uncached greedy continuations are token-identical
// (for windowed models also with a history long enough that the SWA rings wrap - where ring-slot
// restore bugs show), (3) a divergent tail resumes only the shared prefix. Prints the
// resume-vs-replay benchmark table.
package com.qxotic.jinfer.testkit;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CacheStore;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Message;

import java.util.List;

public final class CacheScenario<S extends RuntimeState> {

    /** Per-model knobs: the optional system prompt, the optional long-history story (null = skip
     *  the wrapped-window case), the minimum long-history length to assert, decode budget, and
     *  whether replies are logged tail-only (Harmony analysis channels). */
    public record Config(String systemPrompt, String longStory, String longProbe, int minLongLen,
                         int maxTokens, boolean logTail) {}

    private final Harness<S> h;
    private final Config cfg;
    private final PromptCache<S> cache;
    private final long budget;

    public CacheScenario(Harness<S> h, Config cfg) {
        this.h = h;
        this.cfg = cfg;
        this.budget = Long.getLong("jinfer.promptCacheMB", 8192L) << 20;
        this.cache = new PromptCache<>(h.codec, CacheStore.inMemory(), budget, h.seed);
    }

    public void run(String runName) {
        // ---- short conversation: two turns, committed as it goes ----
        CachedSession<S> a = CachedSession.resume(h.model, cache, h.newState(), new long[0]);
        a.ingest(h.template.conversationStart());
        if (cfg.systemPrompt() != null) a.ingest(h.template.encodeTurn(Message.system(cfg.systemPrompt())));
        a.ingest(h.template.encodeTurn(Message.user("Name the largest planet. Answer briefly.")));
        System.out.println("turn 1: " + log(h.decode(a, cfg.maxTokens())));
        a.ingest(h.template.encodeTurn(Message.user("And the smallest? Answer briefly.")));
        System.out.println("turn 2: " + log(h.decode(a, cfg.maxTokens())));
        long[] shortHist = a.fingerprints();
        double[] shortBench = validate("short (" + shortHist.length + " pos)", shortHist,
                Message.user("Which of those two did you name first? One word."));

        // ---- long conversation: history far beyond the sliding window (rings wrap) ----
        long[] longHist = null;
        double[] longBench = null;
        if (cfg.longStory() != null) {
            CachedSession<S> b = CachedSession.resume(h.model, cache, h.newState(), new long[0]);
            b.ingest(h.template.conversationStart());
            b.ingest(h.template.encodeTurn(Message.user(cfg.longStory())));
            System.out.println("long turn 1: " + log(h.decode(b, cfg.maxTokens())));
            longHist = b.fingerprints();
            h.check(longHist.length > cfg.minLongLen(),
                    "long history exceeds the window (" + longHist.length + " > " + cfg.minLongLen() + ")");
            longBench = validate("long (" + longHist.length + " pos, wrapped rings)", longHist,
                    Message.user(cfg.longProbe()));
        }

        // ---- divergent tail resumes only the shared prefix ----
        long[] mutated = shortHist.clone();
        mutated[mutated.length - 1] = -1;
        CachedSession<S> d = CachedSession.resume(h.model, cache, h.newState(), mutated);
        h.check(d.position() > 0 && d.position() < shortHist.length,
                "divergent tail resumes a shorter prefix (" + d.position() + "/" + shortHist.length + ")");

        // ---- benchmark table ----
        System.out.println("\n=== benchmark: resume vs uncached replay ===");
        System.out.printf("%-38s %12s %14s %10s%n", "history", "resume (ms)", "replay (ms)", "speedup");
        System.out.printf("%-38s %12.1f %14.0f %9.0fx%n", "short (" + shortHist.length + " pos)",
                shortBench[0], shortBench[1], shortBench[1] / shortBench[0]);
        if (longBench != null) {
            System.out.printf("%-38s %12.1f %14.0f %9.0fx%n", "long (" + longHist.length + " pos)",
                    longBench[0], longBench[1], longBench[1] / longBench[0]);
        }
        System.out.printf("decode: %.1f tok/s%n", (longBench != null ? longBench : shortBench)[2]);
        System.out.println(cache.stats());
        h.finish(runName);
    }

    /** Cold-resume {@code history}, append {@code probe}, and check the greedy continuation is
     *  token-identical to a fully uncached replay. Returns {resumeMs, replayMs, tokPerSec}. */
    private double[] validate(String name, long[] history, Message probe) {
        long t0 = System.nanoTime();
        CachedSession<S> cached = CachedSession.resume(h.model, cache, h.newState(), history);
        double resumeMs = (System.nanoTime() - t0) / 1e6;
        h.check(cached.position() == history.length,
                name + ": cold resume restores all " + history.length + " positions (got " + cached.position() + ")");
        List<Batch> turn = h.template.encodeTurn(probe);
        cached.ingest(turn);
        long t1 = System.nanoTime();
        String cachedReply = h.decode(cached, cfg.maxTokens());
        double decodeSec = (System.nanoTime() - t1) / 1e9;

        PromptCache<S> scratch = new PromptCache<>(h.codec, CacheStore.inMemory(), budget, h.seed);
        CachedSession<S> plain = CachedSession.resume(h.model, scratch, h.newState(), new long[0]);
        long t2 = System.nanoTime();
        plain.ingest(List.of(Batch.prefill(Harness.toInts(history))));
        double replayMs = (System.nanoTime() - t2) / 1e6;
        plain.ingest(turn);
        String plainReply = h.decode(plain, cfg.maxTokens());

        h.check(cachedReply.equals(plainReply), name + ": cached and uncached greedy replies identical");
        System.out.println(name + " reply: " + log(cachedReply));
        return new double[]{resumeMs, replayMs, cachedReply.isEmpty() ? 0 : h.lastReplyTokens / decodeSec};
    }

    private String log(String reply) {
        return cfg.logTail() ? Harness.tail(reply) : reply.strip();
    }
}
