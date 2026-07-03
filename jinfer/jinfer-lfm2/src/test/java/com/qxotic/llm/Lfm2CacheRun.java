// PromptCache validation on LFM2.5 (hybrid conv+attention+MoE): a cache-aware chat driver that
// commits blocks at every ingestion boundary, then proves (1) a cold state restores the whole
// cached conversation and continues with IDENTICAL greedy output vs an uncached replay, and
// (2) the resume skips the prefill (timed).
//   java ... com.qxotic.llm.Lfm2CacheRun [model.gguf]
package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CacheStore;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Message;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class Lfm2CacheRun {

    static int failures;

    /** The dual representation, minimal: the exact ingested token stream (fingerprints) plus the
     *  state, with a commit at every ingestion boundary so the cached chain is always contiguous. */
    static final class Session {
        final Lfm2 model;
        final PromptCache<Lfm2.State> cache;
        final Lfm2.State state;
        final List<Long> fingerprints = new ArrayList<>();

        Session(Lfm2 model, PromptCache<Lfm2.State> cache, int ctx) {
            this.model = model;
            this.cache = cache;
            this.state = model.newState(ctx, 512);
        }

        void ingest(int[] ids, boolean commit) {
            for (int[] chunk : chunks(ids, 512)) {
                int from = state.position;
                model.ingest(state, Batch.prefill(chunk));
                for (int id : chunk) fingerprints.add((long) id);
                if (commit) cache.commit(fp(), from, state.position, state);
            }
        }

        long[] fp() {
            long[] a = new long[fingerprints.size()];
            for (int i = 0; i < a.length; i++) a[i] = fingerprints.get(i);
            return a;
        }

        static List<int[]> chunks(int[] ids, int cap) {
            List<int[]> out = new ArrayList<>();
            for (int from = 0; from < ids.length; from += cap) {
                out.add(java.util.Arrays.copyOfRange(ids, from, Math.min(from + cap, ids.length)));
            }
            return out;
        }
    }

    public static void main(String[] args) throws Exception {
        Path path = Path.of(args.length > 0 ? args[0] : "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        Lfm2 model = Lfm2.loadModel(path, 4096);
        var tk = model.tokenizer();
        Lfm2TurnTemplate template = new Lfm2TurnTemplate(tk);
        Set<Integer> stops = model.stopTokens();
        int imEnd = tk.getSpecialTokens().get("<|im_end|>");

        CacheStore store = CacheStore.inMemory();
        PromptCache<Lfm2.State> cache = new PromptCache<>(new Lfm2KvCodec(model.config()), store, 1L << 30);

        // ---- conversation A: two turns, committed as it goes ----
        Session a = new Session(model, cache, 4096);
        a.ingest(flat(template.conversationStart(),
                template.encodeTurn(Message.system("You are a concise assistant.")),
                template.encodeTurn(Message.user("Name the largest planet."))), true);
        List<Integer> reply1 = decode(a, template, tk, stops, imEnd, 120);
        System.out.println("turn 1: " + tk.decode(reply1).strip());

        a.ingest(flat(template.encodeTurn(Message.user("And the smallest?"))), true);
        List<Integer> reply2 = decode(a, template, tk, stops, imEnd, 120);
        System.out.println("turn 2: " + tk.decode(reply2).strip());
        int cachedLen = a.fingerprints.size();
        System.out.println("cached conversation: " + cachedLen + " positions, " + cache.stats());

        // ---- (1) cold resume: fresh state must restore the WHOLE conversation ----
        Session b = new Session(model, cache, 4096);
        b.fingerprints.addAll(a.fingerprints);
        long t0 = System.nanoTime();
        int resumed = cache.restore(b.fp(), b.state);
        double resumeMs = (System.nanoTime() - t0) / 1e6;
        check(resumed == cachedLen, "cold resume restores all " + cachedLen + " positions (got " + resumed + ")");
        check(b.state.position == cachedLen, "state cursor at " + cachedLen);

        // ---- (2) identical continuation: cached-restored vs uncached-replayed, same greedy decode ----
        int[] turn3 = flat(template.encodeTurn(Message.user("Which of those two did you name first? One word.")));
        long t1 = System.nanoTime();
        b.ingest(turn3, true);
        List<Integer> cachedReply = decode(b, template, tk, stops, imEnd, 120);
        double cachedMs = (System.nanoTime() - t1) / 1e6;

        Session c = new Session(model, new PromptCache<>(new Lfm2KvCodec(model.config()), CacheStore.inMemory(), 1L << 30), 4096);
        long t2 = System.nanoTime();
        int[] history = new int[cachedLen];
        for (int i = 0; i < cachedLen; i++) history[i] = (int) (long) a.fingerprints.get(i);
        c.ingest(history, false);                                // full uncached re-prefill
        c.ingest(turn3, false);
        List<Integer> uncachedReply = decode(c, template, tk, stops, imEnd, 120);
        double uncachedMs = (System.nanoTime() - t2) / 1e6;

        check(cachedReply.equals(uncachedReply), "cached and uncached greedy replies identical");
        System.out.println("turn 3 (cached path): " + tk.decode(cachedReply).strip());
        System.out.printf("resume %.1fms + cached turn %.0fms  vs  uncached replay %.0fms  (%.1fx)%n",
                resumeMs, cachedMs, uncachedMs, uncachedMs / (resumeMs + cachedMs));

        // ---- (3) divergence: a different turn-2 must reuse only the shared prefix ----
        Session d = new Session(model, cache, 4096);
        d.fingerprints.addAll(a.fingerprints.subList(0, cachedLen));
        // mutate the tail: divergent final token stream
        d.fingerprints.set(cachedLen - 1, -1L);
        int partial = cache.restore(d.fp(), d.state);
        check(partial > 0 && partial < cachedLen, "divergent tail resumes a shorter prefix (" + partial + "/" + cachedLen + ")");

        System.out.println(cache.stats());
        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println("Lfm2CacheRun: all checks passed");
    }

    /** Greedy-decode a reply, ingesting each step; close the turn; commit the whole reply span
     *  (conv checkpoint is current at the boundary). Returns the reply token ids. */
    static List<Integer> decode(Session s, Lfm2TurnTemplate template, com.qxotic.jinfer.LFMTokenizer tk,
                                Set<Integer> stops, int imEnd, int maxTokens) {
        s.ingest(flat(template.generationPrompt(true)), true);
        int replyFrom = s.state.position;
        List<Integer> reply = new ArrayList<>();
        int tok = LLM.argmax(s.model.logits(s.state), s.model.config().vocabularySize());
        for (int n = 0; n < maxTokens && !stops.contains(tok); n++) {
            reply.add(tok);
            s.model.ingest(s.state, Batch.step(tok));
            s.fingerprints.add((long) tok);
            tok = LLM.argmax(s.model.logits(s.state), s.model.config().vocabularySize());
        }
        // close the assistant turn exactly as the template frames it, then commit [replyFrom, here)
        List<Integer> close = new ArrayList<>(List.of(imEnd));
        close.addAll(tk.encode("\n"));
        int[] closeIds = close.stream().mapToInt(Integer::intValue).toArray();
        s.model.ingest(s.state, Batch.prefill(closeIds));
        for (int id : closeIds) s.fingerprints.add((long) id);
        s.cache.commit(s.fp(), replyFrom, s.state.position, s.state);
        return reply;
    }

    static int[] flat(List<Batch>... groups) {
        List<Integer> ids = new ArrayList<>();
        for (List<Batch> g : groups) {
            for (Batch b : g) for (int id : ((Batch.Input.Tokens) b.input()).ids()) ids.add(id);
        }
        return ids.stream().mapToInt(Integer::intValue).toArray();
    }

    static void check(boolean ok, String what) {
        if (ok) {
            System.out.println("ok:   " + what);
        } else {
            failures++;
            System.out.println("FAIL: " + what);
        }
    }
}
