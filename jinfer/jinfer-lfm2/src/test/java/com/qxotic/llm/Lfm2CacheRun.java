// PromptCache validation on LFM2.5 (hybrid conv+attention+MoE), on the pristine surface:
// CachedSession (fingerprints + boundary commits) + TurnTemplate framing + Lfm2KvCodec.
// Proves (1) a cold state resumes the whole cached conversation, (2) cached and uncached greedy
// continuations are token-identical, (3) a divergent tail resumes only the shared prefix.
//   java ... com.qxotic.llm.Lfm2CacheRun [model.gguf]
package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CacheStore;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Message;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class Lfm2CacheRun {

    static int failures;
    static Lfm2 model;
    static Lfm2TurnTemplate template;
    static Set<Integer> stops;

    public static void main(String[] args) throws Exception {
        Path path = Path.of(args.length > 0 ? args[0] : "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        model = Lfm2.loadModel(path, 4096);
        template = new Lfm2TurnTemplate(model.tokenizer());
        stops = model.stopTokens();
        long budget = Long.getLong("jinfer.promptCacheMB", 1024L) << 20;
        PromptCache<Lfm2.State> cache = new PromptCache<>(new Lfm2KvCodec(model.config()), CacheStore.inMemory(), budget, PromptCache.modelSeed(path));

        // ---- conversation A: two turns, committed as it goes ----
        CachedSession<Lfm2.State> a = CachedSession.resume(model, cache, model.newState(4096, 512), new long[0]);
        a.ingest(concat(template.conversationStart(),
                template.encodeTurn(Message.system("You are a concise assistant.")),
                template.encodeTurn(Message.user("Name the largest planet."))));
        System.out.println("turn 1: " + decode(a, 120).strip());
        a.ingest(template.encodeTurn(Message.user("And the smallest?")));
        System.out.println("turn 2: " + decode(a, 120).strip());
        long[] history = a.fingerprints();
        System.out.println("cached conversation: " + history.length + " positions, " + cache.stats());

        // ---- (1) cold resume restores the whole conversation ----
        long t0 = System.nanoTime();
        CachedSession<Lfm2.State> b = CachedSession.resume(model, cache, model.newState(4096, 512), history);
        double resumeMs = (System.nanoTime() - t0) / 1e6;
        check(b.position() == history.length, "cold resume restores all " + history.length + " positions (got " + b.position() + ")");

        // ---- (2) identical continuation: cached vs uncached, same greedy decode ----
        List<Batch> turn3 = template.encodeTurn(Message.user("Which of those two did you name first? One word."));
        long t1 = System.nanoTime();
        b.ingest(turn3);
        String cachedReply = decode(b, 120);
        double cachedMs = (System.nanoTime() - t1) / 1e6;

        PromptCache<Lfm2.State> scratch = new PromptCache<>(new Lfm2KvCodec(model.config()), CacheStore.inMemory(), budget, PromptCache.modelSeed(path));
        CachedSession<Lfm2.State> c = CachedSession.resume(model, scratch, model.newState(4096, 512), new long[0]);
        long t2 = System.nanoTime();
        int[] ids = new int[history.length];
        for (int i = 0; i < ids.length; i++) ids[i] = (int) history[i];
        c.ingest(List.of(Batch.prefill(ids)));
        c.ingest(turn3);
        String uncachedReply = decode(c, 120);
        double uncachedMs = (System.nanoTime() - t2) / 1e6;

        check(cachedReply.equals(uncachedReply), "cached and uncached greedy replies identical");
        System.out.println("turn 3 (cached path): " + cachedReply.strip());
        System.out.printf("resume %.1fms + cached turn %.0fms  vs  uncached replay %.0fms%n", resumeMs, cachedMs, uncachedMs);

        // ---- (3) divergent tail resumes only the shared prefix ----
        long[] mutated = history.clone();
        mutated[mutated.length - 1] = -1;
        CachedSession<Lfm2.State> d = CachedSession.resume(model, cache, model.newState(4096, 512), mutated);
        check(d.position() > 0 && d.position() < history.length,
                "divergent tail resumes a shorter prefix (" + d.position() + "/" + history.length + ")");

        System.out.println(cache.stats());
        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println("Lfm2CacheRun: all checks passed");
    }

    /** Open the assistant turn, greedy-decode (each step a single-token block), close the turn. */
    static String decode(CachedSession<Lfm2.State> s, int maxTokens) {
        s.ingest(template.generationPrompt(true));
        StringBuilder out = new StringBuilder();
        int tok = LLM.argmax(model.logits(s.state()), model.config().vocabularySize());
        for (int n = 0; n < maxTokens && !stops.contains(tok); n++) {
            out.append(model.tokenizer().decode(tok));
            s.step(tok);
            tok = LLM.argmax(model.logits(s.state()), model.config().vocabularySize());
        }
        s.ingest(template.closeTurn());
        return out.toString();
    }

    @SafeVarargs
    static List<Batch> concat(List<Batch>... groups) {
        List<Batch> out = new ArrayList<>();
        for (List<Batch> g : groups) out.addAll(g);
        return out;
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
