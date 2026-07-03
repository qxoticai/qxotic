// PromptCache validation + benchmark on Llama 1B (uniform full attention - the degenerate codec):
// CachedSession + LlamaTurnTemplate + LlamaKvCodec. Proves (1) a cold state resumes the whole
// cached conversation, (2) cached and uncached greedy continuations are token-identical, (3) a
// divergent tail resumes only the shared prefix; then benchmarks resume vs uncached replay for a
// short and a long (~1500+ token) history.
//   java ... com.qxotic.llm.LlamaCacheRun [model.gguf]
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

public final class LlamaCacheRun {

    static int failures;
    static Llama model;
    static LlamaTurnTemplate template;
    static Set<Integer> stops;

    public static void main(String[] args) throws Exception {
        Path path = Path.of(args.length > 0 ? args[0] : "/home/mukel/Desktop/playground/models/unsloth/Llama-3.2-1B-Instruct-Q8_0.gguf");
        model = Llama.loadModel(path, 8192);
        template = new LlamaTurnTemplate(model.tokenizer());
        stops = model.stopTokens();
        long budget = Long.getLong("jinfer.promptCacheMB", 2048L) << 20;
        PromptCache<Llama.State> cache = new PromptCache<>(new LlamaKvCodec(model.config()), CacheStore.inMemory(), budget, PromptCache.modelSeed(path));

        // ---- conversation A: two turns, committed as it goes ----
        CachedSession<Llama.State> a = CachedSession.resume(model, cache, model.newState(8192, 512), new long[0]);
        a.ingest(concat(template.conversationStart(),
                template.encodeTurn(Message.system("You are a concise assistant.")),
                template.encodeTurn(Message.user("Name the largest planet. One word."))));
        System.out.println("turn 1: " + decode(a, 60).strip());
        a.ingest(template.encodeTurn(Message.user("And the smallest? One word.")));
        System.out.println("turn 2: " + decode(a, 60).strip());
        long[] history = a.fingerprints();
        System.out.println("cached conversation: " + history.length + " positions, " + cache.stats());

        // ---- (1) cold resume restores the whole conversation ----
        long t0 = System.nanoTime();
        CachedSession<Llama.State> b = CachedSession.resume(model, cache, model.newState(8192, 512), history);
        double resumeShortMs = (System.nanoTime() - t0) / 1e6;
        check(b.position() == history.length, "cold resume restores all " + history.length + " positions (got " + b.position() + ")");

        // ---- (2) identical continuation: cached vs uncached, same greedy decode ----
        List<Batch> turn3 = template.encodeTurn(Message.user("Which of those two did you name first? One word."));
        b.ingest(turn3);
        String cachedReply = decode(b, 60);

        PromptCache<Llama.State> scratch = new PromptCache<>(new LlamaKvCodec(model.config()), CacheStore.inMemory(), budget, PromptCache.modelSeed(path));
        CachedSession<Llama.State> c = CachedSession.resume(model, scratch, model.newState(8192, 512), new long[0]);
        long t2 = System.nanoTime();
        c.ingest(List.of(Batch.prefill(toInts(history))));
        double replayShortMs = (System.nanoTime() - t2) / 1e6;
        c.ingest(turn3);
        String uncachedReply = decode(c, 60);

        check(cachedReply.equals(uncachedReply), "cached and uncached greedy replies identical");
        System.out.println("turn 3 (cached path): " + cachedReply.strip());

        // ---- (3) divergent tail resumes only the shared prefix ----
        long[] mutated = history.clone();
        mutated[mutated.length - 1] = -1;
        CachedSession<Llama.State> d = CachedSession.resume(model, cache, model.newState(8192, 512), mutated);
        check(d.position() > 0 && d.position() < history.length,
                "divergent tail resumes a shorter prefix (" + d.position() + "/" + history.length + ")");

        // ---- long-history benchmark (~1500+ tokens) ----
        StringBuilder story = new StringBuilder("Summarize the following notes.\n");
        for (int i = 0; i < 90; i++) {
            story.append("Entry ").append(i).append(": the expedition logged river depth, canopy density, ")
                 .append("and soil acidity at station ").append(i)
                 .append("; readings were nominal and the weather held clear through the afternoon.\n");
        }
        CachedSession<Llama.State> e = CachedSession.resume(model, cache, model.newState(8192, 512), new long[0]);
        long t3 = System.nanoTime();
        e.ingest(concat(template.conversationStart(),
                template.encodeTurn(Message.system("You are a concise assistant.")),
                template.encodeTurn(Message.user(story.toString()))));
        double prefillLongMs = (System.nanoTime() - t3) / 1e6;
        String longReply = decode(e, 80);
        check(!longReply.isBlank(), "long-history reply non-empty");
        long[] longHistory = e.fingerprints();

        long t4 = System.nanoTime();
        CachedSession<Llama.State> f = CachedSession.resume(model, cache, model.newState(8192, 512), longHistory);
        double resumeLongMs = (System.nanoTime() - t4) / 1e6;
        check(f.position() == longHistory.length, "long history fully resumed (" + f.position() + "/" + longHistory.length + ")");

        // decode throughput on the resumed long-history state
        f.ingest(template.encodeTurn(Message.user("How many entries were there? One number.")));
        f.ingest(template.generationPrompt(true));
        int n = 0;
        long t5 = System.nanoTime();
        int tok = LLM.argmax(model.logits(f.state()), model.config().vocabularySize());
        for (; n < 60 && !stops.contains(tok); n++) {
            f.step(tok);
            tok = LLM.argmax(model.logits(f.state()), model.config().vocabularySize());
        }
        double tokPerSec = n / ((System.nanoTime() - t5) / 1e9);

        System.out.println();
        System.out.println("=== benchmark (Llama 1B Q8_0) ===");
        System.out.printf("short history (%4d tok): resume %6.1f ms   vs uncached prefill %8.1f ms%n",
                history.length, resumeShortMs, replayShortMs);
        System.out.printf("long  history (%4d tok): resume %6.1f ms   vs uncached prefill %8.1f ms%n",
                longHistory.length, resumeLongMs, prefillLongMs);
        System.out.printf("decode: %.1f tok/s (%d tokens)%n", tokPerSec, n);
        System.out.println(cache.stats());

        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println("LlamaCacheRun: all checks passed");
    }

    /** Open the assistant turn, greedy-decode (each step a single-token block), close the turn. */
    static String decode(CachedSession<Llama.State> s, int maxTokens) {
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

    static int[] toInts(long[] fp) {
        int[] ids = new int[fp.length];
        for (int i = 0; i < ids.length; i++) ids[i] = (int) fp[i];
        return ids;
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
