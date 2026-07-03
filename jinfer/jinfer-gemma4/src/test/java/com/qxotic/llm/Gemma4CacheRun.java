// PromptCache validation on Gemma 4 E2B (interleaved SWA rings + full attention + shared KV tail):
// CachedSession + Gemma4TurnTemplate + Gemma4KvCodec. Proves (1) cold resume restores the whole
// conversation, (2) cached and uncached greedy continuations are token-identical - including with
// a history LONGER than the 512-token sliding window, so the SWA ring checkpoints actually wrap
// (the case where ring-slot restore bugs show), (3) a divergent tail resumes only the shared
// prefix. Prints a resume-vs-replay benchmark table.
//   java ... com.qxotic.llm.Gemma4CacheRun [model.gguf]
package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CacheStore;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Message;

import java.nio.file.Path;
import java.util.List;
import java.util.Set;

public final class Gemma4CacheRun {

    static int failures;
    static Gemma4 model;
    static Gemma4TurnTemplate template;
    static Set<Integer> stops;
    static PromptCache<Gemma4.State> cache;
    static long budget;

    public static void main(String[] args) throws Exception {
        Path path = Path.of(args.length > 0 ? args[0] : "/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf");
        model = Gemma4.loadModel(path, 8192);
        template = new Gemma4TurnTemplate(model.tokenizer());
        stops = model.stopTokens();
        budget = Long.getLong("jinfer.promptCacheMB", 8192L) << 20;
        cache = new PromptCache<>(new Gemma4KvCodec(model.config()), CacheStore.inMemory(), budget);

        // ================= short conversation (window does NOT wrap) =================
        CachedSession<Gemma4.State> a = CachedSession.resume(model, cache, model.newState(8192, 512), new long[0]);
        a.ingest(template.conversationStart());
        a.ingest(template.encodeTurn(Message.user("Name the largest planet. Answer briefly.")));
        System.out.println("turn 1: " + decode(a, 60).strip());
        a.ingest(template.encodeTurn(Message.user("And the smallest? Answer briefly.")));
        System.out.println("turn 2: " + decode(a, 60).strip());
        long[] shortHist = a.fingerprints();
        double[] shortBench = validate("short (" + shortHist.length + " pos, un-wrapped rings)", shortHist,
                Message.user("Which of those two did you name first? One word."));

        // ================= long conversation (history >> 512: SWA rings WRAP) =================
        StringBuilder story = new StringBuilder("Remember this codeword: PELICAN. Now read this story. ");
        for (int i = 0; i < 60; i++) {
            story.append("Chapter ").append(i).append(": the river wound through the valley, past mills and ")
                 .append("orchards, while travelers traded stories of distant mountain passes and the long winter. ");
        }
        story.append("After the story, tell me in one short sentence what the story is about.");
        CachedSession<Gemma4.State> b = CachedSession.resume(model, cache, model.newState(8192, 512), new long[0]);
        b.ingest(template.conversationStart());
        b.ingest(template.encodeTurn(Message.user(story.toString())));
        System.out.println("long turn 1: " + decode(b, 60).strip());
        long[] longHist = b.fingerprints();
        check(longHist.length > 1024, "long history exceeds the sliding window (" + longHist.length + " > 1024)");
        double[] longBench = validate("long (" + longHist.length + " pos, wrapped rings)", longHist,
                Message.user("What was the codeword at the start? One word."));

        // ================= divergent tail =================
        long[] mutated = shortHist.clone();
        mutated[mutated.length - 1] = -1;
        CachedSession<Gemma4.State> d = CachedSession.resume(model, cache, model.newState(8192, 512), mutated);
        check(d.position() > 0 && d.position() < shortHist.length,
                "divergent tail resumes a shorter prefix (" + d.position() + "/" + shortHist.length + ")");

        // ================= benchmark table =================
        System.out.println("\n=== benchmark: resume vs uncached replay ===");
        System.out.printf("%-38s %12s %14s %10s%n", "history", "resume (ms)", "replay (ms)", "speedup");
        System.out.printf("%-38s %12.1f %14.0f %9.0fx%n", "short (" + shortHist.length + " pos)", shortBench[0], shortBench[1], shortBench[1] / shortBench[0]);
        System.out.printf("%-38s %12.1f %14.0f %9.0fx%n", "long (" + longHist.length + " pos)", longBench[0], longBench[1], longBench[1] / longBench[0]);
        System.out.printf("decode: %.1f tok/s%n", longBench[2]);

        System.out.println(cache.stats());
        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println("Gemma4CacheRun: all checks passed");
    }

    /** Cold-resume {@code history}, append {@code probe}, and check the greedy continuation is
     *  token-identical to a fully uncached replay. Returns {resumeMs, replayMs, tokPerSec}. */
    static double[] validate(String name, long[] history, Message probe) {
        long t0 = System.nanoTime();
        CachedSession<Gemma4.State> cached = CachedSession.resume(model, cache, model.newState(8192, 512), history);
        double resumeMs = (System.nanoTime() - t0) / 1e6;
        check(cached.position() == history.length,
                name + ": cold resume restores all " + history.length + " positions (got " + cached.position() + ")");
        List<Batch> turn = template.encodeTurn(probe);
        cached.ingest(turn);
        long t1 = System.nanoTime();
        String cachedReply = decode(cached, 60);
        double decodeSec = (System.nanoTime() - t1) / 1e9;

        PromptCache<Gemma4.State> scratch = new PromptCache<>(new Gemma4KvCodec(model.config()), CacheStore.inMemory(), budget);
        CachedSession<Gemma4.State> plain = CachedSession.resume(model, scratch, model.newState(8192, 512), new long[0]);
        long t2 = System.nanoTime();
        int[] ids = new int[history.length];
        for (int i = 0; i < ids.length; i++) ids[i] = (int) history[i];
        plain.ingest(List.of(Batch.prefill(ids)));
        double replayMs = (System.nanoTime() - t2) / 1e6;
        plain.ingest(turn);
        String plainReply = decode(plain, 60);

        check(cachedReply.equals(plainReply), name + ": cached and uncached greedy replies identical");
        System.out.println(name + " reply: " + cachedReply.strip());
        return new double[]{resumeMs, replayMs, cachedReply.isEmpty() ? 0 : countTokens(cachedReply) / decodeSec};
    }

    static int lastReplyTokens;

    static int countTokens(String reply) {
        return lastReplyTokens;
    }

    /** Open the assistant turn, greedy-decode (each step a single-token block), close the turn. */
    static String decode(CachedSession<Gemma4.State> s, int maxTokens) {
        s.ingest(template.generationPrompt(true));
        StringBuilder out = new StringBuilder();
        int tok = LLM.argmax(model.logits(s.state()), model.config().vocabularySize());
        int n = 0;
        for (; n < maxTokens && !stops.contains(tok); n++) {
            out.append(model.tokenizer().decode(tok));
            s.step(tok);
            tok = LLM.argmax(model.logits(s.state()), model.config().vocabularySize());
        }
        lastReplyTokens = Math.max(n, 1);
        s.ingest(template.closeTurn());
        return out.toString();
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
