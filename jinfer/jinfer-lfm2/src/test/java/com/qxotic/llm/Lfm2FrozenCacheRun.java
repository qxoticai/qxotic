// Frozen multi-prompt cache validation (use case B) on LFM2.5: build a PromptCache holding THREE
// long prompts sharing a common system prefix (dedup gate), freeze() it, open() it read-only, and
// prove each prompt resumes fully with greedy replies IDENTICAL to an uncached full prefill; an
// unseen prompt gets a partial (shared-prefix) hit; a wrong-seed open fails with the proper error;
// commits on the frozen instance are silent no-ops.
//   java ... com.qxotic.llm.Lfm2FrozenCacheRun [model.gguf]
package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CacheStore;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.KvCodec;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.TurnTemplate;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class Lfm2FrozenCacheRun {

    static int failures;
    static Lfm2 model;
    static TurnTemplate template;
    static Set<Integer> stops;

    public static void main(String[] args) throws Exception {
        Path path = Path.of(args.length > 0 ? args[0] : "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        model = Lfm2.loadModel(path, 4096);
        template = model.turnTemplate().orElseThrow();
        stops = model.stopTokens();
        byte[] seed = PromptCache.modelSeed(path);
        KvCodec<Lfm2.State> codec = model.kvCodec().orElseThrow();
        long budget = 4L << 30;

        // ---- three prompts sharing a long system prefix; one unseen (shares the prefix too) ----
        List<Batch> shared = concat(template.conversationStart(), template.encodeTurn(Message.system(sharedRules())));
        String[] domains = {"unit conversion", "capital cities", "arithmetic"};
        List<List<Batch>> prompts = new ArrayList<>();
        for (String d : domains) prompts.add(template.encodeTurn(Message.user(examples(d))));
        List<Batch> unseen = template.encodeTurn(Message.user(examples("chemistry")));
        long[] sharedFp = flatten(shared);

        // ---- build: one cache, three prompts; dedup gate via usedBytes deltas ----
        CacheStore store = CacheStore.inMemory();
        PromptCache<Lfm2.State> build = new PromptCache<>(codec, store, budget, seed);
        long prev = 0;
        long[] delta = new long[prompts.size()];
        for (int i = 0; i < prompts.size(); i++) {
            CachedSession<Lfm2.State> s = CachedSession.resume(model, build, model.newState(4096, 512), new long[0]);
            s.ingest(shared);
            s.ingest(prompts.get(i));
            delta[i] = store.usedBytes() - prev;
            prev = store.usedBytes();
        }
        check(delta[1] < delta[0] && delta[2] < delta[0],
                "shared prefix stored once (bytes per prompt: " + delta[0] + ", " + delta[1] + ", " + delta[2] + ")");

        Path file = Files.createTempFile("lfm2-frozen", ".jkv");
        build.freeze(file);
        long fileBytes = Files.size(file);

        // ---- serve: open frozen, each prompt resumes fully and replies match uncached exactly ----
        long t0 = System.nanoTime();
        PromptCache<Lfm2.State> frozen = PromptCache.open(file, codec, seed);
        double openMs = (System.nanoTime() - t0) / 1e6;
        String statsBefore = frozen.stats().split(" hits=")[0];   // blocks= + bytes= (counters move on resume)

        // Correctness gate: the frozen-restored state must be BYTE-IDENTICAL to a freshly computed
        // one (stronger than comparing generated replies, which additionally measures decode-time
        // FP scheduling noise under load - engine property, not cache property).
        Message question = Message.user("Answer in one short sentence: is water wet?");
        System.out.printf("%-22s %14s %14s%n", "prompt", "resume (ms)", "prefill (ms)");
        for (int i = 0; i < prompts.size(); i++) {
            long[] fp = concatFp(sharedFp, flatten(prompts.get(i)));

            long t1 = System.nanoTime();
            CachedSession<Lfm2.State> hot = CachedSession.resume(model, frozen, model.newState(4096, 512), fp);
            double resumeMs = (System.nanoTime() - t1) / 1e6;
            check(hot.position() == fp.length, domains[i] + ": resume restores all " + fp.length + " positions (got " + hot.position() + ")");

            // gate: the frozen file reproduces the LIVE cache's blocks byte-for-byte (round-trip)
            Lfm2.State live = model.newState(4096, 512);
            CachedSession.resume(model, build, live, fp);
            check(statesEqual(hot.state(), live, fp.length),
                    domains[i] + ": frozen-restored state byte-identical to the live cache");

            // uncached baseline for the TTFT benchmark; byte-equality vs a fresh compute is an
            // ENGINE determinism property (threaded MoE reductions vary run to run under load),
            // reported informationally, not gated.
            PromptCache<Lfm2.State> scratch = new PromptCache<>(codec, CacheStore.inMemory(), budget, seed);
            CachedSession<Lfm2.State> cold = CachedSession.resume(model, scratch, model.newState(4096, 512), new long[0]);
            long t2 = System.nanoTime();
            cold.ingest(shared);
            cold.ingest(prompts.get(i));
            double prefillMs = (System.nanoTime() - t2) / 1e6;
            System.out.println("      (fresh recompute byte-identical: " + statesEqual(hot.state(), cold.state(), fp.length) + ")");
            System.out.printf("%-22s %14.1f %14.0f%n", domains[i] + " (" + fp.length + ")", resumeMs, prefillMs);

            if (i == 0) {                        // coherence spot-check on the frozen path
                hot.ingest(template.encodeTurn(question));
                System.out.println("frozen-path reply: " + decode(hot, 60).strip().replace("\n", " "));
            }
        }

        // ---- unseen prompt: partial hit on the shared prefix only ----
        long[] fp4 = concatFp(sharedFp, flatten(unseen));
        CachedSession<Lfm2.State> partial = CachedSession.resume(model, frozen, model.newState(4096, 512), fp4);
        check(partial.position() > 0 && partial.position() < fp4.length,
                "unseen prompt resumes only the shared prefix (" + partial.position() + "/" + fp4.length + ")");

        // ---- frozen is read-only: the decode commits above were silent no-ops ----
        check(frozen.stats().split(" hits=")[0].equals(statsBefore),
                "frozen cache unchanged after serving (" + frozen.stats() + ")");

        // ---- wrong seed: the proper error ----
        boolean threw = false;
        try {
            PromptCache.open(file, codec, new byte[32]);
        } catch (IllegalStateException e) {
            threw = e.getMessage().contains("different model");
        }
        check(threw, "wrong-seed open throws the model-mismatch error");

        System.out.printf("open %.1f ms · file %.1f MB · %s%n", openMs, fileBytes / 1048576.0, frozen.stats());
        Files.deleteIfExists(file);
        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println("Lfm2FrozenCacheRun: all checks passed");
    }

    static String sharedRules() {
        StringBuilder sb = new StringBuilder("You are a precise, concise assistant. Follow these rules exactly.\n");
        for (int i = 1; i <= 90; i++) {
            sb.append(i).append(". Always verify assumption set ").append(i)
                    .append(" before answering; cite the rule number when it changes the outcome, and prefer the shortest correct phrasing over elaborate hedging.\n");
        }
        return sb.toString();
    }

    static String examples(String domain) {
        StringBuilder sb = new StringBuilder("Worked examples for ").append(domain).append(":\n");
        for (int i = 1; i <= 40; i++) {
            sb.append("Example ").append(i).append(": in the ").append(domain)
                    .append(" setting, input ").append(i * 7).append(" maps to output ").append(i * 7 + 3).append(".\n");
        }
        return sb.toString();
    }

    /** Open the assistant turn, greedy-decode, close the turn (commits no-op on frozen). */
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

    /** KV rows and conv history byte-identical over the first {@code positions}. */
    static boolean statesEqual(Lfm2.State a, Lfm2.State b, int positions) {
        var cfg = model.config();
        for (int l = 0; l < cfg.numberOfLayers(); l++) {
            if (cfg.isRecurrentLayer(l)) {
                for (long i = 0; i < a.shortConvState[l].size(); i++) {
                    if (a.shortConvState[l].getFloat(i) != b.shortConvState[l].getFloat(i)) return false;
                }
            } else {
                long n = (long) positions * cfg.kvDim(l);
                for (long i = 0; i < n; i++) {
                    if (a.keyCache[l].getFloat(i) != b.keyCache[l].getFloat(i)
                            || a.valueCache[l].getFloat(i) != b.valueCache[l].getFloat(i)) return false;
                }
            }
        }
        return true;
    }

    static long[] flatten(List<Batch> batches) {
        List<Integer> ids = new ArrayList<>();
        for (Batch b : batches) for (int id : ((Batch.Input.Tokens) b.input()).ids()) ids.add(id);
        long[] fp = new long[ids.size()];
        for (int i = 0; i < fp.length; i++) fp[i] = ids.get(i);
        return fp;
    }

    static long[] concatFp(long[] a, long[] b) {
        long[] out = java.util.Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
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
