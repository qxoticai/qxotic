// The frozen multi-prompt cache (use case B) validation scenario: build a PromptCache holding
// THREE long prompts sharing a common system prefix (dedup gate), freeze() it, open() read-only,
// and prove each prompt resumes fully with a resume-state BYTE-IDENTICAL to the live cache's (the
// round-trip gate - fresh recomputes are not byte-deterministic under threaded scheduling, so
// recompute equality is reported informationally, not gated). An unseen prompt gets a partial
// (shared-prefix) hit; a wrong-seed open fails with the proper error; commits on the frozen
// instance are silent no-ops.
package com.qxotic.jinfer.testkit;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CacheStore;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Message;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class FrozenScenario<S extends RuntimeState> {

    private final Harness<S> h;

    public FrozenScenario(Harness<S> h) {
        this.h = h;
    }

    public void run(String runName) throws Exception {
        long budget = 4L << 30;

        // ---- three prompts sharing a long system prefix; one unseen (shares the prefix too) ----
        List<Batch> shared =
                Harness.concat(
                        h.template.conversationStart(),
                        h.template.encodeTurn(Message.system(sharedRules())));
        String[] domains = {"unit conversion", "capital cities", "arithmetic"};
        List<List<Batch>> prompts = new ArrayList<>();
        for (String d : domains) prompts.add(h.template.encodeTurn(Message.user(examples(d))));
        List<Batch> unseen = h.template.encodeTurn(Message.user(examples("chemistry")));
        long[] sharedFp = Harness.flatten(shared);

        // ---- build: one cache, three prompts; dedup gate via usedBytes deltas ----
        CacheStore store = CacheStore.inMemory();
        PromptCache<S> build = new PromptCache<>(h.codec, store, budget, h.seed);
        long prev = 0;
        long[] delta = new long[prompts.size()];
        for (int i = 0; i < prompts.size(); i++) {
            CachedSession<S> s =
                    CachedSession.resume(h.model.model(), build, h.newState(), new long[0]);
            s.ingest(shared);
            s.ingest(prompts.get(i));
            delta[i] = store.usedBytes() - prev;
            prev = store.usedBytes();
        }
        h.check(
                delta[1] < delta[0] && delta[2] < delta[0],
                "shared prefix stored once (bytes per prompt: "
                        + delta[0]
                        + ", "
                        + delta[1]
                        + ", "
                        + delta[2]
                        + ")");

        Path file = Files.createTempFile("frozen-cache", ".jkv");
        build.freeze(file);
        long fileBytes = Files.size(file);

        // ---- serve: open frozen, each prompt resumes fully, byte-identical to the live cache ----
        long t0 = System.nanoTime();
        PromptCache<S> frozen = PromptCache.open(file, h.codec, h.seed);
        double openMs = (System.nanoTime() - t0) / 1e6;
        String statsBefore =
                frozen.stats().split(" hits=")[0]; // blocks= + bytes= (counters move on resume)

        System.out.printf("%-22s %14s %14s%n", "prompt", "resume (ms)", "prefill (ms)");
        for (int i = 0; i < prompts.size(); i++) {
            long[] fp = Harness.concatFp(sharedFp, Harness.flatten(prompts.get(i)));

            long t1 = System.nanoTime();
            CachedSession<S> hot = CachedSession.resume(h.model.model(), frozen, h.newState(), fp);
            double resumeMs = (System.nanoTime() - t1) / 1e6;
            h.check(
                    hot.position() == fp.length,
                    domains[i]
                            + ": resume restores all "
                            + fp.length
                            + " positions (got "
                            + hot.position()
                            + ")");

            // gate: the frozen file reproduces the LIVE cache's resume-state byte-for-byte
            S live = h.newState();
            CachedSession.resume(h.model.model(), build, live, fp);
            h.check(
                    h.statesEqual(hot.state(), live, fp.length),
                    domains[i] + ": frozen-restored state byte-identical to the live cache");

            // uncached baseline for the TTFT benchmark; fresh-recompute byte-equality is an engine
            // determinism property (threaded reductions vary run to run), reported, not gated.
            CachedSession<S> cold =
                    CachedSession.resume(
                            h.model.model(),
                            new PromptCache<>(h.codec, CacheStore.inMemory(), budget, h.seed),
                            h.newState(),
                            new long[0]);
            long t2 = System.nanoTime();
            cold.ingest(shared);
            cold.ingest(prompts.get(i));
            double prefillMs = (System.nanoTime() - t2) / 1e6;
            System.out.println(
                    "      (fresh recompute byte-identical: "
                            + h.statesEqual(hot.state(), cold.state(), fp.length)
                            + ")");
            System.out.printf(
                    "%-22s %14.1f %14.0f%n",
                    domains[i] + " (" + fp.length + ")", resumeMs, prefillMs);

            if (i == 0) { // coherence spot-check on the frozen path
                hot.ingest(
                        h.template.encodeTurn(
                                Message.user("Answer in one short sentence: is water wet?")));
                System.out.println(
                        "frozen-path reply: "
                                + h.decode(hot, 60).text().strip().replace("\n", " "));
            }
        }

        // ---- unseen prompt: partial hit on the shared prefix only ----
        long[] fp4 = Harness.concatFp(sharedFp, Harness.flatten(unseen));
        CachedSession<S> partial = CachedSession.resume(h.model.model(), frozen, h.newState(), fp4);
        h.check(
                partial.position() > 0 && partial.position() < fp4.length,
                "unseen prompt resumes only the shared prefix ("
                        + partial.position()
                        + "/"
                        + fp4.length
                        + ")");

        // ---- frozen is read-only: the decode commits above were silent no-ops ----
        h.check(
                frozen.stats().split(" hits=")[0].equals(statsBefore),
                "frozen cache unchanged after serving (" + frozen.stats() + ")");

        // ---- wrong seed: the proper error ----
        boolean threw = false;
        try {
            PromptCache.open(file, h.codec, new byte[32]);
        } catch (IllegalStateException e) {
            threw = e.getMessage().contains("different model");
        }
        h.check(threw, "wrong-seed open throws the model-mismatch error");

        System.out.printf(
                "open %.1f ms · file %.1f MB · %s%n",
                openMs, fileBytes / 1048576.0, frozen.stats());
        Files.deleteIfExists(file);
        h.finish(runName);
    }

    private static String sharedRules() {
        StringBuilder sb =
                new StringBuilder(
                        "You are a precise, concise assistant. Follow these rules exactly.\n");
        for (int i = 1; i <= 90; i++) {
            sb.append(i)
                    .append(". Always verify assumption set ")
                    .append(i)
                    .append(
                            " before answering; cite the rule number when it changes the outcome,"
                                    + " and prefer the shortest correct phrasing over elaborate"
                                    + " hedging.\n");
        }
        return sb.toString();
    }

    private static String examples(String domain) {
        StringBuilder sb = new StringBuilder("Worked examples for ").append(domain).append(":\n");
        for (int i = 1; i <= 40; i++) {
            sb.append("Example ")
                    .append(i)
                    .append(": in the ")
                    .append(domain)
                    .append(" setting, input ")
                    .append(i * 7)
                    .append(" maps to output ")
                    .append(i * 7 + 3)
                    .append(".\n");
        }
        return sb.toString();
    }
}
