// The frozen single-prompt (use case A) validation scenario: compile a long static prompt
// (system rules + few-shot examples - the detailed prompts small models need) into a one-chain
// FrozenBlocks artifact, then serve from a fresh state. Checks: full restore; frozen vs uncached
// greedy replies IDENTICAL; prompt mismatch falls through to plain prefill (same reply); wrong
// model seed fails with the descriptive error.
package com.qxotic.jinfer.testkit;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.FrozenBlocks;
import com.qxotic.jinfer.chat.Message;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class FrozenPromptScenario<S extends RuntimeState> {

    private final Harness<S> h;

    public FrozenPromptScenario(Harness<S> h) {
        this.h = h;
    }

    public void run(String runName) throws Exception {
        Path artifact = Files.createTempFile("frozen-prompt", ".jkv");

        // ---- build: compile the static prompt into the artifact (one prefill) ----
        List<Batch> staticPrompt = staticPrompt();
        int positions = staticPrompt.stream().mapToInt(Batch::count).sum();
        long t0 = System.nanoTime();
        try (var pc =
                com.qxotic.jinfer.cache.PromptCache.of(
                        h.model.model(),
                        h.seed,
                        new com.qxotic.jinfer.cache.PromptCache.Options(
                                0, Long.MAX_VALUE, null, false))) {
            pc.define(staticPrompt);
            pc.export(artifact);
        }
        double prefillMs = (System.nanoTime() - t0) / 1e6;
        System.out.printf(
                "compiled: %d positions, %.1f MB (%s)%n",
                positions, Files.size(artifact) / 1e6, artifact);

        // ---- reference: uncached full prefill + user turn ----
        Message user =
                Message.user(
                        "Convert 250 kilometers to miles. Answer with the sentence format from your"
                                + " instructions.");
        S ref = h.newState();
        h.ingest(ref, staticPrompt);
        String refReply = h.serve(ref, user, 200);
        System.out.println("reference reply: " + refReply.strip());

        // ---- serve: open + restore from the frozen artifact (fresh state) ----
        long t1 = System.nanoTime();
        FrozenBlocks frozen = FrozenBlocks.open(artifact, h.seed);
        S hot = h.newState();
        CachedSession<S> hs =
                CachedSession.resume(h.model.model(), graftOn(frozen), hot, staticPrompt);
        double restoreMs = (System.nanoTime() - t1) / 1e6;
        h.check(
                hs.position() == positions,
                "frozen restore covers all "
                        + positions
                        + " positions (got "
                        + hs.position()
                        + ")");
        h.check(
                h.serve(hot, user, 200).equals(refReply),
                "frozen and uncached greedy replies identical");

        // ---- mismatch: a diverged prompt restores nothing, plain prefill still serves ----
        List<Batch> other = divergedAtPosition3(staticPrompt);
        S cold = h.newState();
        CachedSession<S> cs = CachedSession.resume(h.model.model(), graftOn(frozen), cold, other);
        h.check(cs.position() == 0, "diverged prompt is discarded (restore 0)");
        h.ingest(cold, staticPrompt);
        h.check(
                h.serve(cold, user, 200).equals(refReply),
                "fall-through prefill serves the same reply");

        // ---- wrong model: open fails with the descriptive error ----
        byte[] wrong = h.seed.clone();
        wrong[0] ^= 1;
        try {
            FrozenBlocks.open(artifact, wrong);
            h.check(false, "wrong-seed open must throw");
        } catch (IllegalStateException e) {
            h.check(
                    e.getMessage().contains("different model"),
                    "wrong model rejected: "
                            + e.getMessage().substring(0, Math.min(100, e.getMessage().length())));
        }

        System.out.printf("%n=== benchmark: frozen restore vs static-prompt prefill ===%n");
        System.out.printf(
                "%-34s %10.1f ms%n", "static prompt prefill (" + positions + " tok)", prefillMs);
        System.out.printf(
                "%-34s %10.1f ms   (%.0fx)%n",
                "frozen open+restore", restoreMs, prefillMs / restoreMs);
        Files.deleteIfExists(artifact);
        h.finish(runName);
    }

    /** A serve-only tree grafted over the artifact (budget 0: restores, never keeps writes). */
    private com.qxotic.jinfer.cache.BlockTree<S> graftOn(FrozenBlocks frozen) {
        return new com.qxotic.jinfer.cache.BlockTree<>(
                h.codec, com.qxotic.jinfer.cache.CacheStore.inMemory(), 0, h.seed, frozen);
    }

    /**
     * The compiled artifact: detailed instructions + few-shot examples, the long prompts small
     * models need (2000+ tokens).
     */
    private List<Batch> staticPrompt() {
        StringBuilder sys =
                new StringBuilder(
                        """
                        You are a precise unit-conversion assistant. Follow these rules exactly:
                        1. Always answer with one sentence of the form: "<input> is <result> <unit>."
                        2. Round results to two decimal places.
                        3. Never add commentary, caveats, or extra sentences.
                        4. If the request is not a unit conversion, reply exactly: "I only convert units."
                        Worked examples you must imitate:
                        """);
        String[][] examples = {
            {"5 kilometers", "miles", "3.11"}, {"12 miles", "kilometers", "19.31"},
            {"100 celsius", "fahrenheit", "212.00"}, {"32 fahrenheit", "celsius", "0.00"},
            {"3 kilograms", "pounds", "6.61"}, {"150 pounds", "kilograms", "68.04"},
            {"2 liters", "gallons", "0.53"}, {"5 gallons", "liters", "18.93"},
            {"90 minutes", "hours", "1.50"}, {"3 hectares", "acres", "7.41"},
            {"60 mph", "km/h", "96.56"}, {"1 nautical mile", "kilometers", "1.85"},
        };
        for (int round = 0; round < 6; round++) {
            for (String[] e : examples) {
                sys.append("Example: convert ")
                        .append(e[0])
                        .append(" to ")
                        .append(e[1])
                        .append(". Correct answer: \"")
                        .append(e[0])
                        .append(" is ")
                        .append(e[2])
                        .append(' ')
                        .append(e[1])
                        .append(
                                ".\" Remember rounding to two decimals and the exact sentence"
                                        + " form.\n");
            }
        }
        List<Batch> out = new ArrayList<>(h.template.conversationStart());
        out.addAll(h.template.encodeTurn(Message.system(sys.toString())));
        out.addAll(h.template.encodeTurn(Message.user("Convert 10 kilometers to miles.")));
        out.addAll(h.template.encodeTurn(Message.assistant("10 kilometers is 6.21 miles.")));
        return out;
    }

    /** The prompt with token position 3 flipped - diverges inside the first block. */
    private static List<Batch> divergedAtPosition3(List<Batch> prompt) {
        List<Batch> out = new java.util.ArrayList<>();
        int pos = 0;
        for (Batch b : prompt) {
            if (b.input() instanceof Batch.Input.Tokens t && pos <= 3 && 3 < pos + b.count()) {
                int[] ids = t.ids().clone();
                ids[3 - pos] ^= 1;
                out.add(Batch.prefill(ids));
            } else {
                out.add(b);
            }
            pos += b.count();
        }
        return out;
    }
}
