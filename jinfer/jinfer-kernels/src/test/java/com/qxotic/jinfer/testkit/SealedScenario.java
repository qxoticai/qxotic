// The SealedPrompt (use case A) validation scenario: compile a long static prompt (system rules +
// few-shot examples - the detailed prompts small models need) to a sealed file, then serve from a
// fresh state. Checks: full restore; sealed vs uncached greedy replies IDENTICAL; prompt mismatch
// falls through to plain prefill (same reply); wrong model seed fails with the descriptive error.
package com.qxotic.jinfer.testkit;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.cache.SealedPrompt;
import com.qxotic.jinfer.chat.Message;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class SealedScenario<S extends RuntimeState> {

    private final Harness<S> h;

    public SealedScenario(Harness<S> h) {
        this.h = h;
    }

    public void run(String runName) throws Exception {
        Path sealed = Files.createTempFile("sealed-prompt", ".jkvs");

        // ---- build: prefill the static prompt once, seal it ----
        List<Batch> staticPrompt = staticPrompt();
        S build = h.newState();
        long t0 = System.nanoTime();
        long[] fp = h.ingest(build, staticPrompt);
        double prefillMs = (System.nanoTime() - t0) / 1e6;
        SealedPrompt.compile(sealed, "support-bot", h.codec, build, fp, h.seed);
        System.out.printf(
                "compiled: %d positions, %.1f MB (%s)%n",
                fp.length, Files.size(sealed) / 1e6, sealed);

        // ---- reference: uncached full prefill + user turn ----
        Message user =
                Message.user(
                        "Convert 250 kilometers to miles. Answer with the sentence format from your"
                                + " instructions.");
        S ref = h.newState();
        h.ingest(ref, staticPrompt);
        String refReply = h.serve(ref, user, 200);
        System.out.println("reference reply: " + refReply.strip());

        // ---- serve: open + restore from the sealed file (fresh state) ----
        long t1 = System.nanoTime();
        SealedPrompt prompt = SealedPrompt.open(sealed, h.seed);
        S hot = h.newState();
        int restored = prompt.tryRestore(hot, h.codec, fp);
        double restoreMs = (System.nanoTime() - t1) / 1e6;
        h.check(
                restored == fp.length,
                "sealed restore covers all " + fp.length + " positions (got " + restored + ")");
        h.check(
                h.serve(hot, user, 200).equals(refReply),
                "sealed and uncached greedy replies identical");

        // ---- mismatch: a different prompt discards the seal, plain prefill still serves ----
        long[] other = fp.clone();
        other[3] ^= 1;
        S cold = h.newState();
        h.check(
                prompt.tryRestore(cold, h.codec, other) == 0,
                "different prompt is discarded (restore 0)");
        h.ingest(cold, staticPrompt);
        h.check(
                h.serve(cold, user, 200).equals(refReply),
                "fall-through prefill serves the same reply");

        // ---- wrong model: open fails with the descriptive error ----
        byte[] wrong = h.seed.clone();
        wrong[0] ^= 1;
        try {
            SealedPrompt.open(sealed, wrong);
            h.check(false, "wrong-seed open must throw");
        } catch (IllegalStateException e) {
            h.check(
                    e.getMessage().contains("different model")
                            && e.getMessage().contains("support-bot"),
                    "wrong model rejected: "
                            + e.getMessage().substring(0, Math.min(100, e.getMessage().length())));
        }

        System.out.printf("%n=== benchmark: sealed restore vs static-prompt prefill ===%n");
        System.out.printf(
                "%-34s %10.1f ms%n", "static prompt prefill (" + fp.length + " tok)", prefillMs);
        System.out.printf(
                "%-34s %10.1f ms   (%.0fx)%n",
                "sealed open+restore", restoreMs, prefillMs / restoreMs);
        Files.deleteIfExists(sealed);
        h.finish(runName);
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
}
