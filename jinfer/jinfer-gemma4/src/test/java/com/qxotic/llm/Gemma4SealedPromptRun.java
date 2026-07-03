// SealedPrompt validation on Gemma 4 E2B: compile a long static prompt (system + few-shot examples)
// to a sealed file, then serve from a fresh state: open+restore in ms instead of re-prefilling.
// Checks: full restore; sealed vs uncached greedy replies IDENTICAL; prompt mismatch falls
// through to plain prefill (same reply); wrong model seed fails with the descriptive error.
//   java ... com.qxotic.llm.Gemma4SealedPromptRun [model.gguf]
package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.cache.SealedPrompt;
import com.qxotic.jinfer.chat.Message;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class Gemma4SealedPromptRun {

    static int failures;
    static Gemma4 model;
    static Gemma4TurnTemplate template;
    static Set<Integer> stops;

    public static void main(String[] args) throws Exception {
        Path path = Path.of(args.length > 0 ? args[0] : "/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf");
        model = Gemma4.loadModel(path, 8192);
        template = new Gemma4TurnTemplate(model.tokenizer());
        stops = model.stopTokens();
        Gemma4KvCodec codec = new Gemma4KvCodec(model.config());
        byte[] seed = PromptCache.modelSeed(path);
        Path sealed = Files.createTempFile("gemma4-prompt", ".jkvs");

        // ---- build: prefill the static prompt once, seal it ----
        List<Batch> staticPrompt = staticPrompt();
        Gemma4.State build = model.newState(8192, 512);
        long t0 = System.nanoTime();
        long[] fp = ingest(build, staticPrompt);
        double prefillMs = (System.nanoTime() - t0) / 1e6;
        SealedPrompt.compile(sealed, "support-bot", codec, build, fp, seed);
        System.out.printf("compiled: %d positions, %.1f MB (%s)%n", fp.length, Files.size(sealed) / 1e6, sealed);

        // ---- reference: uncached full prefill + user turn ----
        Message user = Message.user("Convert 250 kilometers to miles. Answer with the sentence format from your instructions.");
        Gemma4.State ref = model.newState(8192, 512);
        ingest(ref, staticPrompt);
        String refReply = serve(ref, user);
        System.out.println("reference reply: " + refReply.strip());

        // ---- serve: open + restore from the sealed file (fresh state) ----
        long t1 = System.nanoTime();
        SealedPrompt prompt = SealedPrompt.open(sealed, seed);
        Gemma4.State hot = model.newState(8192, 512);
        int restored = prompt.tryRestore(hot, codec, fp);
        double restoreMs = (System.nanoTime() - t1) / 1e6;
        check(restored == fp.length, "sealed restore covers all " + fp.length + " positions (got " + restored + ")");
        String sealedReply = serve(hot, user);
        check(sealedReply.equals(refReply), "sealed and uncached greedy replies identical");

        // ---- mismatch: a different prompt discards the seal, plain prefill still serves ----
        long[] other = fp.clone();
        other[3] ^= 1;
        Gemma4.State cold = model.newState(8192, 512);
        check(prompt.tryRestore(cold, codec, other) == 0, "different prompt is discarded (restore 0)");
        ingest(cold, staticPrompt);
        check(serve(cold, user).equals(refReply), "fall-through prefill serves the same reply");

        // ---- wrong model: open fails with the descriptive error ----
        byte[] wrong = seed.clone();
        wrong[0] ^= 1;
        try {
            SealedPrompt.open(sealed, wrong);
            check(false, "wrong-seed open must throw");
        } catch (IllegalStateException e) {
            check(e.getMessage().contains("different model") && e.getMessage().contains("support-bot"),
                    "wrong model rejected: " + e.getMessage().substring(0, Math.min(100, e.getMessage().length())));
        }

        System.out.printf("%n=== benchmark: sealed restore vs static-prompt prefill ===%n");
        System.out.printf("%-34s %10.1f ms%n", "static prompt prefill (" + fp.length + " tok)", prefillMs);
        System.out.printf("%-34s %10.1f ms   (%.0fx)%n", "sealed open+restore", restoreMs, prefillMs / restoreMs);
        Files.deleteIfExists(sealed);
        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println("Gemma4SealedPromptRun: all checks passed");
    }

    /** The compiled artifact: detailed instructions + few-shot examples, the long prompts small
     *  models need (2000+ tokens). */
    static List<Batch> staticPrompt() {
        StringBuilder sys = new StringBuilder("""
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
                sys.append("Example: convert ").append(e[0]).append(" to ").append(e[1])
                        .append(". Correct answer: \"").append(e[0]).append(" is ").append(e[2])
                        .append(' ').append(e[1]).append(".\" Remember rounding to two decimals and the exact sentence form.\n");
            }
        }
        List<Batch> out = new ArrayList<>(template.conversationStart());
        out.addAll(template.encodeTurn(Message.system(sys.toString())));
        out.addAll(template.encodeTurn(Message.user("Convert 10 kilometers to miles.")));
        out.addAll(template.encodeTurn(Message.assistant("10 kilometers is 6.21 miles.")));
        return out;
    }

    /** Plain chunked ingest (no cache); returns the ingested fingerprints. */
    static long[] ingest(Gemma4.State state, List<Batch> batches) {
        List<Long> fp = new ArrayList<>();
        for (Batch b : Batch.prepare(batches, state.batchCapacity())) {
            model.ingest(state, b);
            for (int id : ((Batch.Input.Tokens) b.input()).ids()) fp.add((long) id);
        }
        long[] a = new long[fp.size()];
        for (int i = 0; i < a.length; i++) a[i] = fp.get(i);
        return a;
    }

    /** Ingest the user turn + generation prompt, greedy-decode the reply. */
    static String serve(Gemma4.State state, Message user) {
        ingest(state, template.encodeTurn(user));
        ingest(state, template.generationPrompt(true));
        StringBuilder out = new StringBuilder();
        long t0 = System.nanoTime();
        int n = 0;
        int tok = LLM.argmax(model.logits(state), model.config().vocabularySize());
        for (; n < 200 && !stops.contains(tok); n++) {
            out.append(model.tokenizer().decode(tok));
            model.ingest(state, Batch.step(tok));
            tok = LLM.argmax(model.logits(state), model.config().vocabularySize());
        }
        System.err.printf("[decode %.1f tok/s]%n", n / ((System.nanoTime() - t0) / 1e9));
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
