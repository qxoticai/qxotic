// Bonsai-27B (arch qwen35, Q1_0) greedy parity vs llama.cpp.
//   java ... com.qxotic.jinfer.models.qwen35.BonsaiParityCheck [model.gguf]
//
// References captured from llama.cpp build b9857-13e673863 (libggml 0.15.3):
//   LD_LIBRARY_PATH=build/bin build/bin/llama-completion -m Bonsai-27B-Q1_0.gguf \
//       -p "<prompt>" -n 32 --temp 0 -no-cnv --no-warmup --verbose-prompt
// Prompt tokenization was verified identical (e.g. "The capital of France is" ->
// [760, 6511, 314, 9338, 369] on both sides; no BOS). llama.cpp dots Q1_0 against
// Q8_0-quantized activations while jinfer keeps activations f32, so long generations may
// diverge; the gate is >= 24 of 32 greedy tokens (prefix-exact by cumulative decode).
package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.testkit.Checks;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class BonsaiParityCheck {

    private record Case(String name, String prompt, String reference, boolean endsWithStop) {}

    private static final List<Case> CASES =
            List.of(
                    new Case(
                            "capital",
                            "The capital of France is",
                            " Paris.\n\n<think>\nHere's a thinking process:\n\n1.  **Analyze User"
                                    + " Input:**\n   - Statement: \"The capital of France is",
                            false),
                    new Case("arithmetic", "2+2=", "4", true),
                    new Case(
                            "qubit",
                            "Quantum computing differs from classical computing in that it exploits"
                                    + " superposition and entanglement to process information. A"
                                    + " classical bit stores either zero or one, while a qubit can",
                            " exist in a superposition of both states. Entanglement allows qubits"
                                    + " to be correlated in ways that classical bits cannot,"
                                    + " enabling quantum computers to perform certain calculations",
                            false));

    private static final int N_TOKENS = 32;
    private static final int MIN_MATCH = 24;

    public static void main(String[] args) throws Exception {
        Path model =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/prism-ml/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf");
        if (!Files.exists(model)) {
            System.out.println("BonsaiParityCheck: model not found (" + model + "), skipping");
            return;
        }
        Qwen35 model35 = Qwen35.loadModel(model, 4096);
        var tokenizer = model35.tokenizer();
        Set<Integer> stops = model35.stopTokens();

        Checks checks = new Checks();
        for (Case c : CASES) {
            int[] prompt = tokenizer.encode(c.prompt()).toArray();
            Qwen35.State state =
                    model35.newState(model35.config().contextLength(), Math.max(16, prompt.length));
            model35.ingest(state, Batch.prefill(prompt));

            List<Integer> generated = new ArrayList<>();
            boolean stopped = false;
            int token = model35.logits(state).argmax();
            for (int n = 0; n < N_TOKENS; n++) {
                if (stops.contains(token)) {
                    stopped = true;
                    break;
                }
                generated.add(token);
                model35.ingest(state, Batch.step(token));
                token = model35.logits(state).argmax();
            }

            // Matched tokens = the longest generated prefix whose cumulative decode is still a
            // prefix of the reference continuation (token-exact runs decode to identical text).
            int matched = 0;
            StringBuilder text = new StringBuilder();
            for (int t : generated) {
                text.append(tokenizer.decode(t));
                if (!isPrefix(text, c.reference())) break;
                matched++;
            }
            int target = Math.min(MIN_MATCH, referenceTokens(c));
            boolean ok = matched >= target && (!c.endsWithStop() || stopped);
            checks.check(
                    ok,
                    String.format(
                            "%s - %d/%d tokens match llama.cpp%s",
                            c.name(),
                            matched,
                            generated.size(),
                            c.endsWithStop()
                                    ? (stopped ? " (+eos parity)" : " (MISSING eos)")
                                    : ""));
            if (!ok) {
                System.out.println("  ours:      " + escape(text.toString()));
                System.out.println("  reference: " + escape(c.reference()));
            }
        }
        checks.finish("BonsaiParityCheck", "all cases within tolerance");
    }

    /** The reference's own token budget bounds the target for short (eos-terminated) cases. */
    private static int referenceTokens(Case c) {
        return c.endsWithStop() ? 1 : MIN_MATCH;
    }

    private static boolean isPrefix(StringBuilder text, String reference) {
        return reference.startsWith(text.toString())
                || text.toString().startsWith(reference); // ours may run past a len-capped ref
    }

    private static String escape(String s) {
        return s.replace("\n", "\\n");
    }
}
