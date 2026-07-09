// E2E coherence check for Lfm2TurnTemplate: drive real chat turns through the template
// (encode -> Batch.prepare -> ingest -> generationPrompt -> greedy decode) and print the replies.
// Multi-turn: the assistant reply tokens stay in the KV verbatim; only the delta is ingested.
//   java ... com.qxotic.jinfer.models.lfm2.Lfm2ChatRun [model.gguf]
package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.TurnTemplate;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class Lfm2ChatRun {

    public static void main(String[] args) throws Exception {
        Path path =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        Lfm2 model = Lfm2.loadModel(path, 4096);
        var c = model.config();
        var tk = model.tokenizer();
        TurnTemplate template = model.turnTemplate().orElseThrow();
        Set<Integer> stops = model.stopTokens();

        Lfm2.State s = model.newState(c.contextLength(), 512);

        // Turn 1: system + user, incremental from conversation start.
        List<Batch> batches = new ArrayList<>(template.conversationStart());
        batches.addAll(
                template.encodeTurn(
                        Message.system("You are a concise assistant. Answer briefly.")));
        batches.addAll(template.encodeTurn(Message.user("Write a haiku about rivers.")));
        String reply1 = generate(model, s, template, batches, tk, stops, 200);
        System.out.println("=== haiku ===\n" + reply1 + "\n");

        // Turn 2: append ONLY the new user turn (reply1's tokens are already in the KV).
        String reply2 =
                generate(
                        model,
                        s,
                        template,
                        template.encodeTurn(
                                Message.user("Now: what is 17 * 23? Reply with just the number.")),
                        tk,
                        stops,
                        200);
        System.out.println("=== arithmetic ===\n" + reply2 + "\n");

        // Turn 3: reference the earlier conversation to prove the KV history is live.
        String reply3 =
                generate(
                        model,
                        s,
                        template,
                        template.encodeTurn(
                                Message.user(
                                        "What was the topic of the haiku you wrote earlier? One"
                                                + " word.")),
                        tk,
                        stops,
                        200);
        System.out.println("=== recall ===\n" + reply3 + "\n");

        System.out.println("context positions used: " + s.position());
    }

    /**
     * Ingest the turn batches + generation prompt, greedy-decode a reply, close the turn so the KV
     * ends exactly where encodeTurn(assistant reply) would have: ... reply <|im_end|> \n.
     */
    static String generate(
            Lfm2 model,
            Lfm2.State s,
            TurnTemplate template,
            List<Batch> turn,
            com.qxotic.jinfer.GgufTokenizer tk,
            Set<Integer> stops,
            int maxTokens) {
        List<Batch> ready = new ArrayList<>(turn);
        ready.addAll(template.generationPrompt(true));
        for (Batch b : Batch.prepare(ready, 512)) model.ingest(s, b);

        StringBuilder out = new StringBuilder();
        int imEnd = tk.getSpecialTokens().get("<|im_end|>");
        int tok = model.logits(s).argmax();
        int n = 0;
        long t0 = System.nanoTime();
        for (; n < maxTokens && !stops.contains(tok); n++) {
            out.append(tk.decode(tok));
            model.ingest(s, Batch.step(tok));
            tok = model.logits(s).argmax();
        }
        double secs = (System.nanoTime() - t0) / 1e9;
        // close the assistant turn in the KV: <|im_end|> \n (the turn framing encodeTurn would
        // emit)
        List<Integer> close = new ArrayList<>(List.of(imEnd));
        close.addAll(tk.encode("\n"));
        model.ingest(s, Batch.prefill(close.stream().mapToInt(Integer::intValue).toArray()));
        System.err.printf("[%d tokens, %.1f tok/s]%n", n, n / secs);
        return out.toString();
    }
}
