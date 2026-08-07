package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JinjaChatTemplate;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.llm.*;
import com.qxotic.toknroll.IntSequence;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** The interactive {@code --chat} loop, in its codec and whole-render forms. */
final class Chat {

    private Chat() {}

    /**
     * Interactive chat on a NATIVE codec: ONE running {@link Conversation}, re-encoded whole each
     * turn; the longest common prefix with the token stream the KV already holds is skipped and
     * only the suffix is ingested. Replies are appended as the parser's structured message
     * (verbatim ids), so the codec's verbatim splice keeps every generated turn inside the common
     * prefix - the append-only happy path ingests exactly closeTurn + the user turn + the scaffold,
     * like the per-turn flow. Any divergence rebuilds the state from scratch (correctness first;
     * the splice makes it rare).
     */
    static <S extends RuntimeState> void runCodec(
            LoadedModel<S> model, ChatTemplate template, Sampler sampler, Options options)
            throws IOException {
        Set<Integer> stops = model.stopTokens();
        int capacity = options.contextCapacity();
        S state = model.model().newState(capacity, RuntimeFlags.BATCH_CAPACITY);
        List<Message> opening = new ArrayList<>();
        if (options.systemPrompt() != null) {
            opening.add(Message.system(options.systemPrompt()));
        }
        Conversation conversation = new Conversation(opening, List.of(), options.think(), "");
        IntSequence ingested = IntSequence.empty(); // the token stream the KV holds
        try (BufferedReader reader =
                new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                String userText = reader.readLine();
                if (userText == null || "/quit".equals(userText) || "/exit".equals(userText)) break;
                if ("/context".equals(userText)) {
                    System.out.printf("context: %d/%d tokens%n", state.position(), capacity);
                    continue;
                }
                conversation = conversation.append(Message.user(userText));
                IntSequence prompt =
                        IntSequence.wrap(Batch.tokenIds(template.encode(conversation)));
                int lcp = commonPrefix(ingested, prompt);
                IntSequence delta;
                if (lcp < ingested.length()) {
                    close(state); // a dropped state frees NOW, not when GC notices gigabytes
                    state = model.model().newState(capacity, RuntimeFlags.BATCH_CAPACITY);
                    delta = prompt;
                } else {
                    delta = prompt.subSequence(lcp, prompt.length());
                }
                Turn.Reply reply = Turn.generate(model, state, delta, stops, sampler, options);
                conversation = conversation.append(reply.message());
                // The KV holds the prompt plus every INGESTED reply token: all of them when a
                // stop token ended the turn, all but the last otherwise (the decode loop never
                // ingests the final sampled token).
                IntSequence generated = reply.result().tokens();
                if (reply.result().stopToken() < 0 && !generated.isEmpty()) {
                    generated = generated.subSequence(0, generated.length() - 1);
                }
                ingested = prompt.concat(generated);
            }
        }
    }

    /**
     * Whole-render fallback for models without a TurnTemplate: re-encode the full conversation
     * through the Jinja template each turn, fresh state.
     */
    static <S extends RuntimeState> void runWholeRender(
            LoadedModel<S> model, Sampler sampler, Options options) throws IOException {
        Set<Integer> stops = model.stopTokens();
        JinjaChatTemplate jinja =
                new JinjaChatTemplate(model.tokenizer(), model.chatTemplateSource());
        List<Object> history = new ArrayList<>();
        if (options.systemPrompt() != null) {
            history.add(Map.of("role", "system", "content", options.systemPrompt()));
        }
        try (BufferedReader reader =
                new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                String userText = reader.readLine();
                if (userText == null || "/quit".equals(userText) || "/exit".equals(userText)) break;
                history.add(Map.of("role", "user", "content", userText));
                IntSequence promptTokens = jinja.render(history, null, true, options.think(), null);
                S state =
                        Generator.stateFor(
                                model.model(), promptTokens.length(), options.contextCapacity());
                try {
                    Turn.Reply reply =
                            Turn.generate(model, state, promptTokens, stops, sampler, options);
                    history.add(Map.of("role", "assistant", "content", reply.text()));
                } finally {
                    // a fresh KV state per turn: without this close, every turn parks a
                    // context-sized allocation on the Cleaner until GC notices
                    close(state);
                }
            }
        }
    }

    /** Frees a state's memory deterministically; states own a shared arena behind BaseState. */
    private static void close(RuntimeState state) {
        if (state instanceof BaseState base) {
            base.close();
        }
    }

    private static int commonPrefix(IntSequence a, IntSequence b) {
        int n = Math.min(a.length(), b.length());
        int i = 0;
        while (i < n && a.intAt(i) == b.intAt(i)) i++;
        return i;
    }
}
