package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/** The one-shot {@code --prompt} mode: encode, generate once, exit. The default mode. */
final class Instruct {

    private Instruct() {}

    /**
     * The model's start tokens (BOS, where the family writes one) ahead of a raw prompt, as
     * llama.cpp's {@code add_bos_token} does: a raw prompt bypasses the chat template, not the
     * framing every input sequence begins with. A prompt that already spells them is left alone.
     */
    static int[] withPromptStart(IntSequence start, int[] tokens) {
        int n = start.length();
        if (n == 0) return tokens;
        boolean present = tokens.length >= n;
        for (int i = 0; present && i < n; i++) present = tokens[i] == start.intAt(i);
        if (present) return tokens;
        int[] out = new int[n + tokens.length];
        for (int i = 0; i < n; i++) out[i] = start.intAt(i);
        System.arraycopy(tokens, 0, out, n, tokens.length);
        return out;
    }

    static void run(ChatEngine engine, Sampling sampling, Options options) {
        if (options.rawPrompt()) {
            int[] tokens =
                    withPromptStart(
                            engine.loaded()
                                    .template()
                                    .map(ChatTemplate::promptStart)
                                    .orElse(IntSequence.empty()),
                            SpecialTokens.encode(engine.loaded().tokenizer(), options.prompt())
                                    .toArray());
            Turn turn = Turn.startRaw(engine.loaded().tokenizer(), tokens, options);
            ChatEngine.Completion completion;
            try (ChatEngine.Prepared prepared =
                    ChatEngine.Prepared.raw(
                            tokens,
                            sampling.sampler(
                                    engine.loaded().model().configuration().vocabularySize()),
                            options.maxOutputTokens(),
                            Duration.ZERO,
                            List.of())) {
                completion = engine.complete(prepared, turn);
            }
            turn.finish(completion, engine.contextCapacity());
            return;
        }

        List<Message> turns = new ArrayList<>();
        if (options.systemPrompt() != null) {
            turns.add(Message.system(options.systemPrompt()));
        }
        turns.add(Message.user(options.prompt()));
        Conversation conversation = new Conversation(turns, List.of(), options.think());

        // --cache: pin the prompt BEFORE generating - the artifact is the point of --cache, and a
        // generation failure must not lose it. The engine's cache then serves the longest cached
        // prefix on the complete() below, on its own.
        if (options.promptCache() != null && !options.promptCacheReadOnly()) {
            int before = engine.cacheSample().blocks();
            try {
                engine.definePrompt(conversation);
                engine.savePrompts();
            } catch (UnsupportedOperationException noCodec) {
                // cached prompts are a prefix-stability bet only a native codec can honor; a
                // Jinja-only model warns and serves without appending, exactly like the old CLI
                System.err.println("cache: " + noCodec.getMessage() + " - serving read-only");
            }
            int added = engine.cacheSample().blocks() - before;
            if (added > 0) {
                System.err.printf(
                        "cache: %d blocks added, catalog appended (%s)%n",
                        added, options.promptCache());
            }
        }

        ChatEngine.Completion completion;
        Turn turn;
        try (ChatEngine.Prepared prepared =
                engine.prepare(Requests.of(conversation.messages(), sampling, options))) {
            turn = Turn.start(engine.loaded().tokenizer(), prepared, options);
            completion = engine.complete(prepared, turn);
        }
        turn.finish(completion, engine.contextCapacity());
    }
}
