package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/** One response, using the conversation template or an explicitly raw prompt. */
final class Instruct {

    private Instruct() {}

    static boolean read(Options o, Options.Args a) {
        switch (a.name) {
            case "--prompt", "-p" -> o.input = a.value();
            case "--raw-prompt" -> o.rawPrompt = a.flag();
            default -> {
                return false;
            }
        }
        o.use(
                a.name,
                a.name.equals("--raw-prompt") ? Set.of("instruct", "server") : Set.of("instruct"));
        return true;
    }

    static void validate(Options o) {
        Options.require(
                o.input != null && !o.input.isBlank(),
                "instruct requires input text or '-' for stdin");
    }

    static void printHelp(PrintStream out) {
        out.println(
                """
                jinfer instruct - generate one response (alias: prompt)
                Usage: jinfer [model options] instruct [options] <text|->
                Examples:
                  jinfer instruct -m model.gguf "Explain virtual threads."
                  jinfer -m model.gguf --temp 0 instruct - < prompt.txt

                  -p, --prompt <text>         explicit spelling of the input argument
                  --raw-prompt               bypass the conversation template
                  --cache <file>             read and append a persistent prompt cache
                  --cache-ro <file>          serve a prompt cache without changing it
                """);
        Options.modelHelp(out);
        Options.generationHelp(out);
        Options.conversationHelp(out);
    }

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

    static void run(
            ChatEngine engine, Sampling sampling, Options options, Main.IO io, String prompt)
            throws IOException {
        if (options.rawPrompt) {
            int[] tokens =
                    withPromptStart(
                            engine.loaded()
                                    .template()
                                    .map(ChatTemplate::promptStart)
                                    .orElse(IntSequence.empty()),
                            SpecialTokens.encode(engine.loaded().tokenizer(), prompt).toArray());
            try (Turn turn = Turn.startRaw(engine.loaded().tokenizer(), tokens, options, io);
                    ChatEngine.Prepared prepared =
                            Requests.checked(
                                    ChatEngine.Prepared.raw(
                                            tokens,
                                            sampling.sampler(
                                                    engine.loaded()
                                                            .model()
                                                            .configuration()
                                                            .vocabularySize()),
                                            options.maxOutputTokens,
                                            Duration.ZERO,
                                            List.of()),
                                    engine.contextCapacity())) {
                turn.finish(engine.complete(prepared, turn), engine.contextCapacity());
            }
            return;
        }

        List<Message> turns = new ArrayList<>();
        if (options.systemPrompt != null) {
            turns.add(Message.system(options.systemPrompt));
        }
        turns.add(Message.user(prompt));
        Conversation conversation = new Conversation(turns, List.of(), options.think);

        // --cache: pin the prompt BEFORE generating - the artifact is the point of --cache, and a
        // generation failure must not lose it. The engine's cache then serves the longest cached
        // prefix on the complete() below, on its own.
        if (options.promptCache != null && !options.promptCacheReadOnly) {
            int before = engine.cacheSample().blocks();
            try {
                engine.definePrompt(conversation);
                engine.savePrompts();
            } catch (UnsupportedOperationException noCodec) {
                // cached prompts are a prefix-stability bet only a native codec can honor; a
                // Jinja-only model warns and serves without appending, exactly like the old CLI
                io.err().println("cache: " + noCodec.getMessage() + " - serving read-only");
            } catch (IllegalArgumentException | UncheckedIOException e) {
                throw Main.failure("cannot cache prompt in '" + options.promptCache + "'", e);
            }
            int added = engine.cacheSample().blocks() - before;
            if (added > 0) {
                io.err()
                        .printf(
                                "cache: %d blocks added, catalog appended (%s)%n",
                                added, options.promptCache);
            }
        }

        try (ChatEngine.Prepared prepared =
                        Requests.prepare(engine, conversation.messages(), sampling, options);
                Turn turn = Turn.start(engine.loaded().tokenizer(), prepared, options, io)) {
            turn.finish(engine.complete(prepared, turn), engine.contextCapacity());
        }
    }
}
