package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.llm.SpecialTokens;
import java.io.IOException;
import java.io.PrintStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/** One response, using the conversation template or an explicitly raw prompt. */
final class Instruct {

    private Instruct() {}

    static boolean read(Options o, Options.Args a) {
        if (!a.name.equals("--raw-prompt")) return false;
        o.rawPrompt = a.flag();
        o.use(a.name, Set.of("instruct", "server"));
        return true;
    }

    static void validate(Options o) {
        Options.require(
                o.input != null && !o.input.isBlank(),
                "instruct requires input text or '-' for stdin");
        o.rejectMedia();
    }

    static void printHelp(PrintStream out) {
        out.println(
                """
                jinfer instruct - generate one response (alias: prompt)
                Usage: jinfer [model options] instruct [options] <text|->
                Examples:
                  jinfer instruct -m model.gguf "Explain virtual threads."
                  jinfer -m model.gguf --temp 0 instruct - < prompt.txt

                Instruct options (after the command):
                  --raw-prompt               bypass the conversation template
                  --cache <file>             read and append a persistent prompt cache
                  --cache-ro <file>          serve a prompt cache without changing it
                """);
        Options.modelHelp(out);
        Options.generationHelp(out);
        Options.conversationHelp(out);
    }

    static void run(
            ChatEngine engine, Sampling sampling, Options options, Main.IO io, String prompt)
            throws IOException {
        if (options.rawPrompt) {
            // prepareRaw puts the model's start tokens (BOS) ahead, as llama.cpp's add_bos_token
            int[] tokens = SpecialTokens.encode(engine.loaded().tokenizer(), prompt).toArray();
            try (ChatEngine.Prepared prepared =
                            Requests.checked(
                                    engine.prepareRaw(
                                            tokens,
                                            sampling,
                                            options.maxOutputTokens,
                                            Duration.ZERO,
                                            null,
                                            List.of()),
                                    engine.contextCapacity());
                    Turn turn =
                            Turn.startRaw(
                                    engine.loaded().tokenizer(),
                                    Batch.tokenIds(prepared.encoded().prompt()),
                                    options,
                                    io)) {
                turn.finish(engine.complete(prepared, turn), engine);
            }
            return;
        }

        List<Message> turns = new ArrayList<>();
        if (options.systemPrompt != null) {
            turns.add(Message.system(options.systemPrompt));
        }
        turns.add(Message.user(prompt));
        ChatEngine.Request request = Requests.of(turns, sampling, options);

        // --cache: pin the prompt BEFORE generating - the artifact is the point of --cache, and a
        // generation failure must not lose it. The engine's cache then serves the longest cached
        // prefix on the complete() below, on its own.
        if (options.promptCache != null && !options.promptCacheReadOnly) {
            int before = engine.cacheSample().blocks();
            long start = System.nanoTime();
            try {
                engine.definePrompt(request);
                engine.savePrompts();
            } catch (UnsupportedOperationException noCodec) {
                // cached prompts are a prefix-stability bet only a native codec can honor; a
                // Jinja-only model warns and serves without appending
                io.err().println("cache: " + noCodec.getMessage() + " - serving read-only");
            }
            int added = engine.cacheSample().blocks() - before;
            if (added > 0) {
                // the prefill happens here, so the pass below restores this prompt from these
                // blocks: say where its time went, or the summary's "restored" reads as a disk hit
                io.err()
                        .printf(
                                Locale.ROOT,
                                "cache: prefilled %d new block%s in %.2f s, appended to %s%n",
                                added,
                                added == 1 ? "" : "s",
                                (System.nanoTime() - start) / 1e9,
                                options.promptCache);
            }
        }

        try (ChatEngine.Prepared prepared =
                        Requests.checked(engine.prepare(request), engine.contextCapacity());
                Turn turn = Turn.start(engine.loaded().tokenizer(), prepared, options, io)) {
            turn.finish(engine.complete(prepared, turn), engine);
        }
    }
}
