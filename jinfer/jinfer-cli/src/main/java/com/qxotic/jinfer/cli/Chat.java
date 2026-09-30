package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.llm.Sampling;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/** One conversation; the engine's retained session carries its KV state across turns. */
final class Chat {

    private Chat() {}

    static void validate(Options options) {
        Options.require(
                options.input == null,
                "chat takes no input argument; use instruct for one response");
    }

    static void printHelp(PrintStream out) {
        out.println(
                """
                jinfer chat - have a conversation
                Usage: jinfer [model options] chat [options]
                Example: jinfer chat -m model.gguf --system-prompt "Be concise."

                Commands: /quit, /exit, /context. EOF also exits.
                """);
        Options.modelHelp(out);
        Options.generationHelp(out);
        Options.conversationHelp(out);
    }

    static void run(ChatEngine engine, Sampling sampling, Options options, Main.IO io)
            throws IOException {
        List<Message> history = new ArrayList<>();
        if (options.systemPrompt != null) {
            history.add(Message.system(options.systemPrompt));
        }
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(io.in(), StandardCharsets.UTF_8));
        while (true) {
            if (io.isTerminal(0) && io.isTerminal(2)) {
                io.err().print("> ");
                io.err().flush();
            }
            String userText;
            try {
                userText = reader.readLine();
            } catch (IOException e) {
                throw Main.failure("cannot read chat input from stdin", e);
            }
            if (userText == null) break;
            userText = userText.strip();
            if ("/quit".equals(userText) || "/exit".equals(userText)) break;
            if (userText.isEmpty()) {
                continue; // an empty turn would scaffold a reply to nothing
            }
            if ("/context".equals(userText)) {
                io.err()
                        .printf(
                                "context: capacity %d tokens, %s%n",
                                engine.contextCapacity(), engine.sessionStats());
                continue;
            }
            history.add(Message.user(userText));
            ChatEngine.Prepared prepared;
            try {
                prepared = prepare(engine, history, sampling, options, io);
            } catch (IllegalArgumentException
                    | IllegalStateException
                    | UnsupportedOperationException e) {
                // the library refusing this turn (what Main.run prints as one line): drop the
                // turn, keep the session; a bug is not caught here and ends it
                history.removeLast();
                io.err().println("jinfer chat: " + e.getMessage());
                continue;
            }
            ChatEngine.Completion completion;
            try (prepared;
                    Turn turn = Turn.start(engine.loaded().tokenizer(), prepared, options, io)) {
                completion = engine.complete(prepared, turn);
                turn.finish(completion, engine.contextCapacity());
            }
            if (completion.reply() != null) {
                // the parser's structured message (verbatim ids): the codec's verbatim splice
                // keeps generated turns inside the cache's common prefix
                history.add(completion.reply());
            }
        }
    }

    /**
     * The turn prepared to fit the context: past the window, the oldest exchanges are dropped from
     * {@code history} until it does, keeping the system prompt and the message just typed, and
     * stderr says so. A message that does not fit on its own is refused as any other.
     */
    private static ChatEngine.Prepared prepare(
            ChatEngine engine,
            List<Message> history,
            Sampling sampling,
            Options options,
            Main.IO io) {
        int dropped = 0;
        while (true) {
            ChatEngine.Prepared prepared =
                    engine.prepare(Requests.of(List.copyOf(history), sampling, options));
            if (Requests.fits(prepared, engine.contextCapacity())) {
                if (dropped > 0)
                    io.err()
                            .println(
                                    "context full: dropped the oldest "
                                            + (dropped == 1 ? "exchange" : dropped + " exchanges"));
                return prepared;
            }
            int oldest = options.systemPrompt == null ? 0 : 1;
            if (history.size() - oldest == 1) {
                Requests.checked(prepared, engine.contextCapacity()); // refuses; closes prepared
            }
            prepared.close();
            history.remove(oldest); // the oldest user turn, and its reply with it
            if (history.get(oldest).role().equals(Role.ASSISTANT)) history.remove(oldest);
            dropped++;
        }
    }
}
