package com.qxotic.jinfer.x.cli;

import com.qxotic.jinfer.x.chat.ChatEngine;
import com.qxotic.jinfer.x.chat.Message;
import com.qxotic.jinfer.x.llm.Sampling;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * The interactive {@code --chat} loop. ONE running conversation: each turn appends the user's
 * message, prepares, completes with the terminal sink, and appends the parsed reply - the engine's
 * own prompt cache carries the KV across turns (the hot session strictly extends), which is what
 * the old CLI hand-rolled with longest-common-prefix tracking and state rebuilds.
 */
final class Chat {

    private Chat() {}

    static void run(ChatEngine engine, Sampling sampling, Options options) throws IOException {
        List<Message> history = new ArrayList<>();
        if (options.systemPrompt() != null) {
            history.add(Message.system(options.systemPrompt()));
        }
        try (BufferedReader reader =
                new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                String userText = reader.readLine();
                if (userText == null || "/quit".equals(userText) || "/exit".equals(userText)) {
                    break;
                }
                if (userText.isBlank()) {
                    continue; // an empty turn would scaffold a reply to nothing
                }
                if ("/context".equals(userText)) {
                    System.out.printf(
                            "context: capacity %d tokens, %s%n",
                            options.contextCapacity(), engine.sessionStats());
                    continue;
                }
                history.add(Message.user(userText));
                ChatEngine.Completion completion;
                Turn turn;
                try (ChatEngine.Prepared prepared =
                        engine.prepare(Requests.of(List.copyOf(history), sampling, options))) {
                    turn = Turn.start(engine.loaded().tokenizer(), prepared, options);
                    completion = engine.complete(prepared, turn);
                }
                turn.finish(completion, options.contextCapacity());
                if (completion.reply() != null) {
                    // the parser's structured message (verbatim ids): the codec's verbatim splice
                    // keeps generated turns inside the cache's common prefix
                    history.add(completion.reply());
                }
            }
        }
    }
}
