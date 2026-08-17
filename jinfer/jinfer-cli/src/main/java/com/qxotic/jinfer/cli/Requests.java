package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.llm.Sampling;
import java.time.Duration;
import java.util.List;

/**
 * The one request shape the CLI ever sends: the user's flags and nothing else - no tools, no
 * grammar, no stop sequences, no deadline. Centralized because the {@link ChatEngine.Request}
 * record is positional: two modes spelling the same seven fixed arguments is how a transposed pair
 * of nulls ships.
 */
final class Requests {

    private Requests() {}

    static ChatEngine.Request of(List<Message> messages, Sampling sampling, Options options) {
        return new ChatEngine.Request(
                messages,
                List.of(),
                options.think(),
                options.maxOutputTokens(),
                options.reasoningBudget(),
                options.reasoningBudgetMessage(),
                Duration.ZERO,
                sampling,
                null,
                ChatEngine.ForcedTool.NONE,
                List.of(),
                null);
    }
}
