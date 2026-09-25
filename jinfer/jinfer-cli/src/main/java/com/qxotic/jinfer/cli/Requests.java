package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.llm.Sampling;
import java.io.IOException;
import java.util.List;

/** Shared request construction and prompt-size validation for chat and instruct. */
final class Requests {

    private Requests() {}

    static ChatEngine.Prepared prepare(
            ChatEngine engine, List<Message> messages, Sampling sampling, Options options)
            throws IOException {
        try {
            return checked(
                    engine.prepare(of(messages, sampling, options)), engine.contextCapacity());
        } catch (IllegalArgumentException | UnsupportedOperationException e) {
            throw Main.failure("cannot prepare response", e);
        }
    }

    static ChatEngine.Prepared checked(ChatEngine.Prepared prepared, int capacity)
            throws IOException {
        if (prepared.promptTokens() > capacity) {
            try (prepared) {
                throw new IOException(
                        "prompt needs "
                                + prepared.promptTokens()
                                + " tokens but context capacity is "
                                + capacity
                                + "; shorten the prompt or increase --context-capacity");
            }
        }
        return prepared;
    }

    static ChatEngine.Request of(List<Message> messages, Sampling sampling, Options options) {
        return ChatEngine.Request.builder(messages, sampling)
                .thinking(options.think)
                .maxOutputTokens(options.maxOutputTokens)
                .maxReasoningTokens(options.maxReasoningTokens)
                .reasoningCutoffMessage(options.reasoningCutoffMessage)
                .build();
    }
}
