package com.qxotic.jinfer.x.chat;

import java.util.ArrayList;
import java.util.List;

/** Every input that determines a model's chat prompt. */
public record Conversation(
        List<Message> messages, List<Tool> tools, boolean thinking, String reasoningEffort) {
    public Conversation {
        messages = List.copyOf(messages);
        tools = List.copyOf(tools);
        reasoningEffort = reasoningEffort == null ? "" : reasoningEffort;
    }

    public Conversation(List<Message> messages) {
        this(messages, List.of(), true, "");
    }

    public Conversation append(Message message) {
        ArrayList<Message> copy = new ArrayList<>(messages);
        copy.add(message);
        return new Conversation(copy, tools, thinking, reasoningEffort);
    }
}
