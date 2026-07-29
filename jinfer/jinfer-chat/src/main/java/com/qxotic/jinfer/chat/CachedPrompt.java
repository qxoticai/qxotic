package com.qxotic.jinfer.chat;

import java.util.ArrayList;
import java.util.List;

/**
 * A cached-prompt view's prefix: the turns and tools every request on that view starts with,
 * prefilled once into the engine's block tree and restored (not recomputed) per request.
 *
 * <p>Converted to jinfer types ONCE when the view is created - media decoded there, not per request
 * - and immutable afterwards, so branching a view is {@link #merge} on the parent's prefix.
 */
public record CachedPrompt(List<Message> messages, List<Tool> tools) {

    /** The base model's prefix: none. A request on it is exactly what the caller sent. */
    public static final CachedPrompt NONE = new CachedPrompt(List.of(), List.of());

    public CachedPrompt {
        messages = List.copyOf(messages);
        tools = List.copyOf(tools);
    }

    public boolean isEmpty() {
        return messages.isEmpty() && tools.isEmpty();
    }

    /** This prefix extended - what branching a view means: parent first, then the new turns. */
    public CachedPrompt merge(List<Message> moreMessages, List<Tool> moreTools) {
        List<Message> merged = new ArrayList<>(messages);
        merged.addAll(moreMessages);
        List<Tool> welded = new ArrayList<>(tools);
        if (moreTools != null) welded.addAll(moreTools);
        return new CachedPrompt(merged, welded);
    }

    /** This prefix as a conversation to define into the block tree. */
    public Conversation conversation(boolean thinking) {
        return new Conversation(messages, tools, thinking, "");
    }
}
