package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Tool;
import java.util.ArrayList;
import java.util.List;

/**
 * A cached-prompt view's prefix: the turns and tools every request on that view starts with,
 * prefilled once into the engine's block tree and restored (not recomputed) per request.
 *
 * <p>Converted to jinfer types ONCE when the view is created - media decoded there, not per request
 * - and immutable afterwards, so branching a view is {@link #merge} on the parent's prefix.
 */
record CachedPrompt(List<Message> messages, List<Tool> tools) {

    /** The base model's prefix: none. A request on it is exactly what the caller sent. */
    static final CachedPrompt NONE = new CachedPrompt(List.of(), List.of());

    CachedPrompt {
        messages = List.copyOf(messages);
        tools = List.copyOf(tools);
    }

    boolean isEmpty() {
        return messages.isEmpty() && tools.isEmpty();
    }

    /** This prefix extended - what branching a view means: parent first, then the new turns. */
    CachedPrompt merge(List<Message> moreMessages, List<Tool> moreTools) {
        List<Message> merged = new ArrayList<>(messages);
        merged.addAll(moreMessages);
        List<Tool> welded = new ArrayList<>(tools);
        if (moreTools != null) welded.addAll(moreTools);
        return new CachedPrompt(merged, welded);
    }

    /** This prefix as a conversation to define into the block tree. */
    Conversation conversation(boolean thinking) {
        return new Conversation(messages, tools, thinking, "");
    }

    /**
     * Request-over-defaults, THE tool precedence rule: a request that STATES a tool set -
     * explicitly-none included - is served with its own; {@code null} (unstated) falls to this
     * prefix's welded default, empty on the base model.
     */
    List<Tool> resolveTools(List<Tool> stated) {
        return stated == null ? tools : stated;
    }

    /**
     * Whether the prepaid prefill serves this effective set: exactly the welded frame, compared as
     * the rendered bytes that were prefilled. False on the base model - it has no prepaid frame.
     */
    boolean serves(List<Tool> effective) {
        return !isEmpty() && effective.equals(tools);
    }

    /** The once-per-view tools-override warning, worded here so every adapter says the same. */
    String toolsOverrideWarning(List<Tool> requested) {
        return "WARNING: this cached-prompt view's welded tools "
                + names(tools)
                + " were overridden by the request's "
                + names(requested)
                + " - served correctly but UNCACHED (full prefill). Weld this set with"
                + " withCachedPrompt(...), or use the base model for per-request tools."
                + " (warned once per view)";
    }

    private static List<String> names(List<Tool> tools) {
        return tools.stream().map(Tool::name).toList();
    }
}
