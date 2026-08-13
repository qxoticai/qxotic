package com.qxotic.jinfer.x.chat;

/**
 * A reply channel: the parser's innermost structural region, as a flat value. Well-known roots ship
 * as constants; a family may mint its own (gpt-oss's commentary) without touching this type.
 *
 * <p>Channels are flat by design - a claimed call drafted inside reasoning reports {@link
 * #TOOL_CALL} and the suspension underneath is parser state ({@link ReplyParser#pending()}), never
 * a field here. The full nesting survives in the finished message's {@code Content} tree; the
 * stream's deltas carry only the innermost label.
 */
public record Channel(String name) {

    public Channel {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("channel name " + name);
        }
    }

    /** The answer lane - what stop sequences arm on. */
    public static final Channel CONTENT = new Channel("content");

    /** The model's thinking, matching {@link Content.Reasoning}. */
    public static final Channel REASONING = new Channel("reasoning");

    /** A claimed tool call's payload; unclaimed call syntax stays {@link #CONTENT} text. */
    public static final Channel TOOL_CALL = new Channel("tool-call");

    /** Harmony-style narration between tool calls (gpt-oss). */
    public static final Channel COMMENTARY = new Channel("commentary");
}
