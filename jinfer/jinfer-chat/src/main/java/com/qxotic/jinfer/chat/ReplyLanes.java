package com.qxotic.jinfer.chat;

import com.qxotic.toknroll.Tokenizer;
import java.util.Optional;

/**
 * The reply's two text lanes, token by token: the native {@link ReplyParser}'s (content vs
 * reasoning) when the model has a codec, else raw decoded text. Owns the ONE parser instance of a
 * generation - pre-fed the forced-call seed so a seeded reply parses whole - and {@link #finish}es
 * the structured message from exactly the parse that streamed; there is no second decode pass, so
 * the streamed fragments and the final message can never disagree.
 */
public final class ReplyLanes {

    private final ReplyParser parser; // null: raw text, single lane
    private final PendingUtf8 pending;
    private final StringBuilder rawText;
    private final Tokenizer tokenizer;
    private boolean reasoning;

    public ReplyLanes(Optional<ChatTemplate> template, Tokenizer tokenizer, int[] parserSeed) {
        this.parser = template.map(ChatTemplate::parser).orElse(null);
        this.pending = parser == null ? new PendingUtf8() : null;
        this.rawText = parser == null ? new StringBuilder() : null;
        this.tokenizer = tokenizer;
        if (parser != null) {
            for (int token : parserSeed) parser.feed(token);
        }
    }

    /** The text this token adds ("" while pending); {@link #reasoning()} tells its lane. */
    public String feed(int token) {
        if (parser != null) {
            String fragment = parser.feed(token);
            reasoning = parser.reasoning();
            return fragment;
        }
        String fragment = pending.add(tokenizer.decodeBytes(new int[] {token}), token).text();
        rawText.append(fragment);
        return fragment;
    }

    /** The lane of the LAST {@link #feed}ed fragment. */
    public boolean reasoning() {
        return reasoning;
    }

    /** The finished structured reply, from the same parse that streamed. */
    public Message finish() {
        return parser != null ? parser.finish() : new Message(Role.ASSISTANT, rawText.toString());
    }
}
