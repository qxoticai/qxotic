package com.qxotic.jinfer.chat;

/**
 * Signals that a native template cannot encode a conversation exactly. A port's signal to {@link
 * ChatEngine} only: the engine answers it with the Jinja whole-render, or with an {@link
 * UnsupportedOperationException} where no whole-render will do, so it never reaches a caller.
 */
public final class UnsupportedConversation extends RuntimeException {
    public UnsupportedConversation(String reason) {
        super(reason);
    }
}
