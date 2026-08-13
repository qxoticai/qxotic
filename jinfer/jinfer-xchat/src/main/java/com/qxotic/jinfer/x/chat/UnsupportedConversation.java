package com.qxotic.jinfer.x.chat;

/** Signals that a native template cannot encode a conversation exactly. */
public final class UnsupportedConversation extends RuntimeException {
    public UnsupportedConversation(String reason) {
        super(reason);
    }
}
