package com.qxotic.jinfer.chat;

/**
 * Thrown by {@link ChatTemplate#encode} when the template cannot frame a conversation byte-exactly
 * with the model's own reference template (an unsupported part kind, tool framing it does not
 * model). The caller's routing signal: catch it and fall back to the whole-render Jinja path. Each
 * punt on a ported model is a signal to extend the port, not a steady state.
 */
public final class UnsupportedConversation extends RuntimeException {

    public UnsupportedConversation(String reason) {
        super(reason);
    }
}
