package com.qxotic.jinfer.x.chat;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Every input that determines a model's chat prompt.
 *
 * <p>One structural law is enforced at construction: every tool RESULT answers a tool CALL made
 * earlier in the same conversation. An orphan result (a memory trim that dropped the call, a
 * hand-built transcript) would otherwise render as a ghost exchange the model never made - silent
 * context pollution that the prompt cache then commits. Id-carrying results match their call id;
 * id-less results (families without call ids) require only that some call preceded them. Unanswered
 * calls are legal - that is the live hand-off to the caller's tool executor.
 */
public record Conversation(
        List<Message> messages, List<Tool> tools, boolean thinking, String reasoningEffort) {
    public Conversation {
        messages = List.copyOf(messages);
        tools = List.copyOf(tools);
        reasoningEffort = reasoningEffort == null ? "" : reasoningEffort;
        requireMatchedResults(messages);
    }

    private static void requireMatchedResults(List<Message> messages) {
        // ids never render into the prompt (no family's call syntax has an id slot), so the law
        // is about the WIRE: a result with no call anywhere before it is a ghost turn. Precise
        // ids match precisely where both sides carry them; id-less families (the call carries ""
        // and an adapter mints "call_0"-style positional ids its results then echo back) match by
        // presence alone.
        Set<String> pending = new HashSet<>();
        boolean anyCall = false, anyIdlessCall = false;
        for (Message message : messages) {
            for (Content part : message.content()) {
                if (part instanceof Content.ToolCall call) {
                    anyCall = true;
                    if (call.id().isEmpty()) anyIdlessCall = true;
                    else pending.add(call.id());
                } else if (part instanceof Content.ToolResult result) {
                    String id = result.callId();
                    boolean matched = !id.isEmpty() && pending.remove(id);
                    if (!matched) matched = anyIdlessCall || (id.isEmpty() && anyCall);
                    if (!matched) {
                        throw new IllegalArgumentException(
                                "tool result '"
                                        + result.callId()
                                        + "' answers no tool call in this conversation - an orphan"
                                        + " renders as a ghost exchange the model never made."
                                        + " Replay the assistant message carrying the call first,"
                                        + " or drop the result (a trimmed memory must evict both"
                                        + " sides of the round)");
                    }
                }
            }
        }
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
