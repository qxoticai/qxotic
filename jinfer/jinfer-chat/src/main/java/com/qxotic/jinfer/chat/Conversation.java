package com.qxotic.jinfer.chat;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Every input that determines a model's chat prompt.
 *
 * <p>One structural law is enforced at construction: every tool RESULT answers a tool CALL made
 * earlier in the same conversation. An orphan result (a memory trim that dropped the call, a
 * hand-built transcript) would otherwise render as a ghost exchange the model never made - silent
 * context pollution that the prompt cache then commits. Each result answers exactly one call: its
 * id's where both carry one, else the oldest unanswered call (families without call ids, and the
 * ids adapters mint for them). Unanswered calls are legal - that is the live hand-off to the
 * caller's tool executor.
 */
public record Conversation(List<Message> messages, List<Tool> tools, boolean thinking) {
    public Conversation {
        messages = List.copyOf(messages);
        tools = List.copyOf(tools);
        requireMatchedResults(messages);
    }

    private static void requireMatchedResults(List<Message> messages) {
        // ids never render into the prompt (no family's call syntax has an id slot), so the law
        // is about the WIRE: a result with no unanswered call before it is a ghost turn. An
        // id-less call (the call carries "" and an adapter mints "call_0"-style ids its results
        // echo back) is answered by any one result.
        Set<String> pending = new LinkedHashSet<>(); // unanswered calls with an id, oldest first
        int idless = 0; // unanswered calls without one
        for (Message message : messages) {
            for (Content part : message.content()) {
                if (part instanceof Content.ToolCall call) {
                    if (call.id().isEmpty()) idless++;
                    else pending.add(call.id());
                } else if (part instanceof Content.ToolResult result) {
                    String id = result.callId();
                    if (!id.isEmpty() && pending.remove(id)) continue;
                    if (idless > 0) idless--;
                    else if (id.isEmpty() && !pending.isEmpty())
                        pending.remove(pending.iterator().next());
                    else {
                        throw new IllegalArgumentException(
                                "tool result '"
                                        + result.callId()
                                        + "' answers no open tool call in this conversation - an"
                                        + " orphan renders as a ghost exchange the model never"
                                        + " made. Replay the assistant message carrying the call"
                                        + " first, or drop the result (a trimmed memory must evict"
                                        + " both sides of the round)");
                    }
                }
            }
        }
    }

    public Conversation(List<Message> messages) {
        this(messages, List.of(), true);
    }

    public Conversation append(Message message) {
        ArrayList<Message> copy = new ArrayList<>(messages);
        copy.add(message);
        return new Conversation(copy, tools, thinking);
    }
}
