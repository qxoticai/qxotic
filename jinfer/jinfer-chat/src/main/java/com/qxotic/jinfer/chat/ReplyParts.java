package com.qxotic.jinfer.chat;

import java.util.ArrayList;
import java.util.List;

/**
 * A generated reply flattened into the three things every integration needs: the content lane, the
 * reasoning lane, and the tool calls. The walk itself is not framework work - it is the parsed
 * reply's own structure (nested {@link Part.Reasoning} spans, families whose call syntax carries no
 * id) - so it happens once here and each integration only builds its own message type.
 *
 * @param toolCalls arguments already in the canonical JSON both frameworks want on the wire
 */
public record ReplyParts(String text, String thinking, List<ToolCall> toolCalls) {

    /**
     * One call: the id (minted when the family's syntax has no slot for one) and canonical args.
     */
    public record ToolCall(String id, String name, String argumentsJson) {}

    public static ReplyParts of(Message reply) {
        StringBuilder text = new StringBuilder();
        StringBuilder thinking = new StringBuilder();
        List<ToolCall> calls = new ArrayList<>();
        collect(reply.content(), text, thinking, calls, false);
        return new ReplyParts(text.toString(), thinking.toString(), List.copyOf(calls));
    }

    private static void collect(
            List<Part> parts,
            StringBuilder text,
            StringBuilder thinking,
            List<ToolCall> calls,
            boolean inReasoning) {
        for (Part part : parts) {
            switch (part) {
                case Part.Text t -> (inReasoning ? thinking : text).append(t.text());
                case Part.Reasoning r -> collect(r.content(), text, thinking, calls, true);
                case Part.ToolCall c ->
                        // pythonic syntaxes carry no call ids: mint stable positional ones (what
                        // Ollama's server does); ids never render back into the prompt (the
                        // template's call syntax has no id slot), so echoes stay byte-identical
                        calls.add(
                                new ToolCall(
                                        c.id().isEmpty() ? "call_" + calls.size() : c.id(),
                                        c.name(),
                                        JsonCodec.stringify(c.arguments())));
                default -> {} // ToolResult/Blob never appear in a generated reply
            }
        }
    }
}
