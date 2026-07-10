package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Media;

/**
 * One piece of a message's content: text, or a decoded media payload. The sealed {@link Media}
 * union carries the modality (image / audio / video / future), so adding a modality never touches
 * this interface or {@link ChatTemplate}.
 */
public sealed interface Part {

    /**
     * Untrusted conversation text. Templates MUST tokenize it plainly (no special-token
     * recognition), so content can never mint control tokens.
     */
    record Text(String text) implements Part {
        public Text {
            if (text == null) throw new IllegalArgumentException("null text");
        }
    }

    /**
     * A decoded media payload, positioned structurally (by its place in the part list, never by
     * markers parsed out of text).
     */
    record Blob(Media media) implements Part {
        public Blob {
            if (media == null) throw new IllegalArgumentException("null media");
        }
    }

    /**
     * A tool call the assistant emitted, or that the caller replays in a history turn. {@code
     * arguments} is the parsed argument object (the model's detector produces it structurally, so
     * the wire layer is the only place that ever serializes it back to a JSON string). {@code id}
     * correlates the call with its {@link ToolResult}; it may be blank for models that do not mint
     * one, in which case the caller assigns it.
     */
    record ToolCall(String id, String name, java.util.Map<String, Object> arguments)
            implements Part {
        public ToolCall {
            if (name == null || name.isEmpty())
                throw new IllegalArgumentException("empty tool name");
            id = id == null ? "" : id;
            arguments = arguments == null ? java.util.Map.of() : java.util.Map.copyOf(arguments);
        }
    }

    /**
     * The result of a tool call, replayed by the caller in the next turn. {@code callId} matches
     * the {@link ToolCall#id()} it answers; {@code name} is the tool that produced it.
     */
    record ToolResult(String callId, String name, String content) implements Part {
        public ToolResult {
            if (content == null) throw new IllegalArgumentException("null tool result content");
        }
    }
}
