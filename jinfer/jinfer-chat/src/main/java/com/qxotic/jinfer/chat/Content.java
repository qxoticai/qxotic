package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.ContentKey;
import com.qxotic.toknroll.IntSequence;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** One ordered piece of a chat message. */
public sealed interface Content {

    /** Plain, untrusted text. Non-empty verbatim ids preserve an exact generated payload. */
    record Text(String text, IntSequence verbatim) implements Content {
        public Text {
            Objects.requireNonNull(text, "text");
            verbatim = verbatim == null ? IntSequence.empty() : verbatim;
        }

        public Text(String text) {
            this(text, IntSequence.empty());
        }
    }

    /** The model's reasoning channel, including any structured content emitted inside it. */
    record Reasoning(List<Content> content, IntSequence verbatim) implements Content {
        public Reasoning {
            content = List.copyOf(content);
            verbatim = verbatim == null ? IntSequence.empty() : verbatim;
        }

        public String text() {
            StringBuilder out = new StringBuilder();
            for (Content part : content) if (part instanceof Text t) out.append(t.text());
            return out.toString();
        }
    }

    /** A model-produced tool invocation. */
    record ToolCall(String id, String name, Map<String, Object> arguments, IntSequence verbatim)
            implements Content {
        public ToolCall {
            id = id == null ? "" : id;
            if (name == null || name.isEmpty())
                throw new IllegalArgumentException("empty tool name");
            arguments = JsonValues.object(arguments == null ? Map.of() : arguments);
            verbatim = verbatim == null ? IntSequence.empty() : verbatim;
        }

        public ToolCall(String id, String name, Map<String, Object> arguments) {
            this(id, name, arguments, IntSequence.empty());
        }
    }

    /** A caller-authored result correlated to a tool call. */
    record ToolResult(String callId, String text) implements Content {
        public ToolResult {
            callId = callId == null ? "" : callId;
            Objects.requireNonNull(text, "text");
        }
    }

    /** Decoded media at its exact position in a message. */
    record Media(com.qxotic.jinfer.Media value, ContentKey contentKey) implements Content {
        public Media {
            Objects.requireNonNull(value, "value");
        }

        public Media(com.qxotic.jinfer.Media value) {
            this(value, null);
        }
    }
}
