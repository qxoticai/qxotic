package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.ContentKey;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** One ordered piece of a chat message. */
public sealed interface Content {

    /** Plain, untrusted text. */
    record Text(String text) implements Content {
        public Text {
            Objects.requireNonNull(text, "text");
        }
    }

    /**
     * A run of the model's reasoning. A message holds its reply folded in emission order, so a call
     * claimed while thinking is a {@link ToolCall} between two reasoning runs.
     */
    record Reasoning(String text) implements Content {
        public Reasoning {
            Objects.requireNonNull(text, "text");
        }
    }

    /** A model-produced tool invocation. */
    record ToolCall(String id, String name, Map<String, Object> arguments) implements Content {
        public ToolCall {
            id = id == null ? "" : id;
            if (name == null || name.isEmpty())
                throw new IllegalArgumentException("empty tool name");
            arguments = JsonValues.object(arguments == null ? Map.of() : arguments);
        }
    }

    /** A caller-authored result correlated to a tool call. */
    record ToolResult(String callId, String text) implements Content {
        public ToolResult {
            callId = callId == null ? "" : callId;
            Objects.requireNonNull(text, "text");
        }
    }

    /**
     * Decoded media at its exact position in a message: an image, audio or video the caller gives
     * the model, or, in a reply, the model's own speech, which a later prompt replays through the
     * turn's {@link Verbatim} record. The key names a caller's media to the prompt cache (null:
     * unkeyed, encoded every time). The caption is what the media says in words: a reply's speech
     * carries its transcript, the words it spoke; empty when it spoke none, or for a caller's
     * media.
     */
    record Media(com.qxotic.jinfer.media.Media media, ContentKey key, Optional<String> caption)
            implements Content {
        public Media {
            Objects.requireNonNull(media, "media");
            Objects.requireNonNull(caption, "caption");
        }

        public Media(com.qxotic.jinfer.media.Media media, ContentKey key) {
            this(media, key, Optional.empty());
        }

        public Media(com.qxotic.jinfer.media.Media media) {
            this(media, null);
        }
    }
}
