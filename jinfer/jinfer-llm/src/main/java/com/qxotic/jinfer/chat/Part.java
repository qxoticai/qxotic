package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Media;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

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
     * correlates a call with its result (a {@code tool}-role turn); it may be blank for models that
     * do not mint one, in which case the caller assigns it.
     */
    record ToolCall(String id, String name, Map<String, Object> arguments) implements Part {
        public ToolCall {
            if (name == null || name.isEmpty())
                throw new IllegalArgumentException("empty tool name");
            id = id == null ? "" : id;
            // Insertion order is load-bearing: the pythonic render emits arguments in order and the
            // model was trained on the order they appeared, so a defensive copy must preserve it
            // (Map.copyOf does not).
            arguments =
                    arguments == null
                            ? Map.of()
                            : Collections.unmodifiableMap(new LinkedHashMap<>(arguments));
        }
    }
}
