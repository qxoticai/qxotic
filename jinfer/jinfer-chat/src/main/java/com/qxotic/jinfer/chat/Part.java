package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Media;
import com.qxotic.toknroll.IntSequence;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One piece of a message's content - a span tree, not a flat run of text. Two families:
 *
 * <p><b>Model-produced</b> ({@link Text}, {@link Reasoning}, {@link ToolCall}): what {@link
 * ReplyParser} emits. Each may carry {@code verbatim} - the exact generated token ids of the part's
 * CONTENT PAYLOAD, never its delimiters (markers are template-emitted and may differ between
 * generation and history). Verbatim ids give byte-drift-proof KV reuse across a tool loop; {@code
 * null} means "re-tokenize from text" (caller-authored content, or history replayed from
 * structure).
 *
 * <p><b>Caller-authored</b> ({@link ToolResult}, {@link Blob}): decode never yields these, and they
 * never carry verbatim ids. The sealed {@link Media} union carries the modality (image / audio /
 * video / future), so adding a modality never touches this interface.
 */
public sealed interface Part {

    /**
     * Untrusted conversation text. Templates MUST tokenize it plainly (no special-token
     * recognition), so content can never mint control tokens.
     */
    record Text(String text, IntSequence verbatim) implements Part {
        public Text {
            if (text == null) throw new IllegalArgumentException("null text");
        }

        /** Caller-authored text: no verbatim ids, re-tokenized by the template. */
        public Text(String text) {
            this(text, null);
        }
    }

    /**
     * A reasoning span (the model's think channel). A span TREE node: {@code content} holds the
     * ordered parts produced inside the span - text, and tool calls when the model calls tools
     * while thinking.
     */
    record Reasoning(List<Part> content, IntSequence verbatim) implements Part {
        public Reasoning {
            content = List.copyOf(content);
        }
    }

    /**
     * A tool call the assistant emitted, or that the caller replays in a history turn. {@code
     * arguments} is the parsed argument object (the model's decoder produces it structurally, so
     * the wire layer is the only place that ever serializes it back to a JSON string). {@code id}
     * correlates a call with its result; it may be blank for models that do not mint one, in which
     * case the caller assigns it. {@code verbatim} is the payload between the call markers
     * (exclusive), when the decoder can attribute it to exactly this call.
     */
    record ToolCall(String id, String name, Map<String, Object> arguments, IntSequence verbatim)
            implements Part {
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

        public ToolCall(String id, String name, Map<String, Object> arguments) {
            this(id, name, arguments, null);
        }
    }

    /**
     * The caller-supplied result of a tool call, correlated by {@code callId}. The native agentic
     * representation ({@code [Text, ToolCall, ToolResult, Text]} is one ordered part list); a
     * {@code tool}-role message remains the equivalent wire-level shape.
     */
    record ToolResult(String callId, String text) implements Part {
        public ToolResult {
            if (text == null) throw new IllegalArgumentException("null text");
            callId = callId == null ? "" : callId;
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
}
