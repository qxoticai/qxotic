package com.qxotic.jinfer.server;

import com.qxotic.jinfer.chat.Part;
import java.util.function.Consumer;

/**
 * The shared Reasoning-delta to inline {@code <think>...</think>} content-text projection, used by
 * the server's DeltaRouter (reasoning_format "none") and the CLI display text. Stateful per reply:
 * the open bracket is emitted once, at the first reasoning fragment; the empty-Reasoning span-close
 * delta closes the span (an empty span still brackets, and finish() closes nothing - exactly the
 * decoder's span-close semantics). Fragments are emitted through {@code content} with the same
 * boundaries the deltas arrived with (markers separate from payload text).
 */
final class InlineThink {

    private boolean open;

    /** Projects one Reasoning delta to inline content-text fragments. */
    void project(Part.Reasoning reasoning, Consumer<String> content) {
        if (reasoning.content().isEmpty()) { // the span-close event
            if (!open) content.accept("<think>"); // empty span still brackets
            open = false;
            content.accept("</think>");
            return;
        }
        if (!open) {
            open = true;
            content.accept("<think>");
        }
        StringBuilder frag = new StringBuilder();
        for (Part inner : reasoning.content()) {
            if (inner instanceof Part.Text t) frag.append(t.text());
        }
        content.accept(frag.toString());
    }
}
