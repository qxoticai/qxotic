package com.qxotic.jinfer.chat;

import java.util.ArrayList;
import java.util.List;

/**
 * A reply's parts in emission order: text runs become one {@link Content.Text} or {@link
 * Content.Reasoning} each, a change of lane or a structured part closes the run, an empty run is
 * nothing.
 */
final class ContentBuilder {
    private final List<Content> parts = new ArrayList<>();
    private final StringBuilder run = new StringBuilder();
    private boolean reasoning;

    /** Prompt-owned tokens are not reply content: the turn's record carries them. */
    void discardSeedContent() {
        parts.clear();
        run.setLength(0);
    }

    void text(String value, boolean reasoning) {
        if (this.reasoning != reasoning) {
            closeRun();
            this.reasoning = reasoning;
        }
        run.append(value);
    }

    void add(Content part) {
        closeRun();
        parts.add(part);
    }

    List<Content> parts() {
        closeRun();
        return List.copyOf(parts);
    }

    private void closeRun() {
        if (run.isEmpty()) return;
        String text = run.toString();
        parts.add(reasoning ? new Content.Reasoning(text) : new Content.Text(text));
        run.setLength(0);
    }
}
