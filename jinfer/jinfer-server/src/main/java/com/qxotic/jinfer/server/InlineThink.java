package com.qxotic.jinfer.server;

/**
 * The reasoning-to-inline {@code <think>...</think>} content projection, used by the server's
 * FragmentRouter (reasoning_format "none") and the CLI display text. Stateful per reply: the open
 * bracket attaches to the first reasoning fragment, the close bracket to the first content fragment
 * after the span; an unterminated span (generation ended thinking) stays unclosed, matching the raw
 * token stream.
 */
final class InlineThink {

    private boolean open;

    /** The fragment as it should appear inline, brackets attached at channel transitions. */
    String project(String fragment, boolean reasoning) {
        if (reasoning) {
            if (!open) {
                open = true;
                return "<think>" + fragment;
            }
            return fragment;
        }
        if (open) {
            open = false;
            return "</think>" + fragment;
        }
        return fragment;
    }
}
