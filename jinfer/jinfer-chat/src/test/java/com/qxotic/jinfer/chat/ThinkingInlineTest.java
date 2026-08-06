package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/**
 * The markers go at CHANNEL TRANSITIONS, which is the whole reason this holds state: a fragment
 * cannot tell from its own content whether the span is already open.
 */
final class ThinkingInlineTest {

    @Test
    void theOpenMarkerAttachesToTheFirstReasoningFragmentOnly() {
        Thinking.Inline inline = new Thinking.Inline();
        assertEquals("<think>weighing", inline.project("weighing", true));
        assertEquals(" options", inline.project(" options", true));
    }

    @Test
    void theCloseMarkerAttachesToTheFirstContentFragmentAfterTheSpan() {
        Thinking.Inline inline = new Thinking.Inline();
        inline.project("weighing", true);
        assertEquals("</think>Paris.", inline.project("Paris.", false));
        assertEquals(" It is.", inline.project(" It is.", false));
    }

    @Test
    void contentBeforeAnyReasoningIsUntouched() {
        assertEquals("plain", new Thinking.Inline().project("plain", false));
    }

    /** Generation ended mid-thought: the raw token stream had no close either. */
    @Test
    void anUnterminatedSpanStaysUnclosed() {
        Thinking.Inline inline = new Thinking.Inline();
        assertEquals("<think>weighing", inline.project("weighing", true));
        // no content fragment ever arrives - nothing appends </think> on its own
    }

    @Test
    void aSecondSpanOpensAgain() {
        Thinking.Inline inline = new Thinking.Inline();
        inline.project("first", true);
        inline.project("answer", false);
        assertEquals("<think>second", inline.project("second", true));
        assertEquals("</think>again", inline.project("again", false));
    }
}
