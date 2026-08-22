package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/** The {@link ChatTemplate#thinkMarkers()} seam: defaults and the marker record's contract. */
class ChatTemplateTest {

    private static final ChatTemplate STUB =
            (conversation, batchCapacity, sink) -> {
                throw new UnsupportedOperationException();
            };

    @Test
    void theDefaultThinkMarkersAreTheGenericPair() {
        assertSame(ChatTemplate.ThinkMarkers.GENERIC, STUB.thinkMarkers());
        assertEquals("<think>", ChatTemplate.ThinkMarkers.GENERIC.open());
        assertEquals("</think>", ChatTemplate.ThinkMarkers.GENERIC.close());
    }

    @Test
    void markerSpellingsAreRequired() {
        assertThrows(
                NullPointerException.class, () -> new ChatTemplate.ThinkMarkers(null, "</think>"));
        assertThrows(
                NullPointerException.class, () -> new ChatTemplate.ThinkMarkers("<think>", null));
    }
}
