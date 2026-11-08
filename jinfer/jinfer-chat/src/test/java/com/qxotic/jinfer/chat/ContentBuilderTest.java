package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** The folding both reply parsers share: runs by lane, structured parts between them. */
class ContentBuilderTest {

    private static final Content.ToolCall CALL = new Content.ToolCall("", "f", Map.of("x", 1));

    @Test
    void aChangeOfLaneClosesTheRun() {
        ContentBuilder b = new ContentBuilder();
        b.text("Hel", true);
        b.text("lo", true);
        b.text("Hi", false);
        b.text("!", false);
        assertEquals(List.of(new Content.Reasoning("Hello"), new Content.Text("Hi!")), b.parts());
    }

    @Test
    void aStructuredPartClosesTheRunAndSitsWhereItWasEmitted() {
        ContentBuilder b = new ContentBuilder();
        b.text("think", true);
        b.add(CALL);
        b.text("more", true);
        b.text("answer", false);
        assertEquals(
                List.of(
                        new Content.Reasoning("think"),
                        CALL,
                        new Content.Reasoning("more"),
                        new Content.Text("answer")),
                b.parts());
    }

    @Test
    void anEmptyRunIsNothing() {
        ContentBuilder b = new ContentBuilder();
        b.text("", true);
        b.text("", false);
        b.add(CALL);
        b.text("", true);
        assertEquals(List.of(CALL), b.parts());
        assertEquals(List.of(), new ContentBuilder().parts());
    }

    @Test
    void seedContentIsDiscardedWhole() {
        ContentBuilder b = new ContentBuilder();
        b.text("prompt-owned", true);
        b.add(CALL);
        b.discardSeedContent();
        b.text("mine", true);
        assertEquals(List.of(new Content.Reasoning("mine")), b.parts());
    }

    @Test
    void partsIsASnapshotTheBuilderKeepsGrowingPast() {
        ContentBuilder b = new ContentBuilder();
        b.text("a", false);
        List<Content> first = b.parts();
        b.text("b", false);
        assertEquals(List.of(new Content.Text("a")), first);
        assertEquals(List.of(new Content.Text("a"), new Content.Text("b")), b.parts());
    }
}
