package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.media.Media;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** A message's readers over its parts: the words, the reasoning joined, the speech. */
class MessageTest {

    @Test
    void reasoningJoinsTheRunsAsTextJoinsTheText() {
        Message m =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Content.Reasoning("first "),
                                new Content.ToolCall("", "f", Map.of()),
                                new Content.Reasoning("second"),
                                new Content.Text("the "),
                                new Content.Text("answer")));
        assertEquals("first second", m.reasoning(), "runs joined in order, the call skipped");
        assertEquals("the answer", m.text(), "text joined, reasoning left out");
    }

    @Test
    void aMessageWithoutReasoningReadsEmptyNotNull() {
        assertEquals("", Message.assistant("plain").reasoning());
        assertEquals("", Message.user("question").reasoning());
        assertEquals(
                "", new Message(Role.ASSISTANT, List.of(new Content.Reasoning(""))).reasoning());
    }

    @Test
    void speechIsTheAudioTheMessageHolds() {
        Media.Audio clip = new Media.Audio(new float[240], 24_000, 1);
        Message spoken =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Content.Text("Paris."),
                                new Content.Media(clip, null, Optional.of("Paris."))));
        assertTrue(spoken.speech().isPresent());
        assertTrue(spoken.speech().get() == clip);
        assertTrue(Message.assistant("Paris.").speech().isEmpty());
    }
}
