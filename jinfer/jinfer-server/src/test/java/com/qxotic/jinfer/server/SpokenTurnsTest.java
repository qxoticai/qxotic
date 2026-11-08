package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Verbatim;
import com.qxotic.jinfer.media.Media;
import com.qxotic.toknroll.IntSequence;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** The spoken turns a request may refer to by {@code audio.id}: kept, found, evicted, refused. */
class SpokenTurnsTest {

    private static final Verbatim RECORD =
            new Verbatim(
                    new ContentKey("test-model"),
                    IntSequence.of(5, Verbatim.FRAME, 6),
                    IntSequence.of(new int[8]));

    private static Message spoken(String words) {
        return new Message(
                Role.ASSISTANT,
                List.of(
                        new Content.Text(words),
                        new Content.Media(
                                new Media.Audio(new float[480], 24_000, 1),
                                null,
                                Optional.of(words))),
                RECORD);
    }

    @Test
    void idsAreFreshAndSpelledLikeOpenAis() {
        String a = SpokenTurns.newId(), b = SpokenTurns.newId();
        assertTrue(a.startsWith("audio_"), a);
        assertNotEquals(a, b);
    }

    @Test
    void aKeptTurnIsItsWordsAndItsRecordWithoutTheSpeech() {
        SpokenTurns turns = new SpokenTurns();
        String id = SpokenTurns.newId();
        long now = System.currentTimeMillis() / 1000;
        long expiresAt = turns.put(id, spoken("Paris."));
        assertTrue(expiresAt >= now + SpokenTurns.LIFETIME.toSeconds() - 1, "expires in an hour");
        Message kept = turns.get(id);
        assertEquals("Paris.", kept.text());
        assertTrue(kept.speech().isEmpty(), "the waveform is not kept, the record replays it");
        assertTrue(kept.verbatim() == RECORD, "the record is the one given");
        assertEquals(1, kept.content().size(), "the words alone");
    }

    @Test
    void anUnknownIdIsRefusedInWords() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new SpokenTurns().get("audio_nobody"));
        assertTrue(e.getMessage().contains("unknown or expired"), e.getMessage());
    }

    /** Over capacity the least recently USED turn goes, so a live conversation keeps its ids. */
    @Test
    void overCapacityTheLeastRecentlyUsedTurnGoes() {
        SpokenTurns turns = new SpokenTurns();
        String first = "audio_first", second = "audio_second";
        turns.put(first, spoken("one"));
        turns.put(second, spoken("two"));
        for (int i = 2; i < SpokenTurns.CAPACITY; i++) turns.put("audio_" + i, spoken("n"));
        turns.get(first); // a conversation echoing its turn keeps it fresh
        turns.put("audio_overflow", spoken("more"));
        assertEquals("one", turns.get(first).text(), "used, so kept");
        assertThrows(IllegalArgumentException.class, () -> turns.get(second), "eldest unused");
        assertEquals("more", turns.get("audio_overflow").text());
    }
}
