package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.testkit.TestLanguageModel;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** The record of a generated turn, and how a prompt writer replays it. */
final class VerbatimTest {
    private static final Tokenizer TOKENIZER = TestLanguageModel.TOKENIZER; // a byte an id
    private static final ContentKey MODEL = new ContentKey("verbatim-test");

    @Test
    void codesComeInWholeFramesOrNotAtAll() {
        Verbatim text = new Verbatim(MODEL, IntSequence.of(1, 2, 3), IntSequence.empty());
        assertEquals(0, text.frames());
        assertEquals(0, text.codebooks());

        Verbatim spoken =
                new Verbatim(
                        MODEL,
                        IntSequence.of(1, Verbatim.FRAME, 2, Verbatim.FRAME),
                        IntSequence.of(10, 11, 12, 20, 21, 22));
        assertEquals(2, spoken.frames());
        assertEquals(3, spoken.codebooks());

        assertThrows(
                IllegalArgumentException.class,
                () -> new Verbatim(MODEL, IntSequence.of(1, 2), IntSequence.of(10)),
                "codes without a frame");
        assertThrows(
                IllegalArgumentException.class,
                () -> new Verbatim(MODEL, IntSequence.of(Verbatim.FRAME), IntSequence.empty()),
                "a frame without codes");
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new Verbatim(
                                MODEL,
                                IntSequence.of(Verbatim.FRAME, Verbatim.FRAME),
                                IntSequence.of(10, 11, 12)),
                "codes that do not divide by the frames");
        assertTrue(Verbatim.NONE.isEmpty());
        assertFalse(text.isEmpty());
    }

    @Test
    void aTurnWithoutARecordIsRendered() {
        List<Batch> batches = new ArrayList<>();
        PromptWriter out = new PromptWriter(TOKENIZER, 8, batches::add);
        assertFalse(out.replay(Message.assistant("hi")));
        out.text("hi").finish();
        assertArrayEquals(new int[] {'h', 'i'}, Batch.tokenIds(batches));
    }

    @Test
    void aRecordReplaysItsPositionsAsTheyWere() {
        Message turn =
                new Message(
                        Role.ASSISTANT,
                        List.of(new Content.Text("whatever")),
                        new Verbatim(MODEL, IntSequence.of(7, 8, 9), IntSequence.empty()));
        List<Batch> batches = new ArrayList<>();
        PromptWriter out = new PromptWriter(TOKENIZER, 8, batches::add);
        out.text("a");
        assertTrue(out.replay(turn));
        out.text("b").finish();
        assertArrayEquals(new int[] {'a', 7, 8, 9, 'b'}, Batch.tokenIds(batches));
    }

    @Test
    void aSpokenRecordReplaysEachFrameThroughTheVoice() {
        Message turn =
                new Message(
                        Role.ASSISTANT,
                        List.of(new Content.Text("")),
                        new Verbatim(
                                MODEL,
                                IntSequence.of(7, Verbatim.FRAME, Verbatim.FRAME, 8),
                                IntSequence.of(1, 2, 3, 4)));
        List<int[]> frames = new ArrayList<>();
        List<Batch> batches = new ArrayList<>();
        PromptWriter out = new PromptWriter(TOKENIZER, 8, batches::add);
        assertTrue(
                out.replay(
                        turn,
                        codes -> {
                            frames.add(codes);
                            return Batch.prefill(new int[] {100 + codes[0]});
                        }));
        out.finish();
        assertEquals(2, frames.size());
        assertArrayEquals(new int[] {1, 2}, frames.get(0));
        assertArrayEquals(new int[] {3, 4}, frames.get(1));
        assertArrayEquals(new int[] {7, 101, 103, 8}, Batch.tokenIds(batches));

        PromptWriter voiceless = new PromptWriter(TOKENIZER, 8, new ArrayList<>()::add);
        assertThrows(UnsupportedConversation.class, () -> voiceless.replay(turn));
    }
}
