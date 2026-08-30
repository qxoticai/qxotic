package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

class MediaBoundaryTest {

    @Test
    void validatesImages() {
        assertEquals(3, new Media.Image(new float[] {0.0f, 0.5f, 1.0f}, 1, 1, 3).channels());
        assertThrows(
                IllegalArgumentException.class, () -> new Media.Image(new float[] {0.0f}, 0, 1, 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Media.Image(new float[] {0.0f, 0.0f}, 1, 1, 2));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Media.Image(new float[] {Float.NaN}, 1, 1, 1));
    }

    @Test
    void validatesAudioAndBorrowsPcm() {
        float[] pcm = {-1.0f, 0.5f, 1.0f, 0.0f};
        Media.Audio audio = new Media.Audio(pcm, 48_000, 2);

        assertSame(pcm, audio.pcm());
        assertEquals(48_000, audio.sampleRate());
        assertEquals(2, audio.channels());
        float[] empty = {};
        assertSame(empty, new Media.Audio(empty, 48_000, 2).pcm());
    }

    @Test
    void rejectsInvalidAudio() {
        assertThrows(NullPointerException.class, () -> new Media.Audio(null, 48_000, 1));
        assertThrows(
                IllegalArgumentException.class, () -> new Media.Audio(new float[] {0.0f}, 0, 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Media.Audio(new float[] {0.0f}, 48_000, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Media.Audio(new float[] {0.0f}, 48_000, 2));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Media.Audio(new float[] {Float.NaN}, 48_000, 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Media.Audio(new float[] {Float.POSITIVE_INFINITY}, 48_000, 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Media.Audio(new float[] {1.0001f}, 48_000, 1));
    }

    @Test
    void validatesDenseFp32EmbeddingRowsAndKeepsContentKey() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<?> rows =
                    Views.allocateF32(MemoryAllocators.ofArena(arena), 12).view(Shape.flat(3, 4));
            ContentKey key = new ContentKey("media:test");
            Batch.Positions positions =
                    new Batch.Positions(3, new int[] {0, 0, 0, 0, 1, 2, 0, 3, 4}, 5);
            Batch.Input.Embeddings embeddings =
                    new Batch.Input.Embeddings(rows, 3, true, key, positions);
            assertEquals(key, embeddings.contentKey());
            assertEquals(2, embeddings.positions().value(1, 2));

            Batch.Positions tail = positions.slice(1, 2, true);
            assertEquals(4, tail.advance());
            assertEquals(0, tail.value(0, 1));
            assertEquals(3, tail.value(1, 2));

            assertThrows(
                    IllegalArgumentException.class,
                    () -> new Batch.Input.Embeddings(rows, 2, true));
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            new Batch.Input.Embeddings(
                                    rows, 3, true, key, new Batch.Positions(3, new int[6], 2)));
            MemoryView<?> i32 =
                    MemoryView.of(rows.memory(), DataType.I32, Layout.rowMajor(Shape.flat(3, 4)));
            assertThrows(
                    IllegalArgumentException.class, () -> new Batch.Input.Embeddings(i32, 3, true));
            MemoryView<?> strided = rows.slice(1, 0, 4, 2);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> new Batch.Input.Embeddings(strided, 3, true));
        }
    }
}
