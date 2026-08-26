package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.internal.MemoryFactory;
import org.junit.jupiter.api.Test;

class BroadcastTest {

    @Test
    void broadcastsValuesFromAViewWithAnOffset() {
        MemoryView<byte[]> source =
                MemoryView.of(
                        MemoryFactory.ofBytes(new byte[] {9, 1, 2, 3, 9}),
                        1,
                        DataType.I8,
                        com.qxotic.jota.Layout.rowMajor(Shape.flat(3)));

        MemoryView<byte[]> result = source.broadcast(Shape.flat(2, 3));

        assertEquals(Shape.flat(2, 3), result.shape());
        assertEquals(Stride.flat(0, 1), result.stride());
        assertArrayEquals(new byte[] {1, 2, 3, 1, 2, 3}, values(result));
    }

    @Test
    void broadcastsANegativeStrideView() {
        MemoryView<byte[]> reversed =
                MemoryView.rowMajor(
                                MemoryFactory.ofBytes(new byte[] {1, 2, 3}),
                                DataType.I8,
                                Shape.flat(3))
                        .slice(0, 2, -1, -1);

        MemoryView<byte[]> result = reversed.broadcast(Shape.flat(2, 3));

        assertEquals(Stride.flat(0, -1), result.stride());
        assertArrayEquals(new byte[] {3, 2, 1, 3, 2, 1}, values(result));
    }

    @Test
    void broadcastsAnEmptyDimension() {
        MemoryView<byte[]> source =
                MemoryView.rowMajor(
                        MemoryFactory.ofBytes(new byte[0]), DataType.I8, Shape.flat(1, 0));

        MemoryView<byte[]> result = source.broadcast(Shape.flat(4, 0));

        assertEquals(Shape.flat(4, 0), result.shape());
        assertArrayEquals(new byte[0], values(result));
    }

    @Test
    void rejectsIncompatibleTargets() {
        MemoryView<byte[]> vector =
                MemoryView.rowMajor(MemoryFactory.ofBytes(new byte[3]), DataType.I8, Shape.flat(3));
        MemoryView<byte[]> matrix =
                MemoryView.rowMajor(
                        MemoryFactory.ofBytes(new byte[6]), DataType.I8, Shape.flat(2, 3));

        assertAll(
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> vector.broadcast(Shape.flat(4))),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> matrix.broadcast(Shape.flat(3))));
    }

    private static byte[] values(MemoryView<byte[]> view) {
        byte[] values = new byte[Math.toIntExact(view.shape().size())];
        for (int i = 0; i < values.length; i++) {
            values[i] = view.memory().base()[Math.toIntExact(Indexing.linearToOffset(view, i))];
        }
        return values;
    }
}
