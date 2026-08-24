package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.impl.MemoryFactory;
import org.junit.jupiter.api.Test;

class SliceTest {

    @Test
    void positiveStepUpdatesValuesOffsetAndStride() {
        MemoryView<byte[]> slice = vector(0, 1, 2, 3, 4, 5, 6).slice(0, 1, 7, 2);

        assertEquals(Shape.flat(3), slice.shape());
        assertEquals(1, slice.byteOffset());
        assertEquals(Stride.flat(2), slice.byteStride());
        assertArrayEquals(new byte[] {1, 3, 5}, values(slice));
    }

    @Test
    void negativeStepReversesAView() {
        MemoryView<byte[]> slice = vector(0, 1, 2, 3, 4, 5, 6).slice(-1, 6, -1, -2);

        assertEquals(Shape.flat(4), slice.shape());
        assertEquals(6, slice.byteOffset());
        assertEquals(Stride.flat(-2), slice.byteStride());
        assertArrayEquals(new byte[] {6, 4, 2, 0}, values(slice));
    }

    @Test
    void slicesANonContiguousView() {
        MemoryView<byte[]> transposed =
                vector(0, 1, 2, 3, 4, 5).view(Shape.flat(2, 3)).transpose(0, 1);

        MemoryView<byte[]> slice = transposed.slice(0, 0, 3, 2);

        assertEquals(Shape.flat(2, 2), slice.shape());
        assertArrayEquals(new byte[] {0, 3, 2, 5}, values(slice));
    }

    @Test
    void emptySliceHasNoElements() {
        MemoryView<byte[]> slice = vector(0, 1, 2).slice(0, 2, 2);

        assertEquals(Shape.flat(0), slice.shape());
        assertArrayEquals(new byte[0], values(slice));
    }

    @Test
    void rejectsInvalidSlices() {
        MemoryView<byte[]> view = vector(0, 1, 2);

        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () -> view.slice(0, 0, 2, 0)),
                () -> assertThrows(IllegalArgumentException.class, () -> view.slice(0, -1, 2, 1)),
                () -> assertThrows(IllegalArgumentException.class, () -> view.slice(0, 0, 4, 1)),
                () -> assertThrows(IllegalArgumentException.class, () -> view.slice(0, 2, 1, 1)),
                () -> assertThrows(IllegalArgumentException.class, () -> view.slice(0, 3, -1, -1)),
                () -> assertThrows(IllegalArgumentException.class, () -> view.slice(0, 0, 1, -1)),
                () -> assertThrows(IllegalArgumentException.class, () -> view.slice(1, 0, 1, 1)));
    }

    private static MemoryView<byte[]> vector(int... values) {
        byte[] bytes = new byte[values.length];
        for (int i = 0; i < values.length; i++) {
            bytes[i] = (byte) values[i];
        }
        return MemoryView.rowMajor(
                MemoryFactory.ofBytes(bytes), DataType.I8, Shape.flat(bytes.length));
    }

    private static byte[] values(MemoryView<byte[]> view) {
        byte[] values = new byte[Math.toIntExact(view.shape().size())];
        for (int i = 0; i < values.length; i++) {
            values[i] = view.memory().base()[Math.toIntExact(Indexing.linearToOffset(view, i))];
        }
        return values;
    }
}
