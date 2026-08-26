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

class PermuteTest {

    @Test
    void permutesShapeStrideAndValues() {
        MemoryView<byte[]> source =
                MemoryView.rowMajor(
                        MemoryFactory.ofBytes(new byte[] {0, 1, 2, 3, 4, 5}),
                        DataType.I8,
                        Shape.flat(2, 3));

        MemoryView<byte[]> result = source.permute(1, 0);

        assertEquals(Shape.flat(3, 2), result.shape());
        assertEquals(Stride.flat(1, 3), result.stride());
        assertArrayEquals(new byte[] {0, 3, 1, 4, 2, 5}, values(result));
    }

    @Test
    void rejectsInvalidPermutations() {
        MemoryView<byte[]> source =
                MemoryView.rowMajor(
                        MemoryFactory.ofBytes(new byte[6]), DataType.I8, Shape.flat(2, 3));

        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () -> source.permute(0)),
                () -> assertThrows(IllegalArgumentException.class, () -> source.permute(0, 0)),
                () -> assertThrows(IllegalArgumentException.class, () -> source.permute(0, 2)),
                () -> assertThrows(IllegalArgumentException.class, () -> source.permute(-1, 0)));
    }

    private static byte[] values(MemoryView<byte[]> view) {
        byte[] values = new byte[Math.toIntExact(view.shape().size())];
        for (int i = 0; i < values.length; i++) {
            values[i] = view.memory().base()[Math.toIntExact(Indexing.linearToOffset(view, i))];
        }
        return values;
    }
}
