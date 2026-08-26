package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
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
    void elementOffsetsAreScaledByTheDataType() {
        MemoryView<float[]> slice =
                MemoryView.rowMajor(Memories.of(new float[5]), DataType.FP32, Shape.of(5))
                        .slice(0, 2, 5);

        assertEquals(2L * Float.BYTES, slice.byteOffset());
        assertEquals(Stride.of(Float.BYTES), slice.byteStride());
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
    void slicesAnAffineNestedMode() {
        float[] data = new float[24];
        for (int i = 0; i < data.length; i++) {
            data[i] = i;
        }
        MemoryView<float[]> view =
                MemoryView.of(
                        Memories.of(data),
                        DataType.FP32,
                        Layout.rowMajor(Shape.of(Shape.of(2L, 3L), 4)));

        MemoryView<float[]> slice = view.slice(0, 1, 5, 2);

        assertEquals(Shape.of(2, 4), slice.shape());
        assertEquals(4L * Float.BYTES, slice.byteOffset());
        assertArrayEquals(new float[] {4, 5, 6, 7, 12, 13, 14, 15}, valuesAsFloats(slice));
    }

    @Test
    void rejectsANonAffineNestedMode() {
        Layout layout = Layout.of(Shape.of(Shape.of(2L, 3L), 4), Stride.of(Stride.of(4L, 8L), 1));
        MemoryView<float[]> view = MemoryView.of(Memories.of(new float[24]), DataType.FP32, layout);

        assertThrows(IllegalArgumentException.class, () -> view.slice(0, 0, 6));
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
        return MemoryView.rowMajor(Memories.of(bytes), DataType.I8, Shape.flat(bytes.length));
    }

    private static byte[] values(MemoryView<byte[]> view) {
        byte[] values = new byte[Math.toIntExact(view.shape().size())];
        for (int i = 0; i < values.length; i++) {
            values[i] = view.memory().base()[Math.toIntExact(Indexing.linearToOffset(view, i))];
        }
        return values;
    }

    private static float[] valuesAsFloats(MemoryView<float[]> view) {
        float[] values = new float[Math.toIntExact(view.shape().size())];
        for (int i = 0; i < values.length; i++) {
            long byteOffset = Indexing.linearToOffset(view, i);
            values[i] = view.memory().base()[Math.toIntExact(byteOffset / Float.BYTES)];
        }
        return values;
    }
}
