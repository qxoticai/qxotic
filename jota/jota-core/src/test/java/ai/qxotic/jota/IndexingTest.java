package ai.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class IndexingTest {

    @Test
    void linearToCoordRowMajor() {
        Shape shape = Shape.of(2, 3, 4);
        assertArrayEquals(new long[] {0, 0, 0}, Indexing.linearToCoord(shape, 0));
        assertArrayEquals(new long[] {0, 0, 1}, Indexing.linearToCoord(shape, 1));
        assertArrayEquals(new long[] {0, 1, 0}, Indexing.linearToCoord(shape, 4));
        assertArrayEquals(new long[] {1, 2, 3}, Indexing.linearToCoord(shape, 23));
    }

    @Test
    void coordToLinearRowMajor() {
        Shape shape = Shape.of(2, 3, 4);
        assertEquals(0, Indexing.coordToLinear(shape, 0, 0, 0));
        assertEquals(1, Indexing.coordToLinear(shape, 0, 0, 1));
        assertEquals(4, Indexing.coordToLinear(shape, 0, 1, 0));
        assertEquals(23, Indexing.coordToLinear(shape, 1, 2, 3));
    }

    @Test
    void coordToOffsetUsesStride() {
        Shape shape = Shape.of(2, 3, 4);
        Stride stride = Stride.rowMajor(shape);
        assertEquals(23, Indexing.coordToOffset(stride, 1, 2, 3));
        assertEquals(4, Indexing.coordToOffset(stride, 0, 1, 0));
    }

    @Test
    void linearToOffsetMatchesRowMajor() {
        Shape shape = Shape.of(2, 3, 4);
        Stride stride = Stride.rowMajor(shape);
        assertEquals(92, Indexing.linearToOffset(shape, stride, DataType.FP32, 23));
        assertEquals(16, Indexing.linearToOffset(shape, stride, DataType.FP32, 4));
    }

    @Test
    void linearToOffsetLayoutShortcut() {
        Shape shape = Shape.of(2, 3, 4);
        Layout layout = Layout.rowMajor(shape);
        assertEquals(92, Indexing.linearToOffset(layout, DataType.FP32, 23));
    }

    @Test
    void linearToOffsetRespectsStrideLayout() {
        Shape shape = Shape.of(2, 3);
        Stride stride = Stride.columnMajor(shape);
        assertEquals(12, Indexing.linearToOffset(shape, stride, DataType.FP32, 4));
    }

    @Test
    void linearToOffsetMatchesCoordToOffset() {
        Shape shape = Shape.of(2, 3, 4);
        Stride stride = Stride.rowMajor(shape);
        for (long linear = 0; linear < shape.size(); linear++) {
            long[] coord = Indexing.linearToCoord(shape, linear);
            long expected = Indexing.coordToOffset(stride, coord) * DataType.FP32.byteSize();
            assertEquals(expected, Indexing.linearToOffset(shape, stride, DataType.FP32, linear));
        }
    }

    @Test
    void nestedShapeUsesFlatCoordinates() {
        Shape shape = Shape.pattern("(a, (b, c))", 2, 3, 4);
        Stride stride = Stride.rowMajor(shape);
        assertArrayEquals(new long[] {0, 0, 0}, Indexing.linearToCoord(shape, 0));
        assertArrayEquals(new long[] {1, 2, 3}, Indexing.linearToCoord(shape, 23));
        assertEquals(23, Indexing.coordToLinear(shape, 1, 2, 3));
        assertEquals(23, Indexing.coordToOffset(stride, 1, 2, 3));
    }

    @Test
    void customStrideOffsets() {
        Stride stride = Stride.flat(10, 3, 1);
        assertEquals(17, Indexing.coordToOffset(stride, 1, 2, 1));
    }

    @Test
    void scalarConversions() {
        Shape scalar = Shape.scalar();
        assertArrayEquals(new long[0], Indexing.linearToCoord(scalar, 0));
        assertEquals(0, Indexing.coordToLinear(scalar));
        assertEquals(0, Indexing.linearToOffset(scalar, Stride.rowMajor(scalar), DataType.FP32, 0));
    }

    @Test
    void throwsForOutOfBoundsLinearIndex() {
        Shape shape = Shape.of(2, 3);
        assertThrows(IllegalArgumentException.class, () -> Indexing.linearToCoord(shape, 6));
        assertThrows(
                IllegalArgumentException.class, () -> Indexing.linearToCoord(Shape.scalar(), 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> Indexing.linearToOffset(shape, Stride.rowMajor(shape), DataType.FP32, -1));
    }

    @Test
    void throwsForInvalidCoordinates() {
        Shape shape = Shape.of(2, 3, 4);
        assertThrows(IllegalArgumentException.class, () -> Indexing.coordToLinear(shape, 2, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> Indexing.coordToLinear(shape, 0, 3, 0));
        assertThrows(IllegalArgumentException.class, () -> Indexing.coordToLinear(shape, 0, 0, 4));
        assertThrows(IllegalArgumentException.class, () -> Indexing.coordToLinear(shape, 0, 0));
    }

    @Test
    void throwsForMismatchedStrideRank() {
        Stride stride = Stride.of(4, 1);
        assertThrows(IllegalArgumentException.class, () -> Indexing.coordToOffset(stride, 1));
    }
}
