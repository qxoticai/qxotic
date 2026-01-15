package ai.qxotic;

import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for Layout methods including modeAt, flatten, scalar, toString, etc.
 */
class LayoutMethodsTest {

    @Test
    void testLayoutOfBasics() {
        Shape shape = Shape.flat(2, 3, 4);
        Stride stride = Stride.flat(12, 4, 1);
        Layout layout = Layout.of(shape, stride);

        assertSame(shape, layout.shape());
        assertSame(stride, layout.stride());
    }

    @Test
    void testRowMajorLayout() {
        Shape shape = Shape.flat(2, 3, 4);
        Layout layout = Layout.rowMajor(shape);

        assertEquals(shape, layout.shape());
        assertArrayEquals(new long[]{12, 4, 1}, layout.stride().toArray());
    }

    @Test
    void testColumnMajorLayout() {
        Shape shape = Shape.flat(2, 3, 4);
        Layout layout = Layout.columnMajor(shape);

        assertEquals(shape, layout.shape());
        assertArrayEquals(new long[]{1, 2, 6}, layout.stride().toArray());
    }

    @Test
    void testScalarLayout() {
        Layout scalar = Layout.scalar();

        assertTrue(scalar.shape().isScalar());
        assertTrue(scalar.stride().isScalar());
        assertEquals(0, scalar.shape().rank());
        assertEquals(0, scalar.stride().rank());
    }

    @Test
    void testModeAtFlatLayout() {
        Shape shape = Shape.flat(2, 3, 4);
        Layout layout = Layout.rowMajor(shape);

        // Extract mode 0
        Layout mode0 = layout.modeAt(0);
        assertEquals(1, mode0.shape().rank());
        assertEquals(2, mode0.shape().flatAt(0));
        assertEquals(1, mode0.stride().rank());
        assertEquals(12, mode0.stride().flatAt(0));

        // Extract mode 1
        Layout mode1 = layout.modeAt(1);
        assertEquals(1, mode1.shape().rank());
        assertEquals(3, mode1.shape().flatAt(0));
        assertEquals(1, mode1.stride().rank());
        assertEquals(4, mode1.stride().flatAt(0));

        // Extract mode 2
        Layout mode2 = layout.modeAt(2);
        assertEquals(1, mode2.shape().rank());
        assertEquals(4, mode2.shape().flatAt(0));
        assertEquals(1, mode2.stride().rank());
        assertEquals(1, mode2.stride().flatAt(0));
    }

    @Test
    void testModeAtNestedLayout() {
        // Shape: (2, (3, 4)), Stride: (12, (4, 1))
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Layout layout = Layout.rowMajor(shape);

        // Extract mode 0
        Layout mode0 = layout.modeAt(0);
        assertEquals(1, mode0.shape().rank());
        assertTrue(mode0.shape().isFlat());
        assertEquals(2, mode0.shape().flatAt(0));
        assertEquals(12, mode0.stride().flatAt(0));

        // Extract mode 1
        Layout mode1 = layout.modeAt(1);
        assertEquals(2, mode1.shape().rank());
        assertTrue(mode1.shape().isFlat());
        assertArrayEquals(new long[]{3, 4}, mode1.shape().toArray());
        assertArrayEquals(new long[]{4, 1}, mode1.stride().toArray());
    }

    @Test
    void testModeAtNegativeIndex() {
        Shape shape = Shape.flat(2, 3, 4);
        Layout layout = Layout.rowMajor(shape);

        Layout lastMode = layout.modeAt(-1);
        assertEquals(1, lastMode.shape().rank());
        assertEquals(4, lastMode.shape().flatAt(0));
        assertEquals(1, lastMode.stride().flatAt(0));

        Layout secondLastMode = layout.modeAt(-2);
        assertEquals(1, secondLastMode.shape().rank());
        assertEquals(3, secondLastMode.shape().flatAt(0));
        assertEquals(4, secondLastMode.stride().flatAt(0));
    }

    @Test
    void testFlattenFlatLayout() {
        Shape shape = Shape.flat(2, 3, 4);
        Layout layout = Layout.rowMajor(shape);
        Layout flattened = layout.flatten();

        assertSame(layout, flattened);  // Should return same instance
    }

    @Test
    void testFlattenNestedLayout() {
        // Shape: (2, (3, 4)), Stride: (12, (4, 1))
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Layout layout = Layout.rowMajor(shape);
        Layout flattened = layout.flatten();

        assertTrue(flattened.shape().isFlat());
        assertTrue(flattened.stride().isFlat());
        assertEquals(3, flattened.shape().rank());
        assertEquals(3, flattened.stride().rank());
        assertArrayEquals(new long[]{2, 3, 4}, flattened.shape().toArray());
        assertArrayEquals(new long[]{12, 4, 1}, flattened.stride().toArray());
    }

    @Test
    void testToStringFlat() {
        Shape shape = Shape.flat(2, 3, 4);
        Layout layout = Layout.rowMajor(shape);

        assertEquals("(2, 3, 4):(12, 4, 1)", layout.toString());
    }

    @Test
    void testToStringNested() {
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Layout layout = Layout.rowMajor(shape);

        assertEquals("(2, (3, 4)):(12, (4, 1))", layout.toString());
    }

    @Test
    void testToStringScalar() {
        Layout scalar = Layout.scalar();
        assertEquals("():()", scalar.toString());
    }

    @Test
    void testRowMajorNestedShape() {
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Layout layout = Layout.rowMajor(shape);

        assertEquals(2, layout.shape().rank());
        assertEquals(3, layout.shape().flatRank());
        assertFalse(layout.shape().isFlat());
        assertFalse(layout.stride().isFlat());
        assertArrayEquals(new long[]{12, 4, 1}, layout.stride().toArray());
    }

    @Test
    void testColumnMajorNestedShape() {
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Layout layout = Layout.columnMajor(shape);

        assertEquals(2, layout.shape().rank());
        assertEquals(3, layout.shape().flatRank());
        assertFalse(layout.shape().isFlat());
        assertFalse(layout.stride().isFlat());
        assertArrayEquals(new long[]{1, 2, 6}, layout.stride().toArray());
    }

    @Test
    void testDeeplyNestedLayout() {
        // Shape: (2, (3, (4, 5)))
        Shape shape = Shape.of(2, Shape.of(3, Shape.of(4L, 5L)));
        Layout layout = Layout.rowMajor(shape);

        assertEquals(2, layout.shape().rank());
        assertEquals(4, layout.shape().flatRank());
        assertArrayEquals(new long[]{2, 3, 4, 5}, layout.shape().toArray());
        assertArrayEquals(new long[]{60, 20, 5, 1}, layout.stride().toArray());

        // Test mode extraction
        Layout mode1 = layout.modeAt(1);
        assertEquals(2, mode1.shape().rank());
        assertEquals(3, mode1.shape().flatRank());
        assertArrayEquals(new long[]{3, 4, 5}, mode1.shape().toArray());
        assertArrayEquals(new long[]{20, 5, 1}, mode1.stride().toArray());
    }

    @Test
    void testEqualsAndHashCode() {
        Shape shape = Shape.flat(2, 3, 4);
        Layout layout1 = Layout.rowMajor(shape);
        Layout layout2 = Layout.rowMajor(shape);

        assertEquals(layout1, layout2);
        assertEquals(layout1.hashCode(), layout2.hashCode());
    }

    @Test
    void testNotEquals() {
        Shape shape = Shape.flat(2, 3, 4);
        Layout rowMajor = Layout.rowMajor(shape);
        Layout colMajor = Layout.columnMajor(shape);

        assertNotEquals(rowMajor, colMajor);
    }

    @Test
    void testCustomStrides() {
        Shape shape = Shape.flat(2, 3, 4);
        Stride customStride = Stride.flat(100, 10, 1);
        Layout layout = Layout.of(shape, customStride);

        assertEquals(shape, layout.shape());
        assertEquals(customStride, layout.stride());
        assertEquals("(2, 3, 4):(100, 10, 1)", layout.toString());
    }

    @Test
    void testNestedShapeWithCustomStrides() {
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Stride stride = Stride.of(100, Stride.of(10L, 1L));
        Layout layout = Layout.of(shape, stride);

        assertEquals(2, layout.shape().rank());
        assertEquals(2, layout.stride().rank());
        assertFalse(layout.shape().isFlat());
        assertFalse(layout.stride().isFlat());
    }

    @Test
    void testPatternBasedLayout() {
        Shape shape = Shape.pattern("(batch, (N, M))", 2, 3, 4);
        Layout layout = Layout.rowMajor(shape);

        assertEquals(2, layout.shape().rank());
        assertEquals(3, layout.shape().flatRank());
        assertEquals(2, layout.stride().rank());
        assertEquals(3, layout.stride().flatRank());
    }
}
