package com.qxotic;

import com.qxotic.jota.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for Shape methods including rank, size, modeAt, flatten, etc.
 */
class ShapeMethodsTest {

    @Test
    void testFlatShapeBasics() {
        Shape shape = Shape.flat(2, 3, 4);

        assertEquals(3, shape.rank());
        assertEquals(3, shape.flatRank());
        assertEquals(24, shape.size());
        assertTrue(shape.isFlat());
        assertFalse(shape.isScalar());
        assertArrayEquals(new long[]{2, 3, 4}, shape.toArray());
    }

    @Test
    void testNestedShapeBasics() {
        // (2, (3, 4))
        Shape shape = Shape.of(2, Shape.of(3L, 4L));

        assertEquals(2, shape.rank());
        assertEquals(3, shape.flatRank());
        assertEquals(24, shape.size());
        assertFalse(shape.isFlat());
        assertFalse(shape.isScalar());
        assertArrayEquals(new long[]{2, 3, 4}, shape.toArray());
    }

    @Test
    void testScalarShape() {
        Shape scalar = Shape.scalar();

        assertEquals(0, scalar.rank());
        assertEquals(0, scalar.flatRank());
        assertEquals(1, scalar.size());
        assertTrue(scalar.isFlat());
        assertTrue(scalar.isScalar());
        assertArrayEquals(new long[]{}, scalar.toArray());
    }

    @Test
    void testSizeByMode() {
        // (2, (3, 4), 5)
        Shape shape = Shape.of(2, Shape.of(3L, 4L), 5);

        assertEquals(3, shape.rank());
        assertEquals(2, shape.size(0));
        assertEquals(12, shape.size(1));  // 3 * 4
        assertEquals(5, shape.size(2));
    }

    @Test
    void testSizeByModeNegativeIndex() {
        Shape shape = Shape.flat(2, 3, 4);

        assertEquals(4, shape.size(-1));  // Last dimension
        assertEquals(3, shape.size(-2));  // Second to last
        assertEquals(2, shape.size(-3));  // Third to last
    }

    @Test
    void testFlatAtPositiveIndices() {
        Shape shape = Shape.flat(2, 3, 4);

        assertEquals(2, shape.flatAt(0));
        assertEquals(3, shape.flatAt(1));
        assertEquals(4, shape.flatAt(2));
    }

    @Test
    void testFlatAtNegativeIndices() {
        Shape shape = Shape.flat(2, 3, 4);

        assertEquals(4, shape.flatAt(-1));  // Last
        assertEquals(3, shape.flatAt(-2));  // Second to last
        assertEquals(2, shape.flatAt(-3));  // Third to last
    }

    @Test
    void testModeAtFlatShape() {
        Shape shape = Shape.flat(2, 3, 4);

        Shape mode0 = shape.modeAt(0);
        assertEquals(1, mode0.rank());
        assertEquals(2, mode0.flatAt(0));

        Shape mode1 = shape.modeAt(1);
        assertEquals(1, mode1.rank());
        assertEquals(3, mode1.flatAt(0));

        Shape mode2 = shape.modeAt(2);
        assertEquals(1, mode2.rank());
        assertEquals(4, mode2.flatAt(0));
    }

    @Test
    void testModeAtNestedShape() {
        // (2, (3, 4))
        Shape shape = Shape.of(2, Shape.of(3L, 4L));

        Shape mode0 = shape.modeAt(0);
        assertEquals(1, mode0.rank());
        assertTrue(mode0.isFlat());
        assertEquals(2, mode0.flatAt(0));

        Shape mode1 = shape.modeAt(1);
        assertEquals(2, mode1.rank());
        assertTrue(mode1.isFlat());
        assertArrayEquals(new long[]{3, 4}, mode1.toArray());
    }

    @Test
    void testModeAtNegativeIndex() {
        Shape shape = Shape.flat(2, 3, 4);

        Shape lastMode = shape.modeAt(-1);
        assertEquals(1, lastMode.rank());
        assertEquals(4, lastMode.flatAt(0));

        Shape secondLastMode = shape.modeAt(-2);
        assertEquals(1, secondLastMode.rank());
        assertEquals(3, secondLastMode.flatAt(0));
    }

    @Test
    void testFlattenFlatShape() {
        Shape shape = Shape.flat(2, 3, 4);
        Shape flattened = shape.flatten();

        assertSame(shape, flattened);  // Should return same instance
    }

    @Test
    void testFlattenNestedShape() {
        // (2, (3, 4))
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Shape flattened = shape.flatten();

        assertTrue(flattened.isFlat());
        assertEquals(3, flattened.rank());
        assertEquals(3, flattened.flatRank());
        assertArrayEquals(new long[]{2, 3, 4}, flattened.toArray());
    }

    @Test
    void testDeeplyNestedShape() {
        // (2, (3, (4, 5)))
        Shape shape = Shape.of(2, Shape.of(3, Shape.of(4L, 5L)));

        assertEquals(2, shape.rank());
        assertEquals(4, shape.flatRank());
        assertEquals(120, shape.size());
        assertArrayEquals(new long[]{2, 3, 4, 5}, shape.toArray());

        // Test mode extraction
        Shape mode1 = shape.modeAt(1);
        assertEquals(2, mode1.rank());
        assertEquals(3, mode1.flatRank());
        assertArrayEquals(new long[]{3, 4, 5}, mode1.toArray());
    }

    @Test
    void testPatternBasedShape() {
        Shape shape = Shape.pattern("(batch, (N, M))", 2, 3, 4);

        assertEquals(2, shape.rank());
        assertEquals(3, shape.flatRank());
        assertEquals(2, shape.size(0));
        assertEquals(12, shape.size(1));
        assertArrayEquals(new long[]{2, 3, 4}, shape.toArray());
    }

    @Test
    void testTemplateBasedShape() {
        Shape template = Shape.pattern("(_,(_,_))", 1, 2, 3);
        Shape newShape = Shape.template(template, 10, 20, 30);

        assertEquals(template.rank(), newShape.rank());
        assertEquals(template.flatRank(), newShape.flatRank());
        assertEquals(2, newShape.rank());
        assertArrayEquals(new long[]{10, 20, 30}, newShape.toArray());
    }

    @Test
    void testMultipleRootsInNestedShape() {
        // Create (2, (3, 4, 5)) where the nested group has one root with two children
        Shape inner = Shape.flat(3, 4, 5);
        Shape shape = Shape.of(2, inner);

        assertEquals(2, shape.rank());
        assertEquals(4, shape.flatRank());
        assertArrayEquals(new long[]{2, 3, 4, 5}, shape.toArray());
    }

    @Test
    void testToString() {
        Shape flat = Shape.flat(2, 3, 4);
        assertEquals("(2, 3, 4)", flat.toString());

        Shape nested = Shape.of(2, Shape.of(3L, 4L));
        assertEquals("(2, (3, 4))", nested.toString());
    }
}
