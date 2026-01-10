package com.qxotic;

import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Layout.isCongruentWith(Layout other) following CuTe semantics.
 * Two layouts are congruent if both their shapes AND strides have the same hierarchical structure.
 */
class LayoutCongruenceTest {

    @Test
    void testCongruentLayouts() {
        // Two layouts with same nesting structure
        Layout layout1 = Layout.of(
            Shape.of(2, Shape.of(3L, 4L)),
            Stride.of(12, Stride.of(4L, 1L))
        );

        Layout layout2 = Layout.of(
            Shape.of(10, Shape.of(20L, 30L)),
            Stride.of(600, Stride.of(30L, 1L))
        );

        assertTrue(layout1.isCongruentWith(layout2));
        assertTrue(layout2.isCongruentWith(layout1));
    }

    @Test
    void testNonCongruentShapes() {
        // Different shape structures
        Layout layout1 = Layout.of(
            Shape.flat(2, 3, 4),
            Stride.flat(12, 4, 1)
        );

        Layout layout2 = Layout.of(
            Shape.of(2, Shape.of(3L, 4L)),
            Stride.of(12, Stride.of(4L, 1L))
        );

        assertFalse(layout1.isCongruentWith(layout2));
        assertFalse(layout2.isCongruentWith(layout1));
    }

    @Test
    void testNonCongruentStrides() {
        // Same shape structure but different stride structure
        Layout layout1 = Layout.of(
            Shape.of(2, Shape.of(3L, 4L)),
            Stride.flat(12, 4, 1)  // Flat stride
        );

        Layout layout2 = Layout.of(
            Shape.of(2, Shape.of(3L, 4L)),
            Stride.of(12, Stride.of(4L, 1L))  // Nested stride
        );

        assertFalse(layout1.isCongruentWith(layout2));
    }

    @Test
    void testFlatLayoutsCongruent() {
        Layout layout1 = Layout.of(Shape.flat(2, 3, 4), Stride.flat(12, 4, 1));
        Layout layout2 = Layout.of(Shape.flat(5, 6, 7), Stride.flat(42, 7, 1));

        assertTrue(layout1.isCongruentWith(layout2));
    }

    @Test
    void testRowMajorLayouts() {
        Shape shape1 = Shape.of(2, Shape.of(3L, 4L));
        Shape shape2 = Shape.of(10, Shape.of(20L, 30L));

        Layout layout1 = Layout.rowMajor(shape1);
        Layout layout2 = Layout.rowMajor(shape2);

        // Both are row-major with same structure
        assertTrue(layout1.isCongruentWith(layout2));
    }

    @Test
    void testColumnMajorLayouts() {
        Shape shape1 = Shape.of(2, Shape.of(3L, 4L));
        Shape shape2 = Shape.of(10, Shape.of(20L, 30L));

        Layout layout1 = Layout.columnMajor(shape1);
        Layout layout2 = Layout.columnMajor(shape2);

        // Both are column-major with same structure
        assertTrue(layout1.isCongruentWith(layout2));
    }

    @Test
    void testRowMajorVsColumnMajor() {
        Shape shape1 = Shape.flat(2, 3, 4);
        Shape shape2 = Shape.flat(5, 6, 7);

        Layout rowMajor = Layout.rowMajor(shape1);
        Layout colMajor = Layout.columnMajor(shape2);

        // Row-major: [12, 4, 1], Column-major: [1, 5, 30]
        // Different stride values but both are flat structures
        assertTrue(rowMajor.isCongruentWith(colMajor));
    }

    @Test
    void testScalarLayouts() {
        Layout scalar1 = Layout.scalar();
        Layout scalar2 = Layout.scalar();

        assertTrue(scalar1.isCongruentWith(scalar2));
    }

    @Test
    void testCongruenceNull() {
        Layout layout = Layout.of(Shape.flat(2, 3), Stride.flat(3, 1));

        assertThrows(NullPointerException.class, () -> {
            layout.isCongruentWith(null);
        });
    }

    @Test
    void testDeeplyNestedCongruence() {
        // [2, [3, [4, 5]]]
        Layout layout1 = Layout.of(
            Shape.of(2, Shape.of(3, Shape.of(4L, 5L))),
            Stride.of(60, Stride.of(20, Stride.of(5L, 1L)))
        );

        Layout layout2 = Layout.of(
            Shape.of(10, Shape.of(20, Shape.of(30L, 40L))),
            Stride.of(24000, Stride.of(1200, Stride.of(40L, 1L)))
        );

        assertTrue(layout1.isCongruentWith(layout2));
    }

    @Test
    void testModeAtPreservesCongruence() {
        Layout layout1 = Layout.of(
            Shape.of(2, Shape.of(3L, 4L)),
            Stride.of(12, Stride.of(4L, 1L))
        );

        Layout layout2 = Layout.of(
            Shape.of(10, Shape.of(20L, 30L)),
            Stride.of(600, Stride.of(30L, 1L))
        );

        // Extract mode 1 from both
        Layout mode1_layout1 = layout1.modeAt(1);
        Layout mode1_layout2 = layout2.modeAt(1);

        // Modes should also be congruent
        assertTrue(mode1_layout1.isCongruentWith(mode1_layout2));
    }

    @Test
    void testFlattenPreservesCongruence() {
        Layout layout1 = Layout.of(
            Shape.of(2, Shape.of(3L, 4L)),
            Stride.of(12, Stride.of(4L, 1L))
        );

        Layout layout2 = Layout.of(
            Shape.of(10, Shape.of(20L, 30L)),
            Stride.of(600, Stride.of(30L, 1L))
        );

        Layout flat1 = layout1.flatten();
        Layout flat2 = layout2.flatten();

        // Flattened layouts should be congruent
        assertTrue(flat1.isCongruentWith(flat2));
    }
}
