package com.qxotic;

import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for isCongruent() method and Layout flatRank validation.
 */
class CongruenceTest {

    @Test
    void testFlatCongruence() {
        Shape shape = Shape.flat(2, 3, 4);
        Stride stride = Stride.flat(12, 4, 1);

        assertTrue(shape.isCongruentWith(stride));
    }

    @Test
    void testNestedCongruenceSameStructure() {
        // Both: [2, [3, 4]]
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Stride stride = Stride.of(12, Stride.of(4L, 1L));

        assertTrue(shape.isCongruentWith(stride));
    }

    @Test
    void testNestedCongruenceDifferentStructure() {
        // Shape: [2, [3, 4]] - pattern "[,[,]]"
        // Stride: [[12, 4], 1] - pattern "[[,],]"
        // Note: We can only test different nesting between two shapes since
        // Stride doesn't support pattern-based creation
        Shape shape1 = Shape.pattern("[,[,]]", 2, 3, 4);
        Shape shape2 = Shape.pattern("[[,],]", 2, 3, 4);

        assertFalse(shape1.isCongruentWith(shape2)); // Different nesting structure
    }

    @Test
    void testCongruenceFlatVsNested() {
        Shape shape = Shape.flat(2, 3, 4);
        Stride stride = Stride.of(12, Stride.of(4L, 1L));

        assertFalse(shape.isCongruentWith(stride));
    }

    @Test
    void testCongruenceDifferentRanks() {
        Shape shape = Shape.flat(2, 3);
        Stride stride = Stride.flat(12, 4, 1);

        assertFalse(shape.isCongruentWith(stride));
    }

    @Test
    void testCongruenceDifferentFlatRanks() {
        // Shape: [2, [3, 4]] (flatRank=3), Stride: [[12, 4], [2, 1]] (flatRank=4)
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Stride stride = Stride.of(Stride.of(12L, 4L), Stride.of(2L, 1L));

        assertFalse(shape.isCongruentWith(stride));
    }

    @Test
    void testCongruenceScalar() {
        Shape shape = Shape.scalar();
        Stride stride = Stride.scalar();

        assertTrue(shape.isCongruentWith(stride));
    }

    @Test
    void testCongruenceDeeplyNested() {
        // Both: [2, [3, [4, 5]]]
        Shape shape = Shape.of(2, Shape.of(3, Shape.of(4L, 5L)));
        Stride stride = Stride.of(60, Stride.of(20, Stride.of(5L, 1L)));

        assertTrue(shape.isCongruentWith(stride));
    }

    @Test
    void testCongruenceNull() {
        Shape shape = Shape.flat(2, 3, 4);

        assertThrows(NullPointerException.class, () -> {
            shape.isCongruentWith(null);
        });
    }

    @Test
    void testLayoutCongruent() {
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Stride stride = Stride.of(12, Stride.of(4L, 1L));
        Layout layout = Layout.of(shape, stride);

        // Check self-congruence: shape and stride have same structure
        assertTrue(shape.isCongruentWith(stride));

        // Check layout-to-layout congruence
        Layout layout2 = Layout.of(Shape.of(10, Shape.of(20L, 30L)), Stride.of(600, Stride.of(30L, 1L)));
        assertTrue(layout.isCongruentWith(layout2));
    }

    @Test
    void testLayoutRowMajorCongruent() {
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Layout layout = Layout.rowMajor(shape);

        // Row-major should preserve shape structure
        assertTrue(shape.isCongruentWith(layout.stride()));
    }

    @Test
    void testLayoutColumnMajorCongruent() {
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Layout layout = Layout.columnMajor(shape);

        // Column-major should preserve shape structure
        assertTrue(shape.isCongruentWith(layout.stride()));
    }

    @Test
    void testLayoutFlatRankMismatchThrows() {
        Shape shape = Shape.flat(2, 3);
        Stride stride = Stride.flat(12, 4, 1);

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> {
            Layout.of(shape, stride);
        });

        assertTrue(ex.getMessage().contains("same flatRank"));
        assertTrue(ex.getMessage().contains("shape.flatRank()=2"));
        assertTrue(ex.getMessage().contains("stride.flatRank()=3"));
    }

    @Test
    void testLayoutModeAtPreservesCongruence() {
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Layout layout = Layout.rowMajor(shape);

        // Check self-congruence
        assertTrue(layout.shape().isCongruentWith(layout.stride()));

        // Extract mode 1 - should preserve congruence
        Layout mode1 = layout.modeAt(1);
        assertTrue(mode1.shape().isCongruentWith(mode1.stride()));
    }

    @Test
    void testLayoutFlattenPreservesCongruence() {
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Layout layout = Layout.rowMajor(shape);

        // Check self-congruence before flattening
        assertTrue(layout.shape().isCongruentWith(layout.stride()));

        // Flatten should preserve congruence
        Layout flattened = layout.flatten();
        assertTrue(flattened.shape().isCongruentWith(flattened.stride()));
    }

    @Test
    void testPatternBasedCongruence() {
        Shape shape = Shape.pattern("[batch, [N, M]]", 2, 3, 4);
        Stride stride = Stride.template(shape, 12, 4, 1);

        assertTrue(shape.isCongruentWith(stride));
    }

    @Test
    void testTemplateBasedCongruence() {
        Shape template = Shape.pattern("[,[,]]", 1, 2, 3);
        Shape shape = Shape.template(template, 10, 20, 30);
        Stride stride = Stride.template(template, 100, 10, 1);

        assertTrue(shape.isCongruentWith(stride));
    }
}
