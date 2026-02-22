package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

/**
 * Tests demonstrating F-bounded polymorphism type safety. Shape methods return Shape, Stride
 * methods return Stride. No runtime type checking or casting required.
 */
class FBoundedPolymorphismTest {

    @Test
    void testShapeMethodsReturnShape() {
        Shape shape = Shape.flat(2, 3, 4);

        // All these methods return Shape - no casting needed
        Shape mode = shape.modeAt(0);
        Shape flattened = shape.flatten();
        Shape replaced = shape.replace(1, Shape.of(10L));
        Shape inserted = shape.insert(1, Shape.of(5L));
        Shape removed = shape.remove(1);
        Shape permuted = shape.permute(2, 1, 0);

        // Can chain Shape-specific methods
        long size = shape.replace(0, Shape.of(10L)).size(0);
        assertEquals(10, size);

        // All return types are Shape
        assertTrue(mode instanceof Shape);
        assertTrue(flattened instanceof Shape);
        assertTrue(replaced instanceof Shape);
        assertTrue(inserted instanceof Shape);
        assertTrue(removed instanceof Shape);
        assertTrue(permuted instanceof Shape);
    }

    @Test
    void testStrideMethodsReturnStride() {
        Stride stride = Stride.flat(100, 10, 1);

        // All these methods return Stride - no casting needed
        Stride mode = stride.modeAt(0);
        Stride flattened = stride.flatten();
        Stride replaced = stride.replace(1, Stride.of(50L));
        Stride inserted = stride.insert(1, Stride.of(5L));
        Stride removed = stride.remove(1);
        Stride permuted = stride.permute(2, 1, 0);

        // Can chain Stride-specific methods
        Stride rowMajor = Stride.rowMajor(Shape.flat(2, 3, 4));
        Stride modified = rowMajor.replace(0, Stride.of(1000L));

        // All return types are Stride
        assertTrue(mode instanceof Stride);
        assertTrue(flattened instanceof Stride);
        assertTrue(replaced instanceof Stride);
        assertTrue(inserted instanceof Stride);
        assertTrue(removed instanceof Stride);
        assertTrue(permuted instanceof Stride);
    }

    @Test
    void testTypeSafety() {
        Shape shape = Shape.flat(2, 3, 4);
        Stride stride = Stride.flat(12, 4, 1);

        // Cannot mix types - compile-time safety
        // shape.replace(0, stride); // Won't compile!
        // stride.replace(0, shape); // Won't compile!

        // But can use as template (accepts NestedTuple<?>)
        Shape shapeFromStride = Shape.template(stride, 10, 20, 30);
        Stride strideFromShape = Stride.template(shape, 100, 10, 1);

        assertEquals(3, shapeFromStride.rank());
        assertEquals(3, strideFromShape.rank());
    }

    @Test
    void testCongruenceWithWildcard() {
        Shape shape = Shape.flat(2, 3, 4);
        Stride stride = Stride.flat(12, 4, 1);

        // isCongruentWith accepts NestedTuple<?> - can compare Shape and Stride
        assertTrue(shape.isCongruentWith(stride));
        assertTrue(stride.isCongruentWith(shape));

        // Different structures are not congruent
        Shape nested = Shape.of(2, Shape.of(3L, 4L));
        assertFalse(shape.isCongruentWith(nested));
    }

    @Test
    void testMethodChaining() {
        // Complex chaining with type-safe operations
        Shape result =
                Shape.flat(2, 3, 4)
                        .insert(1, Shape.of(10L)) // Returns Shape
                        .replace(2, Shape.of(20L, 30L)) // Returns Shape
                        .permute(3, 0, 1, 2) // Returns Shape
                        .remove(1); // Returns Shape

        assertEquals(3, result.rank());

        // Stride chaining
        Stride strideResult =
                Stride.flat(100, 10, 1)
                        .insert(0, Stride.of(1000L)) // Returns Stride
                        .replace(1, Stride.of(200L)) // Returns Stride
                        .permute(3, 2, 1, 0); // Returns Stride

        assertEquals(4, strideResult.rank());
    }
}
