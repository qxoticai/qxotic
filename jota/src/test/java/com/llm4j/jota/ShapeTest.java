package com.llm4j.jota;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class ShapeTest {

    // Test data
    static Stream<Arguments> shapeProvider() {
        return Stream.of(
                Arguments.of(Shape.scalar(), new long[]{}),
                Arguments.of(Shape.of(3), new long[]{3}),
                Arguments.of(Shape.of(2, 3), new long[]{2, 3}),
                Arguments.of(Shape.of(1, 0, 5), new long[]{1, 0, 5}),
                Arguments.of(Shape.of(0), new long[]{0})
        );
    }

    @Test
    void scalarShape() {
        Shape scalar = Shape.scalar();
        assertTrue(scalar.isScalar());
        assertFalse(scalar.hasZeroElements());
        assertEquals(0, scalar.rank());
        assertEquals(1, scalar.totalNumberOfElements());
    }

    @Test
    void singletonNotScalar() {
        Shape scalar = Shape.of(1, 1, 1);
        assertFalse(scalar.isScalar());
        assertEquals(1, scalar.totalNumberOfElements());
    }

    @ParameterizedTest
    @MethodSource("shapeProvider")
    void rankAndDimensions(Shape shape, long[] expectedDims) {
        assertEquals(expectedDims.length, shape.rank());
        assertArrayEquals(expectedDims, shape.toArray());
    }

    @ParameterizedTest
    @MethodSource("shapeProvider")
    void dimensionWithWrapAround(Shape shape, long[] expectedDims) {
        for (int i = 0; i < expectedDims.length; i++) {
            assertEquals(expectedDims[i], shape.dimension(i));
            // Test negative indices
            assertEquals(expectedDims[i], shape.dimension(i - expectedDims.length));
        }
    }

    @Test
    void dimensionThrowsForInvalidIndex() {
        Shape shape = Shape.of(2, 3);
        assertThrows(IllegalArgumentException.class, () -> shape.dimension(2));
        assertThrows(IllegalArgumentException.class, () -> shape.dimension(-3));
    }

    @Test
    void permute() {
        Shape original = Shape.of(2, 3, 4);
        Shape permuted = original.permute(2, 0, 1);
        assertArrayEquals(new long[]{4, 2, 3}, permuted.toArray());
    }

    @Test
    void permuteThrowsForInvalidPermutation() {
        Shape shape = Shape.of(2, 3);
        assertThrows(IllegalArgumentException.class, () -> shape.permute(0, 0));
        assertThrows(IllegalArgumentException.class, () -> shape.permute(0, 2));
        assertThrows(IllegalArgumentException.class, () -> shape.permute(1, -1));
    }

    @Test
    void swap() {
        Shape shape = Shape.of(2, 3, 4);
        Shape swapped = shape.swap(0, 2);
        assertArrayEquals(new long[]{4, 3, 2}, swapped.toArray());

        // Test negative indices
        Shape swappedNeg = shape.swap(-1, -3);
        assertArrayEquals(new long[]{4, 3, 2}, swappedNeg.toArray());
    }

    @Test
    void remove() {
        Shape shape = Shape.of(2, 3, 4, 5);
        Shape removed = shape.remove(1, 3);
        assertArrayEquals(new long[]{2, 4}, removed.toArray());

        // Test empty result
        Shape empty = Shape.of(2, 3).remove(0, 1);
        assertArrayEquals(new long[]{}, empty.toArray());
    }

    @Test
    void removeThrowsForInvalidIndices() {
        Shape shape = Shape.of(2, 3);
        assertThrows(IllegalArgumentException.class, () -> shape.remove(2));
    }

    @Test
    void keep() {
        Shape shape = Shape.of(2, 3, 4, 5);
        Shape kept = shape.keep(0, 2);
        assertArrayEquals(new long[]{2, 4}, kept.toArray());
    }

    @Test
    void append() {
        Shape shape = Shape.of(2, 3);
        Shape appended = shape.append(4, 5);
        assertArrayEquals(new long[]{2, 3, 4, 5}, appended.toArray());

        // Append to scalar
        Shape scalarAppended = Shape.scalar().append(1);
        assertArrayEquals(new long[]{1}, scalarAppended.toArray());
    }

    @Test
    void subShape() {
        Shape shape = Shape.of(1, 2, 3, 4);
        assertEquals(Shape.of(2, 3), shape.subShape(1, 3));
        assertEquals(shape, shape.subShape(0, shape.rank()));
    }

    @Test
    void replace() {
        Shape shape = Shape.of(2, 3);
        Shape replaced = shape.replace(1, 5);
        assertArrayEquals(new long[]{2, 5}, replaced.toArray());

        // No-op when same dimension
        assertSame(shape, shape.replace(1, 3));
    }

    @Test
    void prefixAndSuffix() {
        Shape shape = Shape.of(1, 2, 3, 4);
        assertEquals(Shape.of(1, 2), shape.prefix(2));
        assertEquals(Shape.of(3, 4), shape.suffix(2));
    }

    @Test
    void emptyShapeOperations() {
        Shape empty = Shape.of(0);
        assertEquals(0, empty.totalNumberOfElements());

        Shape appended = empty.append(1);
        assertArrayEquals(new long[]{0, 1}, appended.toArray());

        Shape removed = empty.remove(0);
        assertTrue(removed.isScalar());
    }

    @Test
    void zeroDimensionHandling() {
        Shape zeroDim = Shape.of(2, 0, 3);
        assertEquals(0, zeroDim.totalNumberOfElements());
        assertArrayEquals(new long[]{2, 0, 3}, zeroDim.toArray());

        Shape permuted = zeroDim.permute(2, 1, 0);
        assertArrayEquals(new long[]{3, 0, 2}, permuted.toArray());
    }

    @Test
    void sameAsComparison() {
        assertTrue(Shape.sameAs(Shape.of(2, 3), Shape.of(2, 3)));
        assertFalse(Shape.sameAs(Shape.of(2, 3), Shape.of(3, 2)));
        assertTrue(Shape.sameAs(Shape.of(0), Shape.of(0)));
    }

    @Test
    void invalidShapeCreation() {
        assertThrows(IllegalArgumentException.class, () -> Shape.of(-1));
        assertThrows(IllegalArgumentException.class, () -> Shape.of(1, -1));
    }

    @Test
    void emptyPermutation() {
        Shape scalar = Shape.scalar();
        assertDoesNotThrow(() -> scalar.permute(new int[0]));
    }

    @Test
    void invalidSubShapeRange() {
        Shape shape = Shape.of(1, 2, 3);
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> shape.subShape(-1, 2));
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> shape.subShape(1, 4));
    }

    @Test
    void negativeInRemove() {
        Shape s = Shape.of(1, 2, 3);
        assertArrayEquals(new long[]{1, 3}, s.remove(-2).toArray());
    }

    @Test
    void multipleZeroDims() {
        Shape s = Shape.of(0, 0, 0);
        assertEquals(0, s.totalNumberOfElements());
        assertArrayEquals(new long[]{0, 0, 0}, s.toArray());
    }

    @Test
    void operationsOnZeroDim() {
        Shape s = Shape.of(2, 0, 3);
        assertEquals(Shape.of(0, 3), s.remove(0));
        assertEquals(Shape.of(0, 3), s.suffix(2));
        assertEquals(0, s.totalNumberOfElements());
    }

    @Test
    void operationChaining() {
        Shape s = Shape.of(1, 2, 3, 4)
                .remove(1)
                .replace(0, 5)
                .append(6);

        assertArrayEquals(new long[]{5, 3, 4, 6}, s.toArray());
    }

    @Test
    void equalityContracts() {
        // Different objects, same dimensions
        Shape s1 = Shape.of(1, 2);
        Shape s2 = Shape.of(1, 2);
        assertEquals(s1, s2);
        assertEquals(s1.hashCode(), s2.hashCode());

        // Empty vs scalar
        assertNotEquals(Shape.scalar(), Shape.of(0));
    }

    @Test
    void keepAllDimensions() {
        Shape s = Shape.of(1, 2);
        assertEquals(s, s.keep(0, 1));
    }

    @Test
    void wrapAroundDocumentation() {
        Shape s = Shape.of(1, 2, 3);
        // Verify documented behavior for negative indices
        assertEquals(3, s.dimension(-1));
        assertEquals(1, s.dimension(-3));
    }

    @Test
    void unmodifiableToArray() {
        Shape s = Shape.of(1, 2, 3);
        long[] dims = s.toArray();
        dims[0] = 42;
        assertEquals(1, s.dimension(0));
    }

    @Test
    void squeezeAll_removesAllSingletonDimensions() {
        Shape shape = Shape.of(1, 3, 1, 5, 1);
        Shape squeezed = shape.squeezeAll();
        assertEquals(Shape.of(3, 5), squeezed);
    }

    @Test
    void squeezeAll_noOpWhenNoSingletonDimensions() {
        Shape shape = Shape.of(2, 3, 4);
        Shape squeezed = shape.squeezeAll();
        assertEquals(shape, squeezed);
    }

    @Test
    void squeezeAll_emptyShapeRemainsEmpty() {
        Shape shape = Shape.of(1, 0, 1);
        Shape squeezed = shape.squeezeAll();
        assertEquals(Shape.of(0), squeezed);
    }

    @Test
    void squeezeAll_allSingletonBecomesScalar() {
        Shape shape = Shape.of(1, 1, 1);
        Shape squeezed = shape.squeezeAll();
        assertEquals(Shape.scalar(), squeezed);
    }

    // ======================== squeeze(int axis) Tests ========================

    @Test
    void squeeze_removesSpecifiedSingletonDimension() {
        Shape shape = Shape.of(1, 3, 1, 5);
        Shape squeezed = shape.squeeze(2); // Remove 3rd dimension (0-based)
        assertEquals(Shape.of(1, 3, 5), squeezed);
    }

    @Test
    void squeeze_handlesNegativeAxis() {
        Shape shape = Shape.of(1, 3, 1, 5);
        Shape squeezed = shape.squeeze(-2); // Same as axis 2
        assertEquals(Shape.of(1, 3, 5), squeezed);
    }

    @Test
    void squeeze_throwsWhenAxisOutOfBounds() {
        Shape shape = Shape.of(1, 3, 5);
        assertThrows(IllegalArgumentException.class, () -> shape.squeeze(3));
        assertThrows(IllegalArgumentException.class, () -> shape.squeeze(-4));
    }

    @Test
    void squeeze_throwsWhenDimensionNotSingleton() {
        Shape shape = Shape.of(1, 3, 5);
        assertThrows(IllegalArgumentException.class, () -> shape.squeeze(1));
        assertThrows(IllegalArgumentException.class, () -> shape.squeeze(2));
    }

    @Test
    void squeeze_firstDimension() {
        Shape shape = Shape.of(1, 3, 5);
        Shape squeezed = shape.squeeze(0);
        assertEquals(Shape.of(3, 5), squeezed);
    }

    @Test
    void squeeze_lastDimension() {
        Shape shape = Shape.of(3, 5, 1);
        Shape squeezed = shape.squeeze(-1);
        assertEquals(Shape.of(3, 5), squeezed);
    }

    @Test
    void squeeze_onlyRemovesSpecifiedAxis() {
        Shape shape = Shape.of(1, 1, 3, 1);
        Shape squeezed = shape.squeeze(1); // Only remove second dimension
        assertEquals(Shape.of(1, 3, 1), squeezed);
    }

    // ======================== Edge Cases ========================

    @Test
    void squeeze_scalarShapeThrows() {
        Shape scalar = Shape.scalar(); // ()
        assertThrows(IllegalArgumentException.class, () -> scalar.squeeze(0));
    }

    @Test
    void squeezeAll_scalarShapeRemainsScalar() {
        Shape scalar = Shape.scalar();
        assertEquals(scalar, scalar.squeezeAll());
    }

    @Test
    void squeeze_emptyDimensionPreserved() {
        Shape shape = Shape.of(1, 0, 1);
        Shape squeezed = shape.squeeze(0);
        assertEquals(Shape.of(0, 1), squeezed);
    }

    @Test
    void squeezeAll_removesMultipleNonAdjacentSingletons() {
        Shape shape = Shape.of(1, 2, 1, 3, 1, 4, 1);
        Shape squeezed = shape.squeezeAll();
        assertEquals(Shape.of(2, 3, 4), squeezed);
    }

    @Test
    public void testUnsqueezeBasicCases() {
        // Scalar case
        Shape scalar = Shape.scalar();
        assertArrayEquals(new long[]{1}, scalar.unsqueeze(0).toArray());

        // Vector cases
        Shape vector = Shape.of(3);
        assertArrayEquals(new long[]{1, 3}, vector.unsqueeze(0).toArray());
        assertArrayEquals(new long[]{3, 1}, vector.unsqueeze(1).toArray());
        assertArrayEquals(new long[]{3, 1}, vector.unsqueeze(-1).toArray());

        // Matrix cases
        Shape matrix = Shape.of(2, 3);
        assertArrayEquals(new long[]{1, 2, 3}, matrix.unsqueeze(0).toArray());
        assertArrayEquals(new long[]{2, 1, 3}, matrix.unsqueeze(1).toArray());
        assertArrayEquals(new long[]{2, 3, 1}, matrix.unsqueeze(2).toArray());
        assertArrayEquals(new long[]{2, 1, 3}, matrix.unsqueeze(-2).toArray());
    }

    @Test
    public void testUnsqueezeEdgeCases() {
        Shape empty = Shape.of();
        assertArrayEquals(new long[]{1}, empty.unsqueeze(0).toArray());

        Shape singleDim = Shape.of(1);
        assertArrayEquals(new long[]{1, 1}, singleDim.unsqueeze(1).toArray());

        Shape alreadyUnsqueezed = Shape.of(1, 2, 1, 3);
        assertArrayEquals(new long[]{1, 1, 2, 1, 3}, alreadyUnsqueezed.unsqueeze(1).toArray());
    }

    @Test
    public void testInsertBasicCases() {
        // Scalar case
        Shape scalar = Shape.scalar();
        assertArrayEquals(new long[]{5}, scalar.insert(0, 5).toArray());

        // Vector cases
        Shape vector = Shape.of(3);
        assertArrayEquals(new long[]{5, 3}, vector.insert(0, 5).toArray());
        assertArrayEquals(new long[]{3, 5}, vector.insert(1, 5).toArray());
        assertArrayEquals(new long[]{3, 5}, vector.insert(-1, 5).toArray());

        // Matrix cases
        Shape matrix = Shape.of(2, 3);
        assertArrayEquals(new long[]{5, 2, 3}, matrix.insert(0, 5).toArray());
        assertArrayEquals(new long[]{2, 5, 3}, matrix.insert(1, 5).toArray());
        assertArrayEquals(new long[]{2, 3, 5}, matrix.insert(2, 5).toArray());
        assertArrayEquals(new long[]{2, 5, 3}, matrix.insert(-2, 5).toArray());
    }

    @Test
    public void testInsertEdgeCases() {
        Shape empty = Shape.of();
        assertArrayEquals(new long[]{5}, empty.insert(0, 5).toArray());

        Shape singleDim = Shape.of(1);
        assertArrayEquals(new long[]{5, 1}, singleDim.insert(0, 5).toArray());
        assertArrayEquals(new long[]{1, 5}, singleDim.insert(1, 5).toArray());

        Shape alreadyInserted = Shape.of(2, 3, 4);
        assertArrayEquals(new long[]{2, 5, 3, 4}, alreadyInserted.insert(1, 5).toArray());

        assertArrayEquals(new long[]{0}, empty.insert(0, 0).toArray());
    }

    @Test
    public void testUnsqueezeInvalidAxis() {
        Shape shape = Shape.of(2, 3);
        assertThrows(IllegalArgumentException.class, () -> shape.unsqueeze(3));
        assertThrows(IllegalArgumentException.class, () -> shape.unsqueeze(-4));
    }

    @Test
    public void testInsertInvalidArguments() {
        Shape shape = Shape.of(2, 3);
        assertThrows(IllegalArgumentException.class, () -> shape.insert(3, 5));
        assertThrows(IllegalArgumentException.class, () -> shape.insert(-4, 5));
        assertThrows(IllegalArgumentException.class, () -> shape.insert(1, -1));
    }

    @Test
    public void testUnsqueezeAsSpecialCaseOfInsert() {
        Shape shape = Shape.of(2, 3);
        assertArrayEquals(shape.unsqueeze(1).toArray(), shape.insert(1, 1).toArray());
    }

    @Test
    public void testInsertMaintainsTotalElements() {
        Shape original = Shape.of(2, 3);
        assertEquals(original.totalNumberOfElements() * 4, original.insert(1, 4).totalNumberOfElements());
        assertEquals(original.totalNumberOfElements(), original.unsqueeze(1).totalNumberOfElements());
    }
}
