package ai.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

/** Tests for Shape manipulation methods: replace, insert, remove, permute */
class ShapeManipulationTest {

    @Test
    void testReplaceFlatShape() {
        Shape shape = Shape.flat(2, 3, 4);

        // Replace middle dimension with a single value
        Shape replaced = shape.replace(1, Shape.of(10L));
        assertEquals(3, replaced.rank());
        assertArrayEquals(new long[] {2, 10, 4}, replaced.toArray());

        // Replace with nested shape
        Shape replaced2 = shape.replace(1, Shape.of(5L, 6L));
        assertEquals(3, replaced2.rank());
        assertEquals(4, replaced2.flatRank());
        assertArrayEquals(new long[] {2, 5, 6, 4}, replaced2.toArray());

        // Replace using negative index
        Shape replaced3 = shape.replace(-1, Shape.of(100L));
        assertArrayEquals(new long[] {2, 3, 100}, replaced3.toArray());
    }

    @Test
    void testReplaceNestedShape() {
        // (2, (3, 4), 5)
        Shape shape = Shape.of(2, Shape.of(3L, 4L), 5);

        // Replace nested mode with flat
        Shape replaced = shape.replace(1, Shape.of(10L));
        assertEquals(3, replaced.rank());
        assertEquals(3, replaced.flatRank());
        assertArrayEquals(new long[] {2, 10, 5}, replaced.toArray());

        // Replace with another nested shape
        Shape replaced2 = shape.replace(1, Shape.of(6L, 7L, 8L));
        assertEquals(3, replaced2.rank());
        assertEquals(5, replaced2.flatRank());
        assertArrayEquals(new long[] {2, 6, 7, 8, 5}, replaced2.toArray());
    }

    @Test
    void testInsertFlatShape() {
        Shape shape = Shape.flat(2, 3, 4);

        // Insert at beginning
        Shape inserted = shape.insert(0, Shape.of(10L));
        assertEquals(4, inserted.rank());
        assertArrayEquals(new long[] {10, 2, 3, 4}, inserted.toArray());

        // Insert in middle
        Shape inserted2 = shape.insert(1, Shape.of(10L));
        assertEquals(4, inserted2.rank());
        assertArrayEquals(new long[] {2, 10, 3, 4}, inserted2.toArray());

        // Insert at end
        Shape inserted3 = shape.insert(3, Shape.of(10L));
        assertEquals(4, inserted3.rank());
        assertArrayEquals(new long[] {2, 3, 4, 10}, inserted3.toArray());

        // Insert nested shape
        Shape inserted4 = shape.insert(1, Shape.of(5L, 6L));
        assertEquals(4, inserted4.rank());
        assertEquals(5, inserted4.flatRank());
        assertArrayEquals(new long[] {2, 5, 6, 3, 4}, inserted4.toArray());
    }

    @Test
    void testInsertNestedShape() {
        // (2, (3, 4))
        Shape shape = Shape.of(2, Shape.of(3L, 4L));

        // Insert at beginning
        Shape inserted = shape.insert(0, Shape.of(10L));
        assertEquals(3, inserted.rank());
        assertEquals(4, inserted.flatRank());
        assertArrayEquals(new long[] {10, 2, 3, 4}, inserted.toArray());

        // Insert at end
        Shape inserted2 = shape.insert(2, Shape.of(10L));
        assertEquals(3, inserted2.rank());
        assertEquals(4, inserted2.flatRank());
        assertArrayEquals(new long[] {2, 3, 4, 10}, inserted2.toArray());
    }

    @Test
    void testRemoveFlatShape() {
        Shape shape = Shape.flat(2, 3, 4, 5);

        // Remove first
        Shape removed = shape.remove(0);
        assertEquals(3, removed.rank());
        assertArrayEquals(new long[] {3, 4, 5}, removed.toArray());

        // Remove middle
        Shape removed2 = shape.remove(1);
        assertEquals(3, removed2.rank());
        assertArrayEquals(new long[] {2, 4, 5}, removed2.toArray());

        // Remove last
        Shape removed3 = shape.remove(3);
        assertEquals(3, removed3.rank());
        assertArrayEquals(new long[] {2, 3, 4}, removed3.toArray());

        // Remove using negative index
        Shape removed4 = shape.remove(-1);
        assertEquals(3, removed4.rank());
        assertArrayEquals(new long[] {2, 3, 4}, removed4.toArray());
    }

    @Test
    void testRemoveNestedShape() {
        // (2, (3, 4), 5)
        Shape shape = Shape.of(2, Shape.of(3L, 4L), 5);

        // Remove nested mode
        Shape removed = shape.remove(1);
        assertEquals(2, removed.rank());
        assertEquals(2, removed.flatRank());
        assertArrayEquals(new long[] {2, 5}, removed.toArray());

        // Remove first
        Shape removed2 = shape.remove(0);
        assertEquals(2, removed2.rank());
        assertEquals(3, removed2.flatRank());
        assertArrayEquals(new long[] {3, 4, 5}, removed2.toArray());
    }

    @Test
    void testRemoveSingleMode() {
        Shape shape = Shape.of(10L);
        Shape removed = shape.remove(0);
        assertTrue(removed.isScalar());
        assertEquals(0, removed.rank());
    }

    @Test
    void testPermuteFlatShape() {
        Shape shape = Shape.flat(2, 3, 4);

        // Identity permutation
        Shape permuted1 = shape.permute(0, 1, 2);
        assertArrayEquals(new long[] {2, 3, 4}, permuted1.toArray());

        // Reverse
        Shape permuted2 = shape.permute(2, 1, 0);
        assertArrayEquals(new long[] {4, 3, 2}, permuted2.toArray());

        // Swap first and last
        Shape permuted3 = shape.permute(2, 1, 0);
        assertArrayEquals(new long[] {4, 3, 2}, permuted3.toArray());

        // Custom permutation
        Shape permuted4 = shape.permute(1, 2, 0);
        assertArrayEquals(new long[] {3, 4, 2}, permuted4.toArray());

        // Negative indices
        Shape permuted5 = shape.permute(-1, -2, -3);
        assertArrayEquals(new long[] {4, 3, 2}, permuted5.toArray());
    }

    @Test
    void testPermuteNestedShape() {
        // (2, (3, 4), 5)
        Shape shape = Shape.of(2, Shape.of(3L, 4L), 5);

        // Reverse modes
        Shape permuted = shape.permute(2, 1, 0);
        assertEquals(3, permuted.rank());
        assertEquals(4, permuted.flatRank());
        assertArrayEquals(new long[] {5, 3, 4, 2}, permuted.toArray());

        // Swap first and middle
        Shape permuted2 = shape.permute(1, 0, 2);
        assertEquals(3, permuted2.rank());
        assertEquals(4, permuted2.flatRank());
        assertArrayEquals(new long[] {3, 4, 2, 5}, permuted2.toArray());
    }

    @Test
    void testPermuteValidation() {
        Shape shape = Shape.flat(2, 3, 4);

        // Wrong number of axes
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    shape.permute(0, 1);
                });

        // Duplicate axis
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    shape.permute(0, 0, 1);
                });
    }

    @Test
    void testChainedOperations() {
        // Start with (2, 3, 4)
        Shape shape = Shape.flat(2, 3, 4);

        // Insert, then replace, then permute
        Shape result =
                shape.insert(1, Shape.of(10L)) // (2, 10, 3, 4)
                        .replace(2, Shape.of(20L, 30L)) // (2, 10, (20, 30), 4)
                        .permute(3, 0, 1, 2); // (4, 2, 10, (20, 30))

        assertEquals(4, result.rank());
        assertEquals(5, result.flatRank());
        assertArrayEquals(new long[] {4, 2, 10, 20, 30}, result.toArray());
    }

    @Test
    void testReplacePreservesNesting() {
        // ((1, 2), (3, 4))
        Shape shape = Shape.of(Shape.of(1L, 2L), Shape.of(3L, 4L));

        // Replace first mode
        Shape replaced = shape.replace(0, Shape.of(10L, 20L));
        assertEquals(2, replaced.rank());
        assertEquals(4, replaced.flatRank());
        assertArrayEquals(new long[] {10, 20, 3, 4}, replaced.toArray());
    }
}
