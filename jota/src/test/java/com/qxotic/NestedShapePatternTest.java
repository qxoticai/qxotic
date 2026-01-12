package com.qxotic;

import com.qxotic.jota.Shape;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class NestedShapePatternTest {

    @Test
    void testSimpleNestedPattern() {
        // (batch, (N, M)) with dims 2, 3, 4
        // Should create: rank=2, mode0=2, mode1=3*4=12
        Shape shape = Shape.pattern("(batch, (N, M))", 2, 3, 4);

        assertEquals(2, shape.rank(), "Should have rank 2");
        assertEquals(3, shape.flatRank(), "Should have 3 flat dimensions");
        assertEquals(2, shape.size(0), "First mode size should be 2");
        assertEquals(12, shape.size(1), "Second mode size should be 3*4=12");
        assertEquals(24, shape.size(), "Total size should be 2*3*4=24");

        assertArrayEquals(new long[]{2, 3, 4}, shape.toArray());
    }

    @Test
    void testFlatPattern() {
        // (a, b, c) with dims 2, 3, 4
        // Should be equivalent to Shape.of(2, 3, 4)
        Shape shape = Shape.pattern("(a, b, c)", 2, 3, 4);

        assertEquals(3, shape.rank());
        assertEquals(3, shape.flatRank());
        assertTrue(shape.isFlat());
        assertEquals(2, shape.size(0));
        assertEquals(3, shape.size(1));
        assertEquals(4, shape.size(2));
    }

    @Test
    void testDeeplyNestedPattern() {
        // (a, (b, (c, d))) with dims 2, 3, 4, 5
        // rank=2: mode0=2, mode1=3*4*5=60
        Shape shape = Shape.pattern("(a, (b, (c, d)))", 2, 3, 4, 5);

        assertEquals(2, shape.rank());
        assertEquals(4, shape.flatRank());
        assertEquals(2, shape.size(0));
        assertEquals(60, shape.size(1), "Second mode should be 3*4*5=60");
    }

    @Test
    void testMultipleNestedGroups() {
        // ((a, b), (c, d)) with dims 2, 3, 4, 5
        // rank=2: mode0=2*3=6, mode1=4*5=20
        Shape shape = Shape.pattern("((a, b), (c, d))", 2, 3, 4, 5);

        assertEquals(2, shape.rank());
        assertEquals(4, shape.flatRank());
        assertEquals(6, shape.size(0), "First mode should be 2*3=6");
        assertEquals(20, shape.size(1), "Second mode should be 4*5=20");
    }

    @Test
    void testPatternWithPlaceholders() {
        // Use placeholders like '_' for anonymous dimensions
        Shape shape = Shape.pattern("(_, (_, _))", 2, 3, 4);

        assertEquals(2, shape.rank());
        assertEquals(3, shape.flatRank());
        assertEquals(2, shape.size(0));
        assertEquals(12, shape.size(1));
    }

    @Test
    void testScalarPattern() {
        // "()" means scalar/empty nesting (no dimensions)
        Shape shape = Shape.pattern("()");

        assertTrue(shape.isScalar());
        assertEquals(0, shape.rank());
        assertEquals(0, shape.flatRank());
        assertEquals(1, shape.size(), "Scalar has size 1");
        assertArrayEquals(new long[]{}, shape.toArray());
    }

    @Test
    void testSingletonPattern() {
        // "(_)" or "(size)" means a singleton with one dimension
        Shape shape1 = Shape.pattern("(_)", 5);
        Shape shape2 = Shape.pattern("(size)", 5);

        assertEquals(1, shape1.rank());
        assertEquals(1, shape1.flatRank());
        assertEquals(5, shape1.size(0));
        assertEquals(5, shape1.size());
        assertArrayEquals(new long[]{5}, shape1.toArray());

        assertEquals(1, shape2.rank());
        assertEquals(1, shape2.flatRank());
        assertEquals(5, shape2.size(0));
        assertEquals(5, shape2.size());
        assertArrayEquals(new long[]{5}, shape2.toArray());
    }

    @Test
    void testEmptyIdentifiersNotAllowed() {
        // Empty identifiers should throw an exception
        assertThrows(IllegalArgumentException.class, () -> {
            Shape.pattern("(,)", 2, 3);
        }, "Empty identifiers should not be allowed");

        assertThrows(IllegalArgumentException.class, () -> {
            Shape.pattern("(, (, ))", 2, 3, 4);
        }, "Empty identifiers should not be allowed");
    }

    @Test
    void testMalformedPatterns() {
        // Empty nested parentheses (()) are not allowed
        assertThrows(IllegalArgumentException.class, () -> {
            Shape.pattern("(())");
        }, "Empty nested parentheses should not be allowed");

        // Empty nested parentheses as part of a sequence
        assertThrows(IllegalArgumentException.class, () -> {
            Shape.pattern("((), _)", 5);
        }, "Empty nested parentheses in sequence should not be allowed");

        assertThrows(IllegalArgumentException.class, () -> {
            Shape.pattern("(_, ())", 5);
        }, "Empty nested parentheses in sequence should not be allowed");
    }

    @Test
    void testNonNormalizedPatterns() {
        // Single-element nested parentheses ((_)) are not normalized
        assertThrows(IllegalArgumentException.class, () -> {
            Shape.pattern("((_))", 5);
        }, "Single-element nested parentheses should be rejected as non-normalized");

        // ((a)) should also be rejected
        assertThrows(IllegalArgumentException.class, () -> {
            Shape.pattern("((dim))", 5);
        }, "Single-element nested parentheses should be rejected as non-normalized");

        // More deeply nested single elements
        assertThrows(IllegalArgumentException.class, () -> {
            Shape.pattern("(((_)))", 5);
        }, "Deeply nested single elements should be rejected");

        // Single element nested within a valid structure
        assertThrows(IllegalArgumentException.class, () -> {
            Shape.pattern("(a, ((b)))", 2, 3);
        }, "Single-element nested parentheses in sequence should be rejected");
    }

    @Test
    void testMismatchedDimensions() {
        // Pattern expects 3 dims but only 2 provided
        assertThrows(IllegalArgumentException.class, () -> {
            Shape.pattern("(a, (b, c))", 2, 3);
        });
    }

    @Test
    void testToString() {
        Shape shape = Shape.pattern("(batch, (N, M))", 2, 3, 4);
        assertEquals("(2, (3, 4))", shape.toString());
        assertEquals("()", Shape.pattern("()").toString());
        assertEquals("(42)", Shape.pattern("(_)", 42).toString());
        assertEquals("(2, ((3, 4), 5), 6)", Shape.pattern("(_, ((_, _), _), _)", 2, 3, 4, 5, 6).toString());

        assertEquals("(2, (3, (4, 5)), 6, ((7, 8), 9), 10)", Shape.pattern("(_, (_, (_, _)), _, ((_, _), _), _)", 2, 3, 4, 5, 6, 7, 8, 9, 10).toString());
    }

    @Test
    void testModeAtForNestedTuple() {
        Shape shape = Shape.pattern("(_, ((_, _), _), _)", 2, 3, 4, 5, 6);
        assertEquals("(2, ((3, 4), 5), 6)", shape.toString());

        Shape mode1 = shape.modeAt(1);
        assertEquals("((3, 4), 5)", mode1.toString());

        Shape mode1Mode0 = mode1.modeAt(0);
        assertEquals("(3, 4)", mode1Mode0.toString());
    }

    @Test
    void testDeeperNesting() {
        // (_,(_,_,(_,_))) with dims 2, 3, 4, 5, 6
        // Should create: (2, (3, 4, (5, 6)))
        // rank=2: mode0=2, mode1 is nested (3,4,(5,6))
        Shape shape = Shape.pattern("(_,(_,_,(_,_)))", 2, 3, 4, 5, 6);

        System.out.println("Top-level shape: " + shape);
        System.out.println("Top-level rank: " + shape.rank() + ", flatRank: " + shape.flatRank());

        assertEquals(2, shape.rank(), "Top-level should have rank 2");
        assertEquals(5, shape.flatRank(), "Should have 5 flat dimensions");
        assertEquals(2, shape.size(0), "Mode 0 size should be 2");
        assertEquals(3 * 4 * 5 * 6, shape.size(1), "Mode 1 size should be 3*4*5*6");

        // Extract mode 1 and check its structure
        Shape mode1 = shape.modeAt(1);
        System.out.println("Mode 1 shape: " + mode1);
        System.out.println("Mode 1 rank: " + mode1.rank() + ", flatRank: " + mode1.flatRank());
        System.out.println("Mode 1 toArray: " + Arrays.toString(mode1.toArray()));

        assertEquals(3, mode1.rank(), "Mode 1 should have rank 3");
        assertEquals(4, mode1.flatRank(), "Mode 1 should have 4 flat dimensions");
        assertEquals(3, mode1.size(0), "Mode 1, mode 0 size should be 3");
        assertEquals(4, mode1.size(1), "Mode 1, mode 1 size should be 4");
        assertEquals(5 * 6, mode1.size(2), "Mode 1, mode 2 size should be 5*6");

        // Extract mode 1's mode 2 and check it
        Shape mode1_mode2 = mode1.modeAt(2);
        assertEquals(2, mode1_mode2.flatRank());
        assertArrayEquals(new long[]{5, 6}, mode1_mode2.toArray());

        System.out.println("Deep nesting: " + shape);
    }
}
