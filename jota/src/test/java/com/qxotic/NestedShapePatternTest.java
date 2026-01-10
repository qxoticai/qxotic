package com.qxotic;

import com.qxotic.jota.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NestedShapePatternTest {

    @Test
    void testSimpleNestedPattern() {
        // [batch, [N, M]] with dims 2, 3, 4
        // Should create: rank=2, mode0=2, mode1=3*4=12
        Shape shape = Shape.pattern("[batch, [N, M]]", 2, 3, 4);

        assertEquals(2, shape.rank(), "Should have rank 2");
        assertEquals(3, shape.flatRank(), "Should have 3 flat dimensions");
        assertEquals(2, shape.size(0), "First mode size should be 2");
        assertEquals(12, shape.size(1), "Second mode size should be 3*4=12");
        assertEquals(24, shape.size(), "Total size should be 2*3*4=24");

        assertArrayEquals(new long[]{2, 3, 4}, shape.toArray());
    }

    @Test
    void testFlatPattern() {
        // [a, b, c] with dims 2, 3, 4
        // Should be equivalent to Shape.of(2, 3, 4)
        Shape shape = Shape.pattern("[a, b, c]", 2, 3, 4);

        assertEquals(3, shape.rank());
        assertEquals(3, shape.flatRank());
        assertTrue(shape.isFlat());
        assertEquals(2, shape.size(0));
        assertEquals(3, shape.size(1));
        assertEquals(4, shape.size(2));
    }

    @Test
    void testDeeplyNestedPattern() {
        // [a, [b, [c, d]]] with dims 2, 3, 4, 5
        // rank=2: mode0=2, mode1=3*4*5=60
        Shape shape = Shape.pattern("[a, [b, [c, d]]]", 2, 3, 4, 5);

        assertEquals(2, shape.rank());
        assertEquals(4, shape.flatRank());
        assertEquals(2, shape.size(0));
        assertEquals(60, shape.size(1), "Second mode should be 3*4*5=60");
    }

    @Test
    void testMultipleNestedGroups() {
        // [[a, b], [c, d]] with dims 2, 3, 4, 5
        // rank=2: mode0=2*3=6, mode1=4*5=20
        Shape shape = Shape.pattern("[[a, b], [c, d]]", 2, 3, 4, 5);

        assertEquals(2, shape.rank());
        assertEquals(4, shape.flatRank());
        assertEquals(6, shape.size(0), "First mode should be 2*3=6");
        assertEquals(20, shape.size(1), "Second mode should be 4*5=20");
    }

    @Test
    void testPatternWithoutNames() {
        // Can omit names: [, [, ]]
        Shape shape = Shape.pattern("[, [, ]]", 2, 3, 4);

        assertEquals(2, shape.rank());
        assertEquals(3, shape.flatRank());
        assertEquals(2, shape.size(0));
        assertEquals(12, shape.size(1));
    }

    @Test
    void testMismatchedDimensions() {
        // Pattern expects 3 dims but only 2 provided
        assertThrows(IllegalArgumentException.class, () -> {
            Shape.pattern("[a, [b, c]]", 2, 3);
        });
    }

    @Test
    void testToString() {
        Shape shape = Shape.pattern("[batch, [N, M]]", 2, 3, 4);
        System.out.println("Shape: " + shape);
        // Should print nested structure like [2, [3, 4]]
    }

    @Test
    void testDeeperNesting() {
        // [,[,,[,]]] with dims 2, 3, 4, 5, 6
        // Should create: [2, [3, 4, [5, 6]]]
        // rank=2: mode0=2, mode1 is nested [3,4,[5,6]]
        Shape shape = Shape.pattern("[,[,,[,]]]", 2, 3, 4, 5, 6);

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
        System.out.println("Mode 1 toArray: " + java.util.Arrays.toString(mode1.toArray()));

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
