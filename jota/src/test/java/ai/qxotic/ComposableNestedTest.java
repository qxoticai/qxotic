package ai.qxotic;

import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the unified Shape.nested() API that handles:
 * 1. Pattern-based nesting: nested("(batch, (N, M))", 2, 3, 4)
 * 2. Template-based nesting: nested(template, 10, 20, 30)
 * 3. Composition: nested(1, Shape.of(2, 3), 4)
 */
class ComposableNestedTest {

    @Test
    void testSimpleComposition() {
        // Shape.nested(2, Shape.of(4, 5), 6) → (2, (4, 5), 6)
        Shape shape = Shape.of(2, Shape.of(4L, 5L), 6);

        assertEquals(3, shape.rank());
        assertEquals(4, shape.flatRank());
        assertArrayEquals(new long[]{2, 4, 5, 6}, shape.toArray());
        assertEquals(2, shape.size(0));
        assertEquals(20, shape.size(1));
        assertEquals(6, shape.size(2));
    }

    @Test
    void testSingleElementNormalization() {
        // Single-element shapes get unwrapped: nested(1, Shape.of(2), Shape.of(3)) → (1, 2, 3)
        Shape shape = Shape.of(1, Shape.of(2L), Shape.of(3L));

        assertTrue(shape.isFlat());
        assertEquals(3, shape.rank());
        assertArrayEquals(new long[]{1, 2, 3}, shape.toArray());
    }

    @Test
    void testPatternBased() {
        // Pattern-based nesting still works
        Shape shape = Shape.pattern("(batch, (N, M))", 2, 3, 4);

        assertEquals(2, shape.rank());
        assertEquals(3, shape.flatRank());
        assertEquals(2, shape.size(0));
        assertEquals(12, shape.size(1));
    }

    @Test
    void testTemplateBased() {
        // Template-based nesting still works
        Shape template = Shape.pattern("(_,(_,_,(_,_)))", 2, 3, 4, 5, 6);
        Shape newShape = Shape.template(template, 10, 20, 30, 40, 50);

        assertEquals(2, newShape.rank());
        assertEquals(5, newShape.flatRank());
        assertEquals(10, newShape.size(0));
    }

    @Test
    void testDeepComposition() {
        // nested(1, Shape.nested(2, Shape.of(3, 4))) → (1, (2, (3, 4)))
        Shape shape = Shape.of(1, Shape.of(2, Shape.of(3L, 4L)));

        assertEquals(2, shape.rank());
        assertEquals(4, shape.flatRank());
        assertArrayEquals(new long[]{1, 2, 3, 4}, shape.toArray());
    }

    @Test
    void testStrideComposition() {
        // Stride.of(100, Stride.of(10, 1)) → (100, (10, 1))
        Stride stride = Stride.of(100, Stride.of(10L, 1L));

        assertEquals(2, stride.rank());
        assertEquals(3, stride.flatRank());
        assertArrayEquals(new long[]{100, 10, 1}, stride.toArray());
    }

    @Test
    void testUnwrapSingle() {
        // nested(Shape.of(2, 3)) → (2, 3)
        Shape inner = Shape.of(2L, 3L);
        Shape shape = Shape.of(inner);

        assertSame(inner, shape);
    }
}
