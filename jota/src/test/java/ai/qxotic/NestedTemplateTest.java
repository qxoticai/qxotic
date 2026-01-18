package ai.qxotic;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;
import org.junit.jupiter.api.Test;

class NestedTemplateTest {

    @Test
    void testShapeNestedFromShape() {
        // Create a template shape with nested structure
        Shape template = Shape.pattern("(_,(_,_,(_,_)))", 2, 3, 4, 5, 6);

        // Create new shape with same nesting but different dimensions
        Shape shape = Shape.template(template, 10, 20, 30, 40, 50);

        assertEquals(2, shape.rank());
        assertEquals(5, shape.flatRank());
        assertEquals(10, shape.size(0));
        assertEquals(20 * 30 * 40 * 50, shape.size(1));

        // Verify the nesting structure is preserved
        Shape mode1 = shape.modeAt(1);
        assertEquals(3, mode1.rank());
        assertEquals(4, mode1.flatRank());
        assertArrayEquals(new long[] {20, 30, 40, 50}, mode1.toArray());

        System.out.println("Template: " + template);
        System.out.println("New shape: " + shape);
    }

    @Test
    void testShapeNestedFromStride() {
        // Create a template stride with nested structure
        Stride template = Stride.template(Shape.pattern("(batch, (N, M))", 2, 3, 4), 100, 10, 1);

        // Create new shape using stride as template
        Shape shape = Shape.template(template, 5, 6, 7);

        assertEquals(2, shape.rank());
        assertEquals(3, shape.flatRank());
        assertEquals(5, shape.size(0));
        assertEquals(6 * 7, shape.size(1));
        assertArrayEquals(new long[] {5, 6, 7}, shape.toArray());
    }

    @Test
    void testStrideNestedFromShape() {
        // Create a template shape
        Shape template = Shape.pattern("(batch, (N, M))", 2, 3, 4);

        // Create stride with same nesting structure
        Stride stride = Stride.template(template, 1000, 100, 1);

        assertEquals(2, stride.rank());
        assertEquals(3, stride.flatRank());
        assertArrayEquals(new long[] {1000, 100, 1}, stride.toArray());

        // Verify nesting is preserved
        Stride mode1 = stride.modeAt(1);
        assertEquals(2, mode1.flatRank());
        assertArrayEquals(new long[] {100, 1}, mode1.toArray());
    }

    @Test
    void testStrideNestedFromStride() {
        // Create a template stride
        Stride template =
                Stride.template(
                        Shape.pattern("(_,(_,_,(_,_)))", 2, 3, 4, 5, 6), 10000, 1000, 100, 10, 1);

        // Create new stride with same nesting
        Stride stride = Stride.template(template, 50000, 5000, 500, 50, 5);

        assertEquals(2, stride.rank());
        assertEquals(5, stride.flatRank());
        assertArrayEquals(new long[] {50000, 5000, 500, 50, 5}, stride.toArray());

        // Verify deep nesting is preserved
        Stride mode1 = stride.modeAt(1);
        assertEquals(3, mode1.rank());
        assertEquals(4, mode1.flatRank());
        assertArrayEquals(new long[] {5000, 500, 50, 5}, mode1.toArray());

        Stride mode1_mode2 = mode1.modeAt(2);
        assertEquals(2, mode1_mode2.flatRank());
        assertArrayEquals(new long[] {50, 5}, mode1_mode2.toArray());
    }

    @Test
    void testFlatTemplate() {
        // Flat template should create flat shape/stride
        Shape flatTemplate = Shape.of(2, 3, 4);

        Shape shape = Shape.template(flatTemplate, 5, 6, 7);
        assertTrue(shape.isFlat());
        assertEquals(3, shape.rank());
        assertArrayEquals(new long[] {5, 6, 7}, shape.toArray());

        Stride stride = Stride.template(flatTemplate, 100, 10, 1);
        assertTrue(stride.isFlat());
        assertEquals(3, stride.rank());
        assertArrayEquals(new long[] {100, 10, 1}, stride.toArray());
    }

    @Test
    void testMismatchedDimensions() {
        Shape template = Shape.pattern("(_,(_,_))", 2, 3, 4);

        // Too few dimensions
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Shape.template(template, 10, 20);
                });

        // Too many dimensions
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Shape.template(template, 10, 20, 30, 40);
                });

        // Scalar template with non-zero dimensions
        Shape scalarTemplate = Shape.scalar();
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Shape.template(scalarTemplate, 10);
                });
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Stride.template(scalarTemplate, 10);
                });

        // Non-scalar template with zero dimensions
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Shape.template(template);
                });
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Stride.template(template);
                });
    }

    @Test
    void testStrideRowMajorPreservesNesting() {
        // Row-major strides should preserve shape nesting
        Shape shape = Shape.pattern("(batch, (N, M))", 10, 20, 30);
        Stride stride = Stride.rowMajor(shape);

        assertEquals(2, stride.rank());
        assertEquals(3, stride.flatRank());

        // Row major for (10, 20, 30): strides are (600, 30, 1)
        assertArrayEquals(new long[] {600, 30, 1}, stride.toArray());

        // Verify nesting structure
        Stride mode1 = stride.modeAt(1);
        assertEquals(2, mode1.flatRank());
        assertArrayEquals(new long[] {30, 1}, mode1.toArray());
    }

    @Test
    void testStrideColumnMajorPreservesNesting() {
        // Column-major strides should preserve shape nesting
        Shape shape = Shape.pattern("(batch, (N, M))", 10, 20, 30);
        Stride stride = Stride.columnMajor(shape);

        assertEquals(2, stride.rank());
        assertEquals(3, stride.flatRank());

        // Column major for (10, 20, 30): strides are (1, 10, 200)
        assertArrayEquals(new long[] {1, 10, 200}, stride.toArray());

        // Verify nesting structure
        Stride mode1 = stride.modeAt(1);
        assertEquals(2, mode1.flatRank());
        assertArrayEquals(new long[] {10, 200}, mode1.toArray());
    }
}
