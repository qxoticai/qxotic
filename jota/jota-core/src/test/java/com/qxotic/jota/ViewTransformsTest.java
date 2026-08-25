package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class ViewTransformsTest {

    @Test
    void reshapePreservesIndependentAffineChunks() {
        Layout input = Layout.of(Shape.of(2, 4), Stride.of(8, 1));

        Layout result = ViewTransforms.reshape(input, Shape.of(2, 2, 2)).orElseThrow().layout();

        assertEquals(Layout.of(Shape.of(2, 2, 2), Stride.of(8, 2, 1)), result);
        assertTrue(ViewTransforms.reshape(input, Shape.of(8)).isEmpty());
    }

    @Test
    void reshapeSupportsScaledAndNegativeStrides() {
        Layout scaled = Layout.of(Shape.of(6), Stride.of(2));
        Layout reversed = Layout.of(Shape.of(6), Stride.of(-1));

        assertAll(
                () ->
                        assertEquals(
                                Stride.of(6, 2),
                                ViewTransforms.reshape(scaled, Shape.of(2, 3))
                                        .orElseThrow()
                                        .layout()
                                        .stride()),
                () ->
                        assertEquals(
                                Stride.of(-3, -1),
                                ViewTransforms.reshape(reversed, Shape.of(2, 3))
                                        .orElseThrow()
                                        .layout()
                                        .stride()));
    }

    @Test
    void reshapeToTheSameShapePreservesTheLayout() {
        Layout input = Layout.of(Shape.of(2, 1, 3), Stride.of(7, 99, -2));

        assertEquals(input, ViewTransforms.reshape(input, input.shape()).orElseThrow().layout());
    }

    @Test
    void reshapeSupportsEmptyLayouts() {
        Layout input = Layout.of(Shape.of(0, 3), Stride.of(7, -2));
        Shape target = Shape.of(2, 0, 4);

        assertEquals(
                Layout.rowMajor(target),
                ViewTransforms.reshape(input, target).orElseThrow().layout());
    }

    @Test
    void reshapeRejectsNonAffineElementOrder() {
        Layout transposed = Layout.of(Shape.of(3, 2), Stride.of(1, 3));

        assertTrue(ViewTransforms.reshape(transposed, Shape.of(6)).isEmpty());
        assertThrows(
                IllegalArgumentException.class,
                () -> ViewTransforms.reshape(transposed, Shape.of(5)));
    }

    @Test
    void reshapeMatchesEverySmallAffineMapping() {
        List<Shape> shapes = smallShapes();
        for (Shape sourceShape : shapes) {
            for (long[] sourceStrides : smallStrides(sourceShape.flatRank())) {
                Layout source = Layout.of(sourceShape, Stride.flat(sourceStrides));
                for (Shape targetShape : shapes) {
                    if (sourceShape.size() != targetShape.size()) {
                        continue;
                    }

                    Optional<ViewTransforms.Result> actual =
                            ViewTransforms.reshape(source, targetShape);
                    boolean expected = hasAffineMapping(source, targetShape);
                    assertEquals(
                            expected,
                            actual.isPresent(),
                            () -> "reshape " + source + " to " + targetShape);
                    actual.ifPresent(
                            result ->
                                    assertSameMapping(source, result.layout(), targetShape.size()));
                }
            }
        }
    }

    @Test
    void unsqueezeInsertsAZeroStrideSingletonMode() {
        Layout input = Layout.of(Shape.of(2, Shape.of(3L, 4L)), Stride.of(12, Stride.of(4L, 1L)));

        Layout result = ViewTransforms.unsqueeze(input, -1).layout();

        assertEquals(Shape.of(2, Shape.of(3L, 4L), 1), result.shape());
        assertEquals(Stride.of(12, Stride.of(4L, 1L), 0), result.stride());
    }

    @Test
    void expandZerosOnlyExpandedSingletonDimensions() {
        Layout input = Layout.of(Shape.of(2, 1, 3), Stride.of(8, 5, 1));

        Layout result = ViewTransforms.expand(input, Shape.of(2, 4, 3)).layout();

        assertEquals(Layout.of(Shape.of(2, 4, 3), Stride.of(8, 0, 1)), result);
        assertEquals(input, ViewTransforms.expand(input, input.shape()).layout());
        assertAll(
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> ViewTransforms.expand(input, Shape.of(2, 4, 4))),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> ViewTransforms.expand(input, Shape.of(2, Shape.of(4L, 3L)))));
    }

    @Test
    void broadcastAlignsFlattenedDimensionsAndPreservesNesting() {
        Layout input = Layout.rowMajor(Shape.of(2, Shape.of(1L, 3L)));
        Shape target = Shape.of(4, Shape.of(2L, Shape.of(5L, 3L)));

        Layout result = ViewTransforms.broadcast(input, target).layout();

        assertEquals(target, result.shape());
        assertEquals(Stride.of(0, Stride.of(3, Stride.of(0, 1))), result.stride());
    }

    @Test
    void broadcastRejectsIncompatibleShapes() {
        Layout input = Layout.rowMajor(Shape.of(2, 3));

        assertAll(
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> ViewTransforms.broadcast(input, Shape.of(3))),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> ViewTransforms.broadcast(input, Shape.of(4, 3))));
    }

    @Test
    void permuteMovesWholeNestedModes() {
        Layout input = Layout.rowMajor(Shape.of(2, Shape.of(3L, 4L), 5));

        Layout result = ViewTransforms.permute(input, 1, 2, 0).layout();

        assertEquals(Shape.of(Shape.of(3L, 4L), 5, 2), result.shape());
        assertEquals(Stride.of(Stride.of(20L, 5L), 1, 60), result.stride());
    }

    @Test
    void sliceReturnsAnElementOffsetForPositiveAndNegativeSteps() {
        Layout input = Layout.rowMajor(Shape.of(3, 4));

        ViewTransforms.Result rows = ViewTransforms.slice(input, 0, 1, 3, 1);
        ViewTransforms.Result reversedColumns = ViewTransforms.slice(input, 1, 3, -1, -2);

        assertAll(
                () -> assertEquals(Layout.of(Shape.of(2, 4), Stride.of(4, 1)), rows.layout()),
                () -> assertEquals(4, rows.elementOffsetDelta()),
                () ->
                        assertEquals(
                                Layout.of(Shape.of(3, 2), Stride.of(4, -2)),
                                reversedColumns.layout()),
                () -> assertEquals(3, reversedColumns.elementOffsetDelta()));
    }

    @Test
    void sliceMatchesEverySmallValidRange() {
        for (long dimension = 0; dimension <= 5; dimension++) {
            for (long stride = -3; stride <= 3; stride++) {
                Layout input = Layout.of(Shape.of(dimension), Stride.of(stride));
                for (long step = 1; step <= 4; step++) {
                    for (long from = 0; from <= dimension; from++) {
                        for (long to = from; to <= dimension; to++) {
                            assertSliceMapping(input, from, to, step);
                        }
                    }
                }
                for (long step = -1; step >= -4; step--) {
                    for (long from = 0; from < dimension; from++) {
                        for (long to = -1; to <= from; to++) {
                            assertSliceMapping(input, from, to, step);
                        }
                    }
                }
            }
        }
    }

    @Test
    void sliceHandlesTheLargestNegativeStep() {
        Layout input = Layout.rowMajor(Shape.of(3));

        ViewTransforms.Result result = ViewTransforms.slice(input, 0, 2, -1, Long.MIN_VALUE);

        assertEquals(Shape.of(1), result.layout().shape());
        assertEquals(Stride.of(Long.MIN_VALUE), result.layout().stride());
        assertEquals(2, result.elementOffsetDelta());
    }

    @Test
    void sliceLinearizesOnlyAffineNestedModes() {
        Layout affine = Layout.of(Shape.of(Shape.of(2L, 3L), 4), Stride.of(Stride.of(12L, 4L), 1));
        Layout nonAffine =
                Layout.of(Shape.of(Shape.of(2L, 3L), 4), Stride.of(Stride.of(4L, 8L), 1));

        ViewTransforms.Result result = ViewTransforms.slice(affine, 0, 1, 5, 2);

        assertEquals(Layout.of(Shape.of(2, 4), Stride.of(8, 1)), result.layout());
        assertEquals(4, result.elementOffsetDelta());
        assertThrows(
                IllegalArgumentException.class, () -> ViewTransforms.slice(nonAffine, 0, 0, 6, 1));
    }

    @Test
    void sliceRejectsArithmeticOverflow() {
        Layout layout = Layout.of(Shape.of(3), Stride.of(Long.MAX_VALUE));

        assertAll(
                () ->
                        assertThrows(
                                ArithmeticException.class,
                                () -> ViewTransforms.slice(layout, 0, 0, 3, 2)),
                () ->
                        assertThrows(
                                ArithmeticException.class,
                                () -> ViewTransforms.slice(layout, 0, 2, 3, 1)));
    }

    @Test
    void sliceRejectsInvalidRanges() {
        Layout layout = Layout.rowMajor(Shape.of(3));

        assertAll(
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> ViewTransforms.slice(layout, 0, 0, 3, 0)),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> ViewTransforms.slice(layout, 0, -1, 3, 1)),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> ViewTransforms.slice(layout, 0, 2, 1, 1)),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> ViewTransforms.slice(layout, 0, 2, -2, -1)));
    }

    private static List<Shape> smallShapes() {
        List<Shape> shapes = new ArrayList<>();
        for (int first = 1; first <= 4; first++) {
            shapes.add(Shape.of(first));
            for (int second = 1; second <= 4; second++) {
                shapes.add(Shape.of(first, second));
                for (int third = 1; third <= 3; third++) {
                    shapes.add(Shape.of(first, second, third));
                }
            }
        }
        return shapes;
    }

    private static List<long[]> smallStrides(int rank) {
        List<long[]> strides = new ArrayList<>();
        addStrides(strides, new long[rank], 0);
        return strides;
    }

    private static void addStrides(List<long[]> result, long[] strides, int axis) {
        if (axis == strides.length) {
            result.add(strides.clone());
            return;
        }
        for (long stride = -2; stride <= 2; stride++) {
            strides[axis] = stride;
            addStrides(result, strides, axis + 1);
        }
    }

    private static boolean hasAffineMapping(Layout source, Shape targetShape) {
        long size = targetShape.size();
        if (size <= 1) {
            return true;
        }

        long[] targetStrides = new long[targetShape.flatRank()];
        for (int axis = 0; axis < targetShape.flatRank(); axis++) {
            if (targetShape.flatAt(axis) > 1) {
                long linearIndex = 1;
                for (int inner = axis + 1; inner < targetShape.flatRank(); inner++) {
                    linearIndex = Math.multiplyExact(linearIndex, targetShape.flatAt(inner));
                }
                targetStrides[axis] = elementOffset(source, linearIndex);
            }
        }

        Layout candidate = Layout.of(targetShape, Stride.template(targetShape, targetStrides));
        for (long linearIndex = 0; linearIndex < size; linearIndex++) {
            if (elementOffset(source, linearIndex) != elementOffset(candidate, linearIndex)) {
                return false;
            }
        }
        return true;
    }

    private static void assertSameMapping(Layout source, Layout target, long size) {
        for (long linearIndex = 0; linearIndex < size; linearIndex++) {
            long index = linearIndex;
            assertEquals(
                    elementOffset(source, index),
                    elementOffset(target, index),
                    () -> "linear index " + index + ": " + source + " vs " + target);
        }
    }

    private static void assertSliceMapping(Layout input, long from, long to, long step) {
        ViewTransforms.Result result = ViewTransforms.slice(input, 0, from, to, step);
        long length = 0;
        for (long index = from; step > 0 ? index < to : index > to; index += step) {
            long resultOffset =
                    Math.addExact(
                            result.elementOffsetDelta(),
                            Indexing.linearToOffset(result.layout(), DataType.BOOL, length));
            assertEquals(
                    Math.multiplyExact(index, input.stride().flatAt(0)),
                    resultOffset,
                    () -> "slice " + input + " [" + from + ", " + to + ") step " + step);
            length++;
        }
        assertEquals(length, result.layout().shape().size());
    }

    private static long elementOffset(Layout layout, long linearIndex) {
        return Indexing.linearToOffset(layout, DataType.BOOL, linearIndex);
    }
}
