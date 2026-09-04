package com.qxotic.jota.ir.tir;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import org.junit.jupiter.api.Test;

class ViewOperationTest {

    @Test
    void transposeOwnsItsPermutationAndComputesItsInverse() {
        int[] permutation = {2, 0, 1};
        ViewOperation.Transpose transpose = new ViewOperation.Transpose(permutation);
        permutation[0] = 0;

        assertArrayEquals(new int[] {2, 0, 1}, transpose.permutation());
        assertArrayEquals(new int[] {1, 2, 0}, transpose.inverse());

        int[] returned = transpose.permutation();
        returned[0] = 0;
        assertArrayEquals(new int[] {2, 0, 1}, transpose.permutation());

        ViewOperation.Transpose equal = new ViewOperation.Transpose(new int[] {2, 0, 1});
        assertEquals(transpose, equal);
        assertEquals(transpose.hashCode(), equal.hashCode());
        assertNotEquals(transpose, new ViewOperation.Transpose(new int[] {1, 2, 0}));
        assertEquals("Transpose[permutation=[2, 0, 1]]", transpose.toString());
    }

    @Test
    void rejectsInvalidOperations() {
        assertAll(
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewOperation.Transpose(null)),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewOperation.Transpose(new int[0])),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewOperation.Transpose(new int[] {0, 0})),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewOperation.Transpose(new int[] {0, 2})),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewOperation.Reshape(null)),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewOperation.Broadcast(null)),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewOperation.Expand(null)),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewOperation.Slice(0, 0, 1, 0)));
    }

    @Test
    void viewTransformDerivesAndValidatesItsLayout() {
        TensorInput input = new TensorInput(0, DataType.FP32, Layout.rowMajor(Shape.of(2, 3)));

        ViewTransform transposed =
                new ViewTransform(input, new ViewOperation.Transpose(new int[] {1, 0}));

        assertEquals(Layout.of(Shape.of(3, 2), Stride.of(1, 3)), transposed.layout());
        assertAll(
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        new ViewTransform(
                                                input, new ViewOperation.Reshape(Shape.of(5)))),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        new ViewTransform(
                                                input, new ViewOperation.Slice(1, 0, 4, 1))));
    }
}
