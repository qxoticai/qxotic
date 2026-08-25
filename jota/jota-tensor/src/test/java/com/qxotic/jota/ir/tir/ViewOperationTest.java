package com.qxotic.jota.ir.tir;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.Shape;
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
                                () -> new ViewOperation.Reshape(null, Shape.of(1))),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewOperation.Reshape(Shape.of(1), Shape.of(2))),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewOperation.Broadcast(null, Shape.of(1))),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewOperation.Expand(Shape.of(1), null)),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewOperation.Slice(0, 0, 0)));
    }
}
