package com.qxotic.jota.memory.impl;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.Shape;
import org.junit.jupiter.api.Test;

class ViewKindTest {

    @Test
    void transposeOwnsItsPermutationAndComputesItsInverse() {
        int[] permutation = {2, 0, 1};
        ViewKind.Transpose transpose = new ViewKind.Transpose(permutation);
        permutation[0] = 0;

        assertArrayEquals(new int[] {2, 0, 1}, transpose.permutation());
        assertArrayEquals(new int[] {1, 2, 0}, transpose.inverse());

        int[] returned = transpose.permutation();
        returned[0] = 0;
        assertArrayEquals(new int[] {2, 0, 1}, transpose.permutation());

        ViewKind.Transpose equal = new ViewKind.Transpose(new int[] {2, 0, 1});
        assertEquals(transpose, equal);
        assertEquals(transpose.hashCode(), equal.hashCode());
        assertNotEquals(transpose, new ViewKind.Transpose(new int[] {1, 2, 0}));
        assertEquals("Transpose[permutation=[2, 0, 1]]", transpose.toString());
    }

    @Test
    void rejectsInvalidViewKinds() {
        assertAll(
                () ->
                        assertThrows(
                                IllegalArgumentException.class, () -> new ViewKind.Transpose(null)),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewKind.Transpose(new int[0])),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewKind.Transpose(new int[] {0, 0})),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewKind.Transpose(new int[] {0, 2})),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewKind.Reshape(null, Shape.of(1))),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewKind.Reshape(Shape.of(1), Shape.of(2))),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewKind.Broadcast(null, Shape.of(1))),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> new ViewKind.Expand(Shape.of(1), null)),
                () ->
                        assertThrows(
                                IllegalArgumentException.class, () -> new ViewKind.Slice(0, 0, 0)));
    }
}
