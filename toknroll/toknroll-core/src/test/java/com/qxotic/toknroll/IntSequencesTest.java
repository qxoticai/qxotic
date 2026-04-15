package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class IntSequencesTest {

    @Test
    void concatAllHandlesEmptyAndNulls() {
        assertEquals(IntSequence.empty(), IntSequence.concatAll());
        assertThrows(
                NullPointerException.class, () -> IntSequence.concatAll((IntSequence[]) null));
        assertThrows(
                NullPointerException.class,
                () -> IntSequence.concatAll(IntSequence.of(1), null, IntSequence.of(2)));
    }

    @Test
    void concatAllProducesExpectedOrder() {
        IntSequence mergedAll =
                IntSequence.concatAll(
                        IntSequence.of(1), IntSequence.of(2), IntSequence.of(3, 4));
        assertArrayEquals(new int[] {1, 2, 3, 4}, mergedAll.toArray());

        IntSequence mergedWithEmpty =
                IntSequence.concatAll(IntSequence.of(5), IntSequence.of(), IntSequence.of(6));
        assertArrayEquals(new int[] {5, 6}, mergedWithEmpty.toArray());
    }

    @Test
    void concatAllRejectsNullInputs() {
        assertThrows(
                NullPointerException.class,
                () -> IntSequence.concatAll(null, IntSequence.of(1)));
        assertThrows(
                NullPointerException.class,
                () -> IntSequence.concatAll(IntSequence.of(1), null));
        assertThrows(
                NullPointerException.class,
                () -> IntSequence.concatAll((IntSequence[]) null));
    }

    @Test
    void contentEqualsCoversEqualAndDifferentShapes() {
        assertTrue(IntSequence.contentEquals(IntSequence.of(1, 2, 3), IntSequence.of(1, 2, 3)));
        assertTrue(IntSequence.contentEquals(IntSequence.empty(), IntSequence.of()));
        assertFalse(IntSequence.contentEquals(IntSequence.of(1, 2), IntSequence.of(1, 2, 3)));
        assertFalse(IntSequence.contentEquals(IntSequence.of(1, 2, 4), IntSequence.of(1, 2, 3)));

        assertThrows(
                NullPointerException.class,
                () -> IntSequence.contentEquals(null, IntSequence.of(1)));
        assertThrows(
                NullPointerException.class,
                () -> IntSequence.contentEquals(IntSequence.of(1), null));
    }

    @Test
    void compareIsLexicographicAndOverflowSafe() {
        assertEquals(0, IntSequence.compare(IntSequence.of(1, 2), IntSequence.of(1, 2)));
        assertTrue(IntSequence.compare(IntSequence.of(1, 2), IntSequence.of(1, 3)) < 0);
        assertTrue(IntSequence.compare(IntSequence.of(2, 0), IntSequence.of(1, 99)) > 0);
        assertTrue(IntSequence.compare(IntSequence.of(1, 2), IntSequence.of(1, 2, 0)) < 0);
        assertTrue(
                IntSequence.compare(
                                IntSequence.of(Integer.MIN_VALUE),
                                IntSequence.of(Integer.MAX_VALUE))
                        < 0);

        assertThrows(
                NullPointerException.class, () -> IntSequence.compare(null, IntSequence.of(1)));
        assertThrows(
                NullPointerException.class, () -> IntSequence.compare(IntSequence.of(1), null));
    }
}
