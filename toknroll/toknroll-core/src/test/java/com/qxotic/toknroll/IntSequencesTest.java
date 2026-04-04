package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class IntSequencesTest {

    @Test
    void concatAllHandlesEmptyAndNulls() {
        assertEquals(IntSequence.empty(), IntSequences.concatAll());
        assertThrows(
                NullPointerException.class, () -> IntSequences.concatAll((IntSequence[]) null));
        assertThrows(
                NullPointerException.class,
                () -> IntSequences.concatAll(IntSequence.of(1), null, IntSequence.of(2)));
    }

    @Test
    void concatAndConcatAllProduceExpectedOrder() {
        IntSequence merged =
                IntSequences.concat(IntSequence.of(1), IntSequence.of(2), IntSequence.of(3, 4));
        assertArrayEquals(new int[] {1, 2, 3, 4}, merged.toArray());

        IntSequence mergedAll =
                IntSequences.concatAll(IntSequence.of(5), IntSequence.of(), IntSequence.of(6));
        assertArrayEquals(new int[] {5, 6}, mergedAll.toArray());
    }

    @Test
    void concatRejectsNullInputs() {
        assertThrows(
                NullPointerException.class, () -> IntSequences.concat(null, IntSequence.of(1)));
        assertThrows(
                NullPointerException.class, () -> IntSequences.concat(IntSequence.of(1), null));
        assertThrows(
                NullPointerException.class,
                () ->
                        IntSequences.concat(
                                IntSequence.of(1), IntSequence.of(2), (IntSequence[]) null));
        assertThrows(
                NullPointerException.class,
                () ->
                        IntSequences.concat(
                                IntSequence.of(1), IntSequence.of(2), (IntSequence) null));
    }

    @Test
    void contentEqualsCoversEqualAndDifferentShapes() {
        assertTrue(IntSequences.contentEquals(IntSequence.of(1, 2, 3), IntSequence.of(1, 2, 3)));
        assertTrue(IntSequences.contentEquals(IntSequence.empty(), IntSequence.of()));
        assertFalse(IntSequences.contentEquals(IntSequence.of(1, 2), IntSequence.of(1, 2, 3)));
        assertFalse(IntSequences.contentEquals(IntSequence.of(1, 2, 4), IntSequence.of(1, 2, 3)));

        assertThrows(
                NullPointerException.class,
                () -> IntSequences.contentEquals(null, IntSequence.of(1)));
        assertThrows(
                NullPointerException.class,
                () -> IntSequences.contentEquals(IntSequence.of(1), null));
    }

    @Test
    void compareIsLexicographicAndOverflowSafe() {
        assertEquals(0, IntSequences.compare(IntSequence.of(1, 2), IntSequence.of(1, 2)));
        assertTrue(IntSequences.compare(IntSequence.of(1, 2), IntSequence.of(1, 3)) < 0);
        assertTrue(IntSequences.compare(IntSequence.of(2, 0), IntSequence.of(1, 99)) > 0);
        assertTrue(IntSequences.compare(IntSequence.of(1, 2), IntSequence.of(1, 2, 0)) < 0);
        assertTrue(
                IntSequences.compare(
                                IntSequence.of(Integer.MIN_VALUE),
                                IntSequence.of(Integer.MAX_VALUE))
                        < 0);

        assertThrows(
                NullPointerException.class, () -> IntSequences.compare(null, IntSequence.of(1)));
        assertThrows(
                NullPointerException.class, () -> IntSequences.compare(IntSequence.of(1), null));
    }
}
