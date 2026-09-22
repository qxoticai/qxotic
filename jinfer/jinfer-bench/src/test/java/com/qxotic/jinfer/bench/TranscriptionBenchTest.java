package com.qxotic.jinfer.bench;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class TranscriptionBenchTest {

    @Test
    void normalizationMatchesAsrConventions() {
        assertArrayEquals(
                new String[] {"don't", "stop", "me", "now"},
                TranscriptionBench.normalize("  Don't stop, me... NOW!"));
        assertArrayEquals(new String[0], TranscriptionBench.normalize("...!"));
    }

    @Test
    void editDistanceCountsSubstitutionsInsertionsDeletions() {
        String[] reference = TranscriptionBench.normalize("ask not what your country can do");
        assertEquals(0, TranscriptionBench.editDistance(reference, reference));
        assertEquals(
                1,
                TranscriptionBench.editDistance(
                        reference,
                        TranscriptionBench.normalize("ask not what your county can do")));
        assertEquals(
                2,
                TranscriptionBench.editDistance(
                        reference, TranscriptionBench.normalize("not what your country can")));
        assertEquals(reference.length, TranscriptionBench.editDistance(reference, new String[0]));
    }
}
