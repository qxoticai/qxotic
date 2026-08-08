package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/** The bar is rendered output - pinned as strings, like any other behavior. */
class FetchBarTest {

    @Test
    void subCellEdgeGlides() {
        // 4 cells wide: 1/8 of a cell shows the thinnest block, and each eighth advances it
        assertEquals("    ", Fetch.Progress.bar(0, 800, 4, true));
        assertEquals("▏   ", Fetch.Progress.bar(25, 800, 4, true));
        assertEquals("▌   ", Fetch.Progress.bar(100, 800, 4, true));
        assertEquals("▉   ", Fetch.Progress.bar(175, 800, 4, true));
        assertEquals("█   ", Fetch.Progress.bar(200, 800, 4, true));
        assertEquals("██▌ ", Fetch.Progress.bar(500, 800, 4, true));
        assertEquals("████", Fetch.Progress.bar(800, 800, 4, true));
        assertEquals("████", Fetch.Progress.bar(900, 800, 4, true)); // over-report never overflows
    }

    @Test
    void asciiConsolesKeepWholeCells() {
        assertEquals("----", Fetch.Progress.bar(25, 800, 4, false));
        assertEquals("#---", Fetch.Progress.bar(200, 800, 4, false));
        assertEquals(
                "##--", Fetch.Progress.bar(500, 800, 4, false)); // 5/8 truncates, never rounds up
        assertEquals("####", Fetch.Progress.bar(800, 800, 4, false));
    }
}
