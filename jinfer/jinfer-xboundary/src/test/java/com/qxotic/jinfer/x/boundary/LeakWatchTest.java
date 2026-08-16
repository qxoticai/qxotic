package com.qxotic.jinfer.x.boundary;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Pins the leak detector: an owned state dropped without close() reports its creation stack once GC
 * reclaims it; a properly closed one stays silent. Sites are captured at creation, so flipping the
 * flag here cannot make other tests' drops report.
 */
class LeakWatchTest {

    final List<String> reports = Collections.synchronizedList(new ArrayList<>());
    boolean wasEnabled;
    Consumer<String> oldSink;

    @BeforeEach
    void arm() {
        wasEnabled = LeakWatch.enabled;
        oldSink = LeakWatch.sink;
        LeakWatch.enabled = true;
        LeakWatch.sink = reports::add;
    }

    @AfterEach
    void disarm() {
        LeakWatch.enabled = wasEnabled;
        LeakWatch.sink = oldSink;
    }

    @Test
    void droppedUnclosedOwnedStateReportsItsCreationSite() throws Exception {
        createAndDrop();
        long deadline = System.nanoTime() + 10_000_000_000L;
        while (reports.isEmpty() && System.nanoTime() < deadline) {
            System.gc();
            Thread.sleep(50);
        }
        assertFalse(reports.isEmpty(), "GC reclaimed the unclosed state without a report");
        assertTrue(reports.get(0).contains("owned state arena"), reports.get(0));
        assertTrue(reports.get(0).contains("createAndDrop"), "must name the creation site");
    }

    @Test
    void properlyClosedStateStaysSilent() throws Exception {
        RuntimeStateLifecycleTest.ProbeState s =
                new ModelArenaMatrixTest.ProbeModel().newState(8, 8);
        s.close();
        for (int i = 0; i < 3; i++) {
            System.gc();
            Thread.sleep(50);
        }
        assertTrue(reports.isEmpty(), () -> reports.get(0));
    }

    private static void createAndDrop() {
        new ModelArenaMatrixTest.ProbeModel().newState(8, 8);
    }
}
