package com.qxotic.toknroll.libtoknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.Test;

/**
 * The image's logger finder, registered as a service and read here by the JVM's loader: a misnamed
 * service file would be a silent fallback to java.util.logging. The console is given, not found.
 */
class ProvidersTest {
    @Test
    void theLogFinderIsTheJdksFinderHere() {
        assertEquals(LogFinder.class, System.LoggerFinder.getLoggerFinder().getClass());
        assertEquals("com.qxotic.probe", System.getLogger("com.qxotic.probe").getName());
    }

    @Test
    void theShimConsoleIsNoConsoleOnAJvm() {
        ShimConsole shim = new ShimConsole();
        assertFalse(shim.isTerminal(1), "no C binding on a JVM: no console");
        assertEquals(0, shim.columns(2));
        assertFalse(shim.enableAnsi(2));
        assertFalse(shim.enableUtf8());
    }
}
