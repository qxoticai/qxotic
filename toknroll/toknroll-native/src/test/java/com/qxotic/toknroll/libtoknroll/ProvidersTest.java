package com.qxotic.toknroll.libtoknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.hub.Console;
import java.util.ServiceLoader;
import org.junit.jupiter.api.Test;

/**
 * The image's two service registrations, read here by the JVM's loaders: a misnamed file would be a
 * silent fallback to the JDK's.
 */
class ProvidersTest {
    @Test
    void theShimConsoleIsRegisteredAndIsNoConsoleOnAJvm() {
        Console shim =
                ServiceLoader.load(Console.class).stream()
                        .map(ServiceLoader.Provider::get)
                        .filter(ShimConsole.class::isInstance)
                        .findFirst()
                        .orElseThrow();
        assertFalse(shim.isTerminal(1), "no C binding on a JVM: no console");
        assertEquals(0, shim.columns(2));
        assertFalse(shim.enableAnsi(2));
        assertFalse(shim.enableUtf8());
    }

    @Test
    void theLogFinderIsTheJdksFinderHere() {
        assertEquals(LogFinder.class, System.LoggerFinder.getLoggerFinder().getClass());
        assertTrue(System.getLogger("com.qxotic.probe").getName().equals("com.qxotic.probe"));
    }
}
