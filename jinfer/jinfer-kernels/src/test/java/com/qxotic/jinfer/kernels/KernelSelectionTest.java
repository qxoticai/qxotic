package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Segments;
import java.util.List;
import org.junit.jupiter.api.Test;
import com.sun.management.HotSpotDiagnosticMXBean;
import java.lang.management.ManagementFactory;

/**
 * Pins the kernel selection this JVM runs under to the properties it was started with.
 *
 * <p>Every kernel picks its implementation from a static constant read once at class initialization
 * ({@code -Djinfer.convTile}, {@code -Djinfer.vectorBitSize}, {@code -Djinfer.gdn.vector}, {@code
 * -Djinfer.mamba2.vector*}, the JAM rungs on the classpath), so the only way to run the suite
 * against a non-default implementation is a JVM started with that property. {@code
 * jinfer-kernels/pom.xml} does exactly that: one surefire execution per selection, each re-running
 * the whole suite. This test is what makes a misspelled or silently ignored property fail loudly
 * instead of re-testing the default.
 *
 * <p>Run one selection by hand with {@code mvn test -pl jinfer-kernels -Dkernels.execution=<id>}.
 */
class KernelSelectionTest {

    @Test
    void convTileFollowsTheProperty() {
        int expected =
                switch (System.getProperty("jinfer.convTile", "auto")) {
                    case "4x2" -> 1;
                    case "4x4" -> 2;
                    default -> 0;
                };
        assertEquals(expected, Convolutions.tileCode(), "jinfer.convTile");
    }

    @Test
    void vectorWidthFollowsTheProperty() {
        Integer requested = Integer.getInteger("jinfer.vectorBitSize");
        if (requested != null) {
            assertEquals(requested.intValue(), Segments.vectorBits(), "jinfer.vectorBitSize");
        } else {
            assertTrue(Segments.vectorBits() > 0, "the default selects the Vector API");
        }
        assertEquals(Segments.vectorBits() != 0, Segments.USE_VECTOR_API);
    }

    @Test
    void vectorJitFollowsThePropertyOrTheDetection() {
        String knob = System.getProperty("jinfer.vectorJit", "auto");
        boolean trusted =
                switch (knob) {
                    case "fast" -> true;
                    case "slow" -> false;
                    default -> Segments.vectorJitDetected();
                };
        assertEquals(
                Segments.USE_VECTOR_API && trusted, Segments.FAST_VECTOR_JIT, "jinfer.vectorJit");
    }

    @Test
    void detectionReadsTheCompilerOption() throws Exception {
        // the rule from the port: java.vm.version says "jvmci" on GraalVM even under
        // -XX:-UseJVMCICompiler, so the VM option is the only honest source
        boolean jvmci =
                Boolean.parseBoolean(
                        ManagementFactory.getPlatformMXBean(
                                        HotSpotDiagnosticMXBean.class)
                                .getVMOption("UseJVMCICompiler")
                                .getValue());
        assertEquals(jvmci, Segments.vectorJitDetected());
    }

    @Test
    void recurrenceSwitchesFollowTheProperties() {
        assertEquals(
                Boolean.parseBoolean(System.getProperty("jinfer.gdn.vector", "true")),
                VectorGatedDeltaNet.enabled(),
                "jinfer.gdn.vector");
        assertEquals(
                Boolean.parseBoolean(System.getProperty("jinfer.mamba2.vector", "true")),
                VectorMamba2.scanEnabled(),
                "jinfer.mamba2.vector");
        assertEquals(
                Boolean.parseBoolean(System.getProperty("jinfer.mamba2.vectorNorm", "true")),
                VectorMamba2.normEnabled(),
                "jinfer.mamba2.vectorNorm");
    }

    @Test
    void jamRungsMatchTheClasspath() {
        // the default executions exclude every jam-* jar, so the Java floor is what runs;
        // jam-parity puts jam-vector back and asserts the rung itself
        List<String> rungs = MatMul.jamRungs();
        String expected = System.getProperty("jinfer.kernels.expectJam", "");
        assertEquals(
                expected.isEmpty() ? List.of() : List.of(expected.split(",")),
                rungs,
                "jinfer.kernels.expectJam");
    }
}
