package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertNotNull;

import org.junit.jupiter.api.Test;

/**
 * The remedy table cannot drift: every architecture a classpath port enumerates must resolve to
 * an artifact in {@link Models}'s diagnostics table, so a NEW port cannot land without the entry
 * that lets "unsupported architecture" name it on classpaths that lack the port. Lives in
 * jinfer-server because its test classpath carries every port (jinfer-chat's cannot: the ports
 * depend on it).
 */
final class PortArtifactsDriftTest {

    @Test
    void everyPortArchitectureHasARemedyEntry() {
        var archs = Models.supportedArchitectures();
        org.junit.jupiter.api.Assertions.assertFalse(archs.isEmpty(), "no ports on test classpath");
        for (String arch : archs) {
            assertNotNull(
                    Models.artifactFor(arch),
                    "architecture '"
                            + arch
                            + "' has no PORT_ARTIFACTS entry in Models - add it so the"
                            + " missing-port error can name the artifact");
        }
    }
}
