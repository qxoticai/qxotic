package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.util.ServiceLoader;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * The remedy table cannot drift: every architecture a classpath port enumerates must resolve to an
 * artifact in {@link Models}'s diagnostics table, so a NEW port cannot land without the entry that
 * lets "unsupported architecture" name it on classpaths that lack the port. Lives in jinfer-server
 * because its test classpath carries every port (jinfer-chat's cannot: the ports depend on it).
 */
final class PortArtifactsDriftTest {

    @Test
    void everyClasspathProviderIsInTheShadingFallback() {
        var known = Models.knownProviderClasses();
        for (var provider :
                ServiceLoader.load(ModelProvider.class).stream()
                        .map(ServiceLoader.Provider::get)
                        .toList()) {
            Assertions.assertTrue(
                    known.contains(provider.getClass().getName()),
                    provider.getClass().getName()
                            + " is not in Models.KNOWN_PROVIDER_CLASSES - add it so shading"
                            + " without ServicesResourceTransformer still recovers this port");
        }
    }

    @Test
    void everyPortArchitectureHasARemedyEntry() {
        var archs = Models.supportedArchitectures();
        assertFalse(archs.isEmpty(), "no ports on the test classpath");
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
