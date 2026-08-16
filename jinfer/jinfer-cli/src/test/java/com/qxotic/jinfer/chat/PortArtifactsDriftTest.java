package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ServiceLoader;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** The all-model deliverable must keep provider discovery and its diagnostics in sync. */
class PortArtifactsDriftTest {

    @Test
    void serviceDescriptorsAndShadingFallbackAgree() {
        Set<String> services =
                ServiceLoader.load(ModelProvider.class).stream()
                        .map(provider -> provider.type().getName())
                        .collect(Collectors.toSet());
        assertEquals(Set.copyOf(Models.knownProviderClasses()), services);
    }

    @Test
    void everyClasspathArchitectureHasAnArtifactHint() {
        var architectures = Models.supportedArchitectures();
        assertFalse(architectures.isEmpty(), "no model ports on the CLI test classpath");
        for (String architecture : architectures) {
            String artifact = Models.artifactFor(architecture);
            assertNotNull(artifact, "architecture '" + architecture + "' has no artifact hint");
            assertTrue(
                    artifact.startsWith("com.qxotic:jinfer-"),
                    "architecture '" + architecture + "' has invalid artifact hint " + artifact);
        }
    }
}
