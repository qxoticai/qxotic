package com.qxotic.jinfer.x.chat;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ServiceLoader;
import org.junit.jupiter.api.Test;

/** The all-model deliverable must keep provider discovery and its diagnostics in sync. */
class PortArtifactsDriftTest {

    @Test
    void everyClasspathProviderIsInTheShadingFallback() {
        for (ModelProvider provider : ServiceLoader.load(ModelProvider.class)) {
            assertTrue(
                    Models.knownProviderClasses().contains(provider.getClass().getName()),
                    provider.getClass().getName()
                            + " is absent from Models.KNOWN_PROVIDER_CLASSES");
        }
    }

    @Test
    void everyClasspathArchitectureHasAnArtifactHint() {
        var architectures = Models.supportedArchitectures();
        assertFalse(architectures.isEmpty(), "no model ports on the CLI test classpath");
        for (String architecture : architectures) {
            assertNotNull(
                    Models.artifactFor(architecture),
                    "architecture '" + architecture + "' has no artifact hint");
        }
    }
}
