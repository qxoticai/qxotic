package com.qxotic.jam;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The {@code -Djam.<id>.disabled=true} contract, exercised through the real {@link JAM#providers()}
 * path: a test-only provider is registered via {@code META-INF/services} so the discovery and
 * filtering behavior is what is under test, not a factored-out helper.
 */
class JAMProvidersTest {

    @Test
    void disabledProviderDisappearsFromTheList() {
        with("jam.test.disabled", "true", () -> assertFalse(ids().contains("test")));
    }

    @Test
    void onlyLiteralTrueDisables() {
        for (String value : new String[] {"true", "TRUE", "True"}) {
            with("jam.test.disabled", value, () -> assertFalse(ids().contains("test"), value));
        }
        for (String value : new String[] {"1", "yes", "false", "0", "no", ""}) {
            with("jam.test.disabled", value, () -> assertTrue(ids().contains("test"), value));
        }
    }

    @Test
    void unknownIdIsASilentNoOp() {
        with("jam.nosuch.disabled", "true", () -> assertTrue(ids().contains("test")));
    }

    @Test
    void disabledProviderIsNotProbedForAvailability() {
        TestProvider.availabilityChecks = 0;
        with(
                "jam.test.disabled",
                "true",
                () -> {
                    JAM.providers();
                    assertEquals(
                            0,
                            TestProvider.availabilityChecks,
                            "a disabled provider must be dropped before isAvailable() is probed");
                });
    }

    private static List<String> ids() {
        return JAM.providers().stream().map(JAM.Provider::id).toList();
    }

    private static void with(String key, String value, Runnable action) {
        System.setProperty(key, value);
        try {
            action.run();
        } finally {
            System.clearProperty(key);
        }
    }
}
