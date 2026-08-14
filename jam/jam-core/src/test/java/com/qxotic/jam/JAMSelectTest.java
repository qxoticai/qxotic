package com.qxotic.jam;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

/**
 * The {@code -Djam.<id>.disabled=true} knob, exercised through {@link JAM#select} with in-memory
 * providers and properties (no {@code System.setProperty}, no ServiceLoader).
 */
class JAMSelectTest {

    private static JAM.Provider fake(String id, int priority) {
        return new JAM.Provider() {
            public String id() {
                return id;
            }

            public int priority() {
                return priority;
            }

            public boolean isAvailable() {
                return true;
            }

            public JAM create() {
                throw new AssertionError("selection must not create backends");
            }
        };
    }

    private static Function<String, String> props(String... kv) {
        Map<String, String> map = new HashMap<>();
        for (int i = 0; i < kv.length; i += 2) map.put(kv[i], kv[i + 1]);
        return map::get;
    }

    @Test
    void aDisabledProviderIsFilteredOut() {
        List<JAM.Provider> selected =
                JAM.select(
                        List.of(fake("native", 100), fake("scalar", 1)),
                        props("jam.native.disabled", "true"));
        assertEquals(1, selected.size());
        assertEquals("scalar", selected.get(0).id());
    }

    @Test
    void onlyTrueDisables() {
        List<JAM.Provider> all = List.of(fake("vector", 10));
        assertEquals(1, JAM.select(all, props("jam.vector.disabled", "1")).size());
        assertEquals(1, JAM.select(all, props("jam.vector.disabled", "yes")).size());
        assertEquals(0, JAM.select(all, props("jam.vector.disabled", "TRUE")).size());
    }

    @Test
    void anUnknownIdIsASilentNoOpAndDisablingAllYieldsEmpty() {
        List<JAM.Provider> all = List.of(fake("scalar", 1));
        assertEquals(1, JAM.select(all, props("jam.nosuch.disabled", "true")).size());
        assertTrue(JAM.select(all, props("jam.scalar.disabled", "true")).isEmpty());
    }

    @Test
    void priorityOrderSurvivesTheFilter() {
        List<JAM.Provider> selected =
                JAM.select(
                        List.of(fake("scalar", 1), fake("native", 100), fake("vector", 10)),
                        props("jam.native.disabled", "true"));
        assertEquals(List.of("vector", "scalar"), selected.stream().map(JAM.Provider::id).toList());
    }
}
