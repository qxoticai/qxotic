package com.qxotic.jinfer.testkit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class SystemPropertyTest {

    private static final String KEY = "jinfer.test.systemPropertyTest";

    @AfterEach
    void clear() {
        System.clearProperty(KEY);
    }

    @Test
    void anUnsetPropertyIsClearedAgain() {
        try (var property = SystemProperty.override(KEY, "inside")) {
            assertEquals("inside", System.getProperty(KEY));
        }
        assertNull(System.getProperty(KEY));
    }

    @Test
    void aSetPropertyGetsItsValueBack() {
        System.setProperty(KEY, "outside");
        try (var property = SystemProperty.override(KEY, "inside")) {
            assertEquals("inside", System.getProperty(KEY));
        }
        assertEquals("outside", System.getProperty(KEY));
    }

    @Test
    void aNullValueClearsThePropertyForTheBlock() {
        System.setProperty(KEY, "outside");
        try (var property = SystemProperty.override(KEY, null)) {
            assertNull(System.getProperty(KEY));
        }
        assertEquals("outside", System.getProperty(KEY));
    }

    @Test
    void nestedOverridesUnwindInOrder() {
        try (var outer = SystemProperty.override(KEY, "outer")) {
            try (var inner = SystemProperty.override(KEY, "inner")) {
                assertEquals("inner", System.getProperty(KEY));
            }
            assertEquals("outer", System.getProperty(KEY));
        }
        assertNull(System.getProperty(KEY));
    }

    @Test
    void theBlockRestoresEvenWhenItThrows() {
        System.setProperty(KEY, "outside");
        assertThrows(
                IllegalStateException.class,
                () -> {
                    try (var property = SystemProperty.override(KEY, "inside")) {
                        throw new IllegalStateException("boom");
                    }
                });
        assertEquals("outside", System.getProperty(KEY));
    }
}
