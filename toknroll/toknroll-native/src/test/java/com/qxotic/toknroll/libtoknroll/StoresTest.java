package com.qxotic.toknroll.libtoknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class StoresTest {
    @AfterEach
    void unset() {
        System.clearProperty("toknroll.offline");
        System.clearProperty("jinfer.offline");
    }

    @Test
    void offlineIsAnySpellingButFalseAndEmpty() {
        for (Map.Entry<String, Boolean> e :
                Map.of(
                                "true", true, "ture", true, "1", true, "false", false, "0", false,
                                "off", false, "No", false, "", false, "  ", false)
                        .entrySet()) {
            System.setProperty("toknroll.offline", e.getKey());
            assertEquals(e.getValue(), Stores.offline(), "-Dtoknroll.offline=" + e.getKey());
        }
        System.clearProperty("toknroll.offline");
        assertEquals(false, Stores.offline(), "unset");
        System.setProperty("jinfer.offline", "yes");
        assertEquals(true, Stores.offline(), "either library's spelling");
    }
}
