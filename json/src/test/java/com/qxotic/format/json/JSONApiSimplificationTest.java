package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;
import org.junit.jupiter.api.Test;

class JSONApiSimplificationTest {

    @Test
    void testOptionsShortcutMatchesDefaults() {
        JSON.ParseOptions options = JSON.ParseOptions.defaults();
        assertTrue(options.decimalsAsBigDecimal());
        assertEquals(1000, options.maxDepth());
        assertFalse(options.failOnDuplicateKeys());
    }

    @Test
    void testOptionsPresets() {
        JSON.ParseOptions strict = JSON.ParseOptions.defaults().failOnDuplicateKeys(true);
        JSON.ParseOptions fast = JSON.ParseOptions.defaults().decimalsAsBigDecimal(false);

        assertTrue(strict.failOnDuplicateKeys());
        assertTrue(strict.decimalsAsBigDecimal());
        assertFalse(fast.decimalsAsBigDecimal());
    }

    @Test
    void testParseBooleanRoot() {
        assertTrue(JSON.parseBoolean("true"));
        assertFalse(JSON.parseBoolean("false"));
    }

    @Test
    void testParseBooleanRejectsOtherRoots() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parseBoolean("1"));
        assertTrue(e.getMessage().contains("Expected JSON boolean at root"));
    }

    @Test
    void testTypedParseForMaps() {
        Map<?, ?> map = JSON.parseMap("{\"a\":1}");
        assertEquals(1L, map.get("a"));
    }

    @Test
    void testTypedParseRespectsOptions() {
        boolean valid =
                JSON.isValid(
                        "{\"a\":1,\"a\":2}",
                        JSON.ParseOptions.defaults().failOnDuplicateKeys(true));
        assertFalse(valid);
    }

    @Test
    void testJsonNullHelper() {
        Object parsed = JSON.parse("null");
        assertTrue(parsed == JSON.NULL);
        assertFalse(null == JSON.NULL);
    }
}
