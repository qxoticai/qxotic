package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;
import org.junit.jupiter.api.Test;

class JsonApiSimplificationTest {

    @Test
    void testOptionsShortcutMatchesDefaults() {
        Json.ParseOptions options = Json.ParseOptions.defaults();
        assertTrue(options.decimalsAsBigDecimal());
        assertEquals(1000, options.maxDepth());
        assertFalse(options.failOnDuplicateKeys());
    }

    @Test
    void testOptionsPresets() {
        Json.ParseOptions strict = Json.ParseOptions.defaults().failOnDuplicateKeys(true);
        Json.ParseOptions fast = Json.ParseOptions.defaults().decimalsAsBigDecimal(false);

        assertTrue(strict.failOnDuplicateKeys());
        assertTrue(strict.decimalsAsBigDecimal());
        assertFalse(fast.decimalsAsBigDecimal());
    }

    @Test
    void testParseBooleanRoot() {
        assertTrue(Json.parseBoolean("true"));
        assertFalse(Json.parseBoolean("false"));
    }

    @Test
    void testParseBooleanRejectsOtherRoots() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parseBoolean("1"));
        assertTrue(e.getMessage().contains("Expected JSON boolean at root"));
    }

    @Test
    void testTypedParseForMaps() {
        Map<?, ?> map = Json.parseMap("{\"a\":1}");
        assertEquals(1L, map.get("a"));
    }

    @Test
    void testTypedParseRespectsOptions() {
        // isValid() with failOnDuplicateKeys should reject duplicate keys
        assertFalse(
                Json.isValid(
                        "{\"a\":1,\"a\":2}",
                        Json.ParseOptions.defaults().failOnDuplicateKeys(true)));
    }

    @Test
    void testJsonNullHelper() {
        Object parsed = Json.parse("null");
        assertSame(Json.NULL, parsed);
        assertNotNull(Json.NULL);
    }

    @Test
    void testParseOptionsImmutability() {
        Json.ParseOptions defaults = Json.ParseOptions.defaults();
        Json.ParseOptions shallow = defaults.maxDepth(10);
        Json.ParseOptions strict = defaults.failOnDuplicateKeys(true);

        // Original must be unchanged
        assertEquals(1000, defaults.maxDepth());
        assertFalse(defaults.failOnDuplicateKeys());

        // Derived instances must have new values
        assertEquals(10, shallow.maxDepth());
        assertFalse(shallow.failOnDuplicateKeys());
        assertTrue(strict.failOnDuplicateKeys());
        assertEquals(1000, strict.maxDepth());
    }
}
