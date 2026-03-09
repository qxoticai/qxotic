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
        boolean valid =
                Json.isValid(
                        "{\"a\":1,\"a\":2}",
                        Json.ParseOptions.defaults().failOnDuplicateKeys(true));
        assertFalse(valid);
    }

    @Test
    void testJsonNullHelper() {
        Object parsed = Json.parse("null");
        assertSame(Json.NULL, parsed);
        assertNotNull(Json.NULL);
    }
}
