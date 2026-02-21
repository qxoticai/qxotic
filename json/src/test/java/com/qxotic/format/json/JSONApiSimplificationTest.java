package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;
import org.junit.jupiter.api.Test;

class JSONApiSimplificationTest {

    @Test
    void testOptionsShortcutMatchesDefaults() {
        JSON.ParseOptions options = JSON.options();
        assertTrue(options.decimalsAsBigDecimal());
        assertEquals(1000, options.maxDepth());
        assertFalse(options.failOnDuplicateKeys());
    }

    @Test
    void testOptionsPresets() {
        JSON.ParseOptions strict = JSON.ParseOptions.strict();
        JSON.ParseOptions fast = JSON.ParseOptions.fast();

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
    void testTypedParseForObjects() {
        Map<?, ?> map = JSON.parseObject("{\"a\":1}");
        assertEquals(1L, map.get("a"));
    }

    @Test
    void testTypedParseRespectsOptions() {
        boolean valid = JSON.isValid("{\"a\":1,\"a\":2}", JSON.options().failOnDuplicateKeys(true));
        assertFalse(valid);
    }

    @Test
    void testJsonNullHelper() {
        Object parsed = JSON.parse("null");
        assertTrue(JSON.isNull(parsed));
        assertFalse(JSON.isNull(null));
    }
}
