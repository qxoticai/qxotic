package ai.qxotic.jota.impl;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PatternValidationTest {

    @Test
    void testParseScalarPattern() {
        assertArrayEquals(new int[]{}, PatternParser.parsePattern("()", 0, "dimension"));
    }

    @Test
    void testParseFlatPattern() {
        assertArrayEquals(new int[]{0, 0, 0}, PatternParser.parsePattern("(a, b, c)", 3, "dimension"));
    }

    @Test
    void testParseNestedPattern() {
        assertArrayEquals(new int[]{0, 1, -1}, PatternParser.parsePattern("(a, (b, c))", 3, "dimension"));
    }

    @Test
    void testRejectEmptyNestedPattern() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternParser.parsePattern("(a, ())", 1, "dimension");
        });
    }

    @Test
    void testRejectSingleElementNestedPattern() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternParser.parsePattern("((a))", 1, "dimension");
        });
    }
}
