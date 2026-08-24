package com.qxotic.jota.impl;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class PatternValidationTest {

    @Test
    void testRejectNullAndEmptyPatterns() {
        assertThrows(
                IllegalArgumentException.class,
                () -> PatternParser.parsePattern(null, 0, "dimension"));
        assertThrows(
                IllegalArgumentException.class,
                () -> PatternParser.parsePattern("", 0, "dimension"));
        assertThrows(
                IllegalArgumentException.class,
                () -> PatternParser.parsePattern(" ", 0, "dimension"));
    }

    @Test
    void testParseScalarPattern() {
        assertArrayEquals(new int[] {}, PatternParser.parsePattern("()", 0, "dimension"));
    }

    @Test
    void testParseFlatPattern() {
        assertArrayEquals(
                new int[] {0, 0, 0}, PatternParser.parsePattern("(a, b, c)", 3, "dimension"));
    }

    @Test
    void testParseNestedPattern() {
        assertArrayEquals(
                new int[] {0, 1, -1}, PatternParser.parsePattern("(a, (b, c))", 3, "dimension"));
    }

    @Test
    void testRejectEmptyNestedPattern() {
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    PatternParser.parsePattern("(a, ())", 1, "dimension");
                });
    }

    @Test
    void testRejectSingleElementNestedPattern() {
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    PatternParser.parsePattern("((a))", 1, "dimension");
                });
    }

    @Test
    void testRejectMalformedSeparatorsAndNesting() {
        assertAll(
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> PatternParser.parsePattern("(a b)", 2, "dimension")),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> PatternParser.parsePattern("(a,)", 1, "dimension")),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> PatternParser.parsePattern("(,a)", 1, "dimension")),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> PatternParser.parsePattern("((a, b)", 2, "dimension")),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> PatternParser.parsePattern("(a, b))", 2, "dimension")),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> PatternParser.parsePattern("(a)", 2, "dimension")));
    }
}
