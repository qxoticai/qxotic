package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class JsonErrorHandlingTest {

    @Test
    void testErrorHasLineNumber() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{\n  \"key\":,\n}"));
        assertEquals(2, e.getLine());
    }

    @Test
    void testErrorHasColumnNumber() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{\"key\":,\n}"));
        assertTrue(e.getColumn() > 0);
    }

    @Test
    void testErrorHasAbsolutePosition() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{\"a\":,}"));
        assertEquals(5, e.getPosition());
    }

    @Test
    void testManualParseExceptionHasUnknownPosition() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parseList("{\"a\":1}"));
        assertEquals(-1, e.getPosition());
    }

    @Test
    void testErrorMessageContainsLocation() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("[1,2,]"));
        String message = e.getMessage();
        assertTrue(message.contains("Line "));
        assertTrue(message.contains("Column "));
    }

    @Test
    void testLineIncrementsOnNewline() {
        Object result = Json.parse("{\n  \"a\":\n  123\n}");
        assertNotNull(result);
    }

    @Test
    void testLeadingZeroErrorOnLine1() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("01"));
        assertEquals(1, e.getLine());
    }

    @Test
    void testLeadingZeroErrorColumn() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("01"));
        assertTrue(e.getColumn() > 0);
    }

    @Test
    void testErrorInArray() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("[1,,2]"));
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    @Test
    void testMissingColonMessage() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{\"key\" value}"));
        assertTrue(e.getMessage().contains("Expected"));
    }

    @Test
    void testUnexpectedEndOfInput() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("[1,2"));
        assertTrue(
                e.getMessage().contains("end of input") || e.getMessage().contains("Expected ']'"));
    }

    @Test
    void testInvalidEscapeSequence() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\x\""));
        assertTrue(e.getMessage().contains("Invalid escape sequence"));
    }

    @Test
    void testIncompleteUnicodeEscape() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\u123\""));
        assertTrue(e.getMessage().contains("Invalid"));
    }

    @Test
    void testInvalidHexDigit() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uGHIJ\""));
        assertTrue(e.getMessage().contains("Invalid"));
    }

    @Test
    void testLoneSurrogateErrorMessage() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uD800\""));
        assertTrue(e.getMessage().contains("Lone"));
    }

    @Test
    void testControlCharErrorMessage() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\u0001\""));
        assertTrue(e.getMessage().contains("Control"));
    }

    @Test
    void testUnexpectedCharacterError() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("@invalid"));
        assertTrue(e.getMessage().contains("Unexpected character"));
    }

    @Test
    void testExtraCommaError() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("[1,2,3,]"));
        assertTrue(e.getMessage().contains("Expected"));
    }

    @Test
    void testExtraCommaInObjectError() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{\"a\":1,}"));
        assertTrue(e.getMessage().contains("Expected"));
    }

    @Test
    void testMultipleErrorsPosition() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("{"));
        assertTrue(e.getMessage().contains("Expected"));
    }

    @Test
    void testTabInStringRejected() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("\"\t\""));
        assertTrue(e.getMessage().contains("Control"));
    }

    @Test
    void testLineNumberStartsAtOne() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("abc"));
        assertEquals(1, e.getLine());
    }

    @Test
    void testColumnNumberStartsAtOne() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("abc"));
        assertEquals(1, e.getColumn());
    }

    @Test
    void testErrorAfterWhitespace() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("   01"));
        assertEquals(1, e.getLine());
    }

    @Test
    void testErrorMessageIncludesContext() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{\"key\": 123"));
        assertTrue(e.getMessage().contains("Line"));
        assertTrue(e.getMessage().contains("Column"));
    }

    @Test
    void testErrorMessageIncludesCaretSnippet() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{\"key\":,}"));
        assertTrue(e.getMessage().contains("^"));
        assertTrue(e.getMessage().contains("\"key\":"));
    }

    @Test
    void testParseExceptionIsRuntimeException() {
        assertTrue(RuntimeException.class.isAssignableFrom(Json.ParseException.class));
    }

    @Test
    void testParseExceptionCanBeCaught() {
        try {
            Json.parse("invalid json");
            fail("Expected ParseException");
        } catch (Json.ParseException e) {
            assertNotNull(e.getMessage());
            assertTrue(e.getLine() > 0);
            assertTrue(e.getColumn() > 0);
        }
    }

    @Test
    void testTrailingContentAfterValidJson() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("123 456"));
        assertTrue(e.getMessage().contains("end of input"));
    }

    @Test
    void testUnexpectedCharacterInNumber() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("123abc"));
        assertTrue(e.getMessage().contains("end of input"));
    }

    @Test
    void testErrorMessageFormats() {
        Json.ParseException e;

        e = assertThrows(Json.ParseException.class, () -> Json.parse("{unclosed"));
        assertTrue(e.getMessage().startsWith("Line "));
        assertTrue(e.getMessage().contains("Column "));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("[1,2,]"));
        assertTrue(
                e.getMessage().contains("Unexpected character")
                        || e.getMessage().contains("Expected"));
    }

    @Test
    void testLineColumnReporting() {
        Json.ParseException e;

        e = assertThrows(Json.ParseException.class, () -> Json.parse("{\n  \"key\":,\n}"));
        assertEquals(2, e.getLine());

        Object result = Json.parse("{\"key\":\n    \"value\"\n}");
        assertNotNull(result);

        String json =
                "{\n" + "  \"name\": \"John\",\n" + "  \"age\":,\n" + "  \"active\": true\n" + "}";
        e = assertThrows(Json.ParseException.class, () -> Json.parse(json));
        assertEquals(3, e.getLine());
    }

    @Test
    void testErrorMessagesAreClear() {
        assertErrorMessageContains("01", "Leading zeros");
        assertErrorMessageContains("+1", "Unexpected character");
        assertErrorMessageContains(".5", "Unexpected character");
        assertErrorMessageContains("1.", "digit after decimal point");
        assertErrorMessageContains("1e", "Exponent missing");
        assertErrorMessageContains("\"\\uD800\"", "Lone surrogate");
        assertErrorMessageContains("\"\\uDC00\"", "Lone surrogate");
        assertErrorMessageContains("{key:}", "Expected '\"'");
        assertErrorMessageContains("{\"key\":}", "Unexpected character");
        assertErrorMessageContains("[1,,2]", "Unexpected character");
        assertErrorMessageContains("{\"key1\":\"value1\",,\"key2\":\"value2\"}", "Expected '\"'");
    }

    private void assertErrorMessageContains(String json, String expectedSubstring) {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(json));
        assertTrue(
                e.getMessage().contains(expectedSubstring),
                "Expected error message to contain '"
                        + expectedSubstring
                        + "' but got: "
                        + e.getMessage());
    }
}
