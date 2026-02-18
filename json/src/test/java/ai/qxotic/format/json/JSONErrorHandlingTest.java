package ai.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class JSONErrorHandlingTest {

    @Test
    void testErrorHasLineNumber() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{\n  \"key\":,\n}"));
        assertEquals(2, e.getLine());
    }

    @Test
    void testErrorHasColumnNumber() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"key\":,\n}"));
        assertTrue(e.getColumn() > 0);
    }

    @Test
    void testErrorHasAbsolutePosition() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"a\":,}"));
        assertEquals(5, e.getPosition());
    }

    @Test
    void testManualParseExceptionHasUnknownPosition() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parseArray("{\"a\":1}"));
        assertEquals(-1, e.getPosition());
    }

    @Test
    void testErrorMessageContainsLocation() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("[1,2,]"));
        String message = e.getMessage();
        assertTrue(message.contains("Line "));
        assertTrue(message.contains("Column "));
    }

    @Test
    void testLineIncrementsOnNewline() {
        Object result = JSON.parse("{\n  \"a\":\n  123\n}");
        assertNotNull(result);
    }

    @Test
    void testLeadingZeroErrorOnLine1() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("01"));
        assertEquals(1, e.getLine());
    }

    @Test
    void testLeadingZeroErrorColumn() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("01"));
        assertTrue(e.getColumn() > 0);
    }

    @Test
    void testErrorInArray() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("[1,,2]"));
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    @Test
    void testMissingColonMessage() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"key\" value}"));
        assertTrue(e.getMessage().contains("Expected"));
    }

    @Test
    void testUnexpectedEndOfInput() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("[1,2"));
        assertTrue(
                e.getMessage().contains("end of input") || e.getMessage().contains("Expected ']'"));
    }

    @Test
    void testInvalidEscapeSequence() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\x\""));
        assertTrue(e.getMessage().contains("Invalid escape sequence"));
    }

    @Test
    void testIncompleteUnicodeEscape() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\u123\""));
        assertTrue(e.getMessage().contains("Invalid"));
    }

    @Test
    void testInvalidHexDigit() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\uGHIJ\""));
        assertTrue(e.getMessage().contains("Invalid"));
    }

    @Test
    void testLoneSurrogateErrorMessage() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\uD800\""));
        assertTrue(e.getMessage().contains("Lone"));
    }

    @Test
    void testControlCharErrorMessage() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\u0001\""));
        assertTrue(e.getMessage().contains("Control"));
    }

    @Test
    void testUnexpectedCharacterError() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("@invalid"));
        assertTrue(e.getMessage().contains("Unexpected character"));
    }

    @Test
    void testExtraCommaError() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("[1,2,3,]"));
        assertTrue(e.getMessage().contains("Expected"));
    }

    @Test
    void testExtraCommaInObjectError() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"a\":1,}"));
        assertTrue(e.getMessage().contains("Expected"));
    }

    @Test
    void testMultipleErrorsPosition() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("{"));
        assertTrue(e.getMessage().contains("Expected"));
    }

    @Test
    void testTabInStringRejected() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\t\""));
        assertTrue(e.getMessage().contains("Control"));
    }

    @Test
    void testLineNumberStartsAtOne() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("abc"));
        assertEquals(1, e.getLine());
    }

    @Test
    void testColumnNumberStartsAtOne() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("abc"));
        assertEquals(1, e.getColumn());
    }

    @Test
    void testErrorAfterWhitespace() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("   01"));
        assertEquals(1, e.getLine());
    }

    @Test
    void testErrorMessageIncludesContext() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"key\": 123"));
        assertTrue(e.getMessage().contains("Line"));
        assertTrue(e.getMessage().contains("Column"));
    }

    @Test
    void testErrorMessageIncludesCaretSnippet() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"key\":,}"));
        assertTrue(e.getMessage().contains("^"));
        assertTrue(e.getMessage().contains("\"key\":"));
    }

    @Test
    void testParseExceptionIsRuntimeException() {
        assertTrue(RuntimeException.class.isAssignableFrom(JSON.ParseException.class));
    }

    @Test
    void testParseExceptionCanBeCaught() {
        try {
            JSON.parse("invalid json");
            fail("Expected ParseException");
        } catch (JSON.ParseException e) {
            assertNotNull(e.getMessage());
            assertTrue(e.getLine() > 0);
            assertTrue(e.getColumn() > 0);
        }
    }

    @Test
    void testTrailingContentAfterValidJson() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("123 456"));
        assertTrue(e.getMessage().contains("end of input"));
    }

    @Test
    void testUnexpectedCharacterInNumber() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("123abc"));
        assertTrue(e.getMessage().contains("end of input"));
    }

    @Test
    void testErrorMessageFormats() {
        JSON.ParseException e;

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("{unclosed"));
        assertTrue(e.getMessage().startsWith("Line "));
        assertTrue(e.getMessage().contains("Column "));

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("[1,2,]"));
        assertTrue(
                e.getMessage().contains("Unexpected character")
                        || e.getMessage().contains("Expected"));
    }

    @Test
    void testLineColumnReporting() {
        JSON.ParseException e;

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("{\n  \"key\":,\n}"));
        assertEquals(2, e.getLine());

        Object result = JSON.parse("{\"key\":\n    \"value\"\n}");
        assertNotNull(result);

        String json =
                "{\n" + "  \"name\": \"John\",\n" + "  \"age\":,\n" + "  \"active\": true\n" + "}";
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse(json));
        assertEquals(3, e.getLine());
    }

    @Test
    void testErrorMessagesAreClear() {
        assertErrorMessageContains("01", "Leading zeros");
        assertErrorMessageContains("+1", "Unexpected character");
        assertErrorMessageContains(".5", "Unexpected character");
        assertErrorMessageContains("1.", "digit after decimal point");
        assertErrorMessageContains("1e", "exponent missing");
        assertErrorMessageContains("\"\\uD800\"", "Lone surrogate");
        assertErrorMessageContains("\"\\uDC00\"", "Lone surrogate");
        assertErrorMessageContains("{key:}", "Expected '\"'");
        assertErrorMessageContains("{\"key\":}", "Unexpected character");
        assertErrorMessageContains("[1,,2]", "Unexpected character");
        assertErrorMessageContains("{\"key1\":\"value1\",,\"key2\":\"value2\"}", "Expected '\"'");
    }

    private void assertErrorMessageContains(String json, String expectedSubstring) {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse(json));
        assertTrue(
                e.getMessage().contains(expectedSubstring),
                "Expected error message to contain '"
                        + expectedSubstring
                        + "' but got: "
                        + e.getMessage());
    }
}
