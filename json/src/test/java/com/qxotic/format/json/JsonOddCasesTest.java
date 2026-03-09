package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class JsonOddCasesTest {

    @Test
    void testRejectIncompleteLiterals() {
        Json.ParseException e1 = assertThrows(Json.ParseException.class, () -> Json.parse("tru"));
        Json.ParseException e2 = assertThrows(Json.ParseException.class, () -> Json.parse("fals"));
        Json.ParseException e3 = assertThrows(Json.ParseException.class, () -> Json.parse("nul"));

        assertTrue(e1.getMessage().contains("Expected"));
        assertTrue(e2.getMessage().contains("Expected"));
        assertTrue(e3.getMessage().contains("Expected"));
    }

    @Test
    void testRejectLiteralWithSuffix() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("truefalse"));
        assertTrue(e.getMessage().contains("end of input"));
    }

    @Test
    void testRejectArrayWithOnlyComma() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("[, ]"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectObjectWithOnlyComma() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("{,}"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectLeadingCommaInArray() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("[,1]"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectLeadingCommaInObject() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{,\"a\":1}"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectMissingValueAfterComma() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("[1,2,"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectMissingMemberAfterComma() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{\"a\":1,"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectExponentWithoutDigits() {
        Json.ParseException e1 = assertThrows(Json.ParseException.class, () -> Json.parse("1e"));
        Json.ParseException e2 = assertThrows(Json.ParseException.class, () -> Json.parse("1e+"));
        Json.ParseException e3 = assertThrows(Json.ParseException.class, () -> Json.parse("1e-"));

        assertTrue(e1.getMessage().toLowerCase().contains("exponent"));
        assertTrue(e2.getMessage().toLowerCase().contains("exponent"));
        assertTrue(e3.getMessage().toLowerCase().contains("exponent"));
    }

    @Test
    void testRejectPlusSignAndBareNumberTokens() {
        Json.ParseException plus = assertThrows(Json.ParseException.class, () -> Json.parse("+1"));
        Json.ParseException minus = assertThrows(Json.ParseException.class, () -> Json.parse("-"));
        Json.ParseException bareDot =
                assertThrows(Json.ParseException.class, () -> Json.parse(".5"));

        assertTrue(
                plus.getMessage().contains("Unexpected") || plus.getMessage().contains("Expected"));
        assertTrue(minus.getMessage().contains("Expected"));
        assertTrue(
                bareDot.getMessage().contains("Unexpected")
                        || bareDot.getMessage().contains("Expected"));
    }

    @Test
    void testRejectUpperCaseLiterals() {
        Json.ParseException e1 = assertThrows(Json.ParseException.class, () -> Json.parse("TRUE"));
        Json.ParseException e2 = assertThrows(Json.ParseException.class, () -> Json.parse("FALSE"));
        Json.ParseException e3 = assertThrows(Json.ParseException.class, () -> Json.parse("NULL"));

        assertTrue(e1.getMessage().contains("Unexpected"));
        assertTrue(e2.getMessage().contains("Unexpected"));
        assertTrue(e3.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectTrailingGarbageAfterValidJson() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{\"a\":1}\n  x"));
        assertTrue(e.getMessage().contains("end of input"));
    }

    @Test
    void testRejectUnterminatedEscapeAtEndOfString() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"abc\\\""));
        assertTrue(e.getMessage().contains("Unexpected end of input"));
    }

    @Test
    void testRejectNonStringObjectKeyWhenStringifying() {
        Map<Object, Object> map = new LinkedHashMap<>();
        map.put(1, "one");

        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> Json.stringify(map, false));
        assertTrue(e.getMessage().contains("keys must be strings"));
    }

    @Test
    void testRejectNullObjectKeyWhenStringifying() {
        Map<Object, Object> map = new LinkedHashMap<>();
        map.put(null, "value");

        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> Json.stringify(map, false));
        assertTrue(e.getMessage().contains("keys must be strings"));
    }
}
