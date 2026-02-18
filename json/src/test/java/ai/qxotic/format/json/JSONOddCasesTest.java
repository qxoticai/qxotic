package ai.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class JSONOddCasesTest {

    @Test
    void testRejectIncompleteLiterals() {
        JSON.ParseException e1 = assertThrows(JSON.ParseException.class, () -> JSON.parse("tru"));
        JSON.ParseException e2 = assertThrows(JSON.ParseException.class, () -> JSON.parse("fals"));
        JSON.ParseException e3 = assertThrows(JSON.ParseException.class, () -> JSON.parse("nul"));

        assertTrue(e1.getMessage().contains("Expected"));
        assertTrue(e2.getMessage().contains("Expected"));
        assertTrue(e3.getMessage().contains("Expected"));
    }

    @Test
    void testRejectLiteralWithSuffix() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("truefalse"));
        assertTrue(e.getMessage().contains("end of input"));
    }

    @Test
    void testRejectArrayWithOnlyComma() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("[, ]"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectObjectWithOnlyComma() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("{,}"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectLeadingCommaInArray() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("[,1]"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectLeadingCommaInObject() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{,\"a\":1}"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectMissingValueAfterComma() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("[1,2,"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectMissingMemberAfterComma() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"a\":1,"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectExponentWithoutDigits() {
        JSON.ParseException e1 = assertThrows(JSON.ParseException.class, () -> JSON.parse("1e"));
        JSON.ParseException e2 = assertThrows(JSON.ParseException.class, () -> JSON.parse("1e+"));
        JSON.ParseException e3 = assertThrows(JSON.ParseException.class, () -> JSON.parse("1e-"));

        assertTrue(e1.getMessage().toLowerCase().contains("exponent"));
        assertTrue(e2.getMessage().toLowerCase().contains("exponent"));
        assertTrue(e3.getMessage().toLowerCase().contains("exponent"));
    }

    @Test
    void testRejectPlusSignAndBareNumberTokens() {
        JSON.ParseException plus = assertThrows(JSON.ParseException.class, () -> JSON.parse("+1"));
        JSON.ParseException minus = assertThrows(JSON.ParseException.class, () -> JSON.parse("-"));
        JSON.ParseException bareDot =
                assertThrows(JSON.ParseException.class, () -> JSON.parse(".5"));

        assertTrue(
                plus.getMessage().contains("Unexpected") || plus.getMessage().contains("Expected"));
        assertTrue(minus.getMessage().contains("Expected"));
        assertTrue(
                bareDot.getMessage().contains("Unexpected")
                        || bareDot.getMessage().contains("Expected"));
    }

    @Test
    void testRejectUpperCaseLiterals() {
        JSON.ParseException e1 = assertThrows(JSON.ParseException.class, () -> JSON.parse("TRUE"));
        JSON.ParseException e2 = assertThrows(JSON.ParseException.class, () -> JSON.parse("FALSE"));
        JSON.ParseException e3 = assertThrows(JSON.ParseException.class, () -> JSON.parse("NULL"));

        assertTrue(e1.getMessage().contains("Unexpected"));
        assertTrue(e2.getMessage().contains("Unexpected"));
        assertTrue(e3.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectTrailingGarbageAfterValidJson() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"a\":1}\n  x"));
        assertTrue(e.getMessage().contains("end of input"));
    }

    @Test
    void testRejectUnterminatedEscapeAtEndOfString() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"abc\\\""));
        assertTrue(e.getMessage().contains("Unexpected end of input"));
    }

    @Test
    void testRejectNonStringObjectKeyWhenStringifying() {
        Map<Object, Object> map = new LinkedHashMap<>();
        map.put(1, "one");

        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> JSON.stringify(map, false));
        assertTrue(e.getMessage().contains("keys must be strings"));
    }

    @Test
    void testRejectNullObjectKeyWhenStringifying() {
        Map<Object, Object> map = new LinkedHashMap<>();
        map.put(null, "value");

        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> JSON.stringify(map, false));
        assertTrue(e.getMessage().contains("keys must be strings"));
    }
}
