package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class JSONNumberTest {

    @Test
    void testZero() {
        assertEquals(0L, JSON.parse("0"));
    }

    @Test
    void testNegativeZero() {
        Object parsed = JSON.parse("-0");
        assertInstanceOf(BigDecimal.class, parsed);
    }

    @Test
    void testNegativeZeroWithDecimal() {
        Object parsed = JSON.parse("-0.0");
        assertInstanceOf(BigDecimal.class, parsed);
    }

    @Test
    void testPositiveInteger() {
        assertEquals(123L, JSON.parse("123"));
    }

    @Test
    void testNegativeInteger() {
        assertEquals(-123L, JSON.parse("-123"));
    }

    @Test
    void testLeadingZeroRejected() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("01"));
        assertTrue(e.getMessage().contains("Leading zeros"));
    }

    @Test
    void testMultipleLeadingZerosRejected() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("001"));
        assertTrue(e.getMessage().contains("Leading zeros"));
    }

    @Test
    void testPlusSignRejected() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("+1"));
        assertTrue(e.getMessage().contains("Unexpected character"));
    }

    @Test
    void testDecimalPointWithoutDigitsAfterRejected() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("1."));
        assertTrue(e.getMessage().contains("digit after decimal point"));
    }

    @Test
    void testDecimalPointWithoutDigitsBeforeRejected() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse(".5"));
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    @Test
    void testSimpleDecimal() {
        Object parsed = JSON.parse("0.5");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(new BigDecimal("0.5"), parsed);
    }

    @Test
    void testDecimalWithoutLeadingZero() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse(".5"));
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    @Test
    void testExponentPositive() {
        Object parsed = JSON.parse("1e10");
        assertInstanceOf(BigDecimal.class, parsed);
    }

    @Test
    void testExponentUppercase() {
        Object parsed = JSON.parse("1E10");
        assertInstanceOf(BigDecimal.class, parsed);
    }

    @Test
    void testExponentNegative() {
        Object parsed = JSON.parse("1E-2", JSON.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertInstanceOf(BigDecimal.class, parsed);
    }

    @Test
    void testExponentPositiveSign() {
        Object parsed = JSON.parse("1e+2");
        assertInstanceOf(BigDecimal.class, parsed);
    }

    @Test
    void testExponentMissingDigitsRejected() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("1e"));
        assertTrue(e.getMessage().contains("exponent missing digits"));
    }

    @Test
    void testDecimalWithExponent() {
        Object parsed = JSON.parse("123.456e789");
        assertInstanceOf(BigDecimal.class, parsed);
    }

    @Test
    void testLargeInteger() {
        Object parsed = JSON.parse("9999999999999999999999999999999999999999");
        assertInstanceOf(BigInteger.class, parsed);
        assertEquals(new BigInteger("9999999999999999999999999999999999999999"), parsed);
    }

    @Test
    void testLargeDecimal() {
        String json = "12345678901234567890.12345678901234567890";
        Object parsed = JSON.parse(json);
        assertInstanceOf(BigDecimal.class, parsed);
    }

    @Test
    void testFractionalNumberWithTrailingZeros() {
        Object parsed = JSON.parse("1.500");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(new BigDecimal("1.5")));
    }

    @Test
    void testIntegerWithExponent() {
        Object parsed = JSON.parse("1000E2");
        assertInstanceOf(BigDecimal.class, parsed);
    }

    @Test
    void testSmallFraction() {
        Object parsed = JSON.parse("0.000001");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(new BigDecimal("0.000001"), parsed);
    }

    @Test
    void testLargeExponent() {
        Object parsed = JSON.parse("1e308");
        assertInstanceOf(BigDecimal.class, parsed);
    }

    @Test
    void testNegativeExponent() {
        Object parsed = JSON.parse("1e-307");
        assertInstanceOf(BigDecimal.class, parsed);
    }

    @Test
    void testZeroWithExponent() {
        Object parsed = JSON.parse("0e10");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(new BigDecimal("0"), parsed);
    }

    @Test
    void testNegativeLargeInteger() {
        Object parsed = JSON.parse("-9999999999999999999999999999999999999999");
        assertInstanceOf(BigInteger.class, parsed);
        assertEquals(new BigInteger("-9999999999999999999999999999999999999999"), parsed);
    }

    @Test
    void testNumberInArray() {
        List<Object> expected = new ArrayList<>();
        expected.add(1L);
        expected.add(2L);
        expected.add(3L);
        assertEquals(expected, JSON.parse("[1,2,3]"));
    }

    @Test
    void testNumberInObject() {
        List<Object> result = (List<Object>) JSON.parse("[1,2,3]");
        assertEquals(3, result.size());
        assertEquals(1L, result.get(0));
        assertEquals(2L, result.get(1));
        assertEquals(3L, result.get(2));
    }

    @Test
    void testMultipleLeadingZerosWithDecimalRejected() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("01.5"));
        assertTrue(e.getMessage().contains("Leading zeros"));
    }

    @Test
    void testZeroFollowedByNumber() {
        assertEquals(10L, JSON.parse("10"));
    }

    @Test
    void testZeroWithDecimalPoint() {
        Object parsed = JSON.parse("0.0");
        assertInstanceOf(BigDecimal.class, parsed);
    }

    @Test
    void testNegativeDecimal() {
        Object parsed = JSON.parse("-1.5");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(new BigDecimal("-1.5"), parsed);
    }

    @Test
    void testNegativeExponentValue() {
        Object parsed = JSON.parse("-1e10");
        assertInstanceOf(BigDecimal.class, parsed);
    }

    // ===== Extreme value tests =====

    @Test
    void testExtremePositiveExponent() {
        // Very large positive exponents
        Object parsed = JSON.parse("1e999");
        assertInstanceOf(BigDecimal.class, parsed);
        BigDecimal bd = (BigDecimal) parsed;
        assertEquals(0, bd.compareTo(new BigDecimal("1E+999")));

        parsed = JSON.parse("1.5e500");
        assertInstanceOf(BigDecimal.class, parsed);
        bd = (BigDecimal) parsed;
        assertEquals(0, bd.compareTo(new BigDecimal("1.5E+500")));
    }

    @Test
    void testExtremeNegativeExponent() {
        // Very small numbers with large negative exponents
        Object parsed = JSON.parse("1e-999");
        assertInstanceOf(BigDecimal.class, parsed);
        BigDecimal bd = (BigDecimal) parsed;
        assertEquals(0, bd.compareTo(new BigDecimal("1E-999")));

        parsed = JSON.parse("1.5e-500");
        assertInstanceOf(BigDecimal.class, parsed);
        bd = (BigDecimal) parsed;
        assertEquals(0, bd.compareTo(new BigDecimal("1.5E-500")));
    }

    @Test
    void testVeryHighPrecisionDecimals() {
        // Numbers with many decimal places
        String highPrecision = "0.123456789012345678901234567890123456789012345678901234567890";
        Object parsed = JSON.parse(highPrecision);
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(new BigDecimal(highPrecision), parsed);

        // Many digits after decimal
        String manyDigits = "1." + "0".repeat(100) + "1";
        parsed = JSON.parse(manyDigits);
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(new BigDecimal(manyDigits), parsed);
    }

    @Test
    void testScientificNotationEdgeCases() {
        // Zero exponent
        Object parsed = JSON.parse("1e0");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(new BigDecimal("1")));

        parsed = JSON.parse("1.5e0");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(new BigDecimal("1.5")));

        // Leading zeros in exponent
        parsed = JSON.parse("1e01");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(new BigDecimal("1E+1")));

        parsed = JSON.parse("1e001");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(new BigDecimal("1E+1")));

        parsed = JSON.parse("1.5e-01");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(new BigDecimal("1.5E-1")));
    }

    @Test
    void testNumberAtLongBoundary() {
        // Test numbers around Long.MAX_VALUE and Long.MIN_VALUE

        // Just below Long.MAX_VALUE
        long maxLongMinusOne = Long.MAX_VALUE - 1;
        Object parsed = JSON.parse(Long.toString(maxLongMinusOne));
        assertEquals(maxLongMinusOne, parsed);

        // Long.MAX_VALUE itself
        parsed = JSON.parse(Long.toString(Long.MAX_VALUE));
        assertEquals(Long.MAX_VALUE, parsed);

        // Just above Long.MAX_VALUE (should be BigInteger)
        BigInteger aboveMax = BigInteger.valueOf(Long.MAX_VALUE).add(BigInteger.ONE);
        parsed = JSON.parse(aboveMax.toString());
        assertInstanceOf(BigInteger.class, parsed);
        assertEquals(aboveMax, parsed);

        // Just above Long.MIN_VALUE
        long minLongPlusOne = Long.MIN_VALUE + 1;
        parsed = JSON.parse(Long.toString(minLongPlusOne));
        assertEquals(minLongPlusOne, parsed);

        // Long.MIN_VALUE itself
        parsed = JSON.parse(Long.toString(Long.MIN_VALUE));
        assertEquals(Long.MIN_VALUE, parsed);

        // Just below Long.MIN_VALUE (should be BigInteger)
        BigInteger belowMin = BigInteger.valueOf(Long.MIN_VALUE).subtract(BigInteger.ONE);
        parsed = JSON.parse(belowMin.toString());
        assertInstanceOf(BigInteger.class, parsed);
        assertEquals(belowMin, parsed);
    }

    @Test
    void testVeryLargeBigIntegers() {
        // Extremely large integers (hundreds of digits)
        StringBuilder hugeNumber = new StringBuilder();
        hugeNumber.append("9".repeat(100));

        String hugeStr = hugeNumber.toString();
        Object parsed = JSON.parse(hugeStr);
        assertInstanceOf(BigInteger.class, parsed);
        assertEquals(new BigInteger(hugeStr), parsed);

        // Negative version
        parsed = JSON.parse("-" + hugeStr);
        assertInstanceOf(BigInteger.class, parsed);
        assertEquals(new BigInteger("-" + hugeStr), parsed);

        // Mixed digits
        hugeNumber = new StringBuilder();
        hugeNumber.append("1234567890".repeat(50));
        hugeStr = hugeNumber.toString();
        parsed = JSON.parse(hugeStr);
        assertInstanceOf(BigInteger.class, parsed);
        assertEquals(new BigInteger(hugeStr), parsed);
    }

    @Test
    void testNumbersWithExponentAndHighPrecision() {
        // Complex numbers with both exponent and high precision
        String complex = "12345678901234567890.12345678901234567890e100";
        Object parsed = JSON.parse(complex);
        assertInstanceOf(BigDecimal.class, parsed);

        // Verify it parses to correct value
        BigDecimal expected =
                new BigDecimal("12345678901234567890.12345678901234567890").scaleByPowerOfTen(100);
        BigDecimal actual = (BigDecimal) parsed;

        // Compare with tolerance for scale differences
        assertEquals(0, expected.compareTo(actual));
    }

    @Test
    void testZeroVariations() {
        // Many ways to write zero
        assertEquals(0L, JSON.parse("0"));

        Object parsed = JSON.parse("0.0");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(BigDecimal.ZERO));

        parsed = JSON.parse("0e0");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(BigDecimal.ZERO));

        parsed = JSON.parse("0e10");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(BigDecimal.ZERO));

        parsed = JSON.parse("0e-10");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(BigDecimal.ZERO));

        parsed = JSON.parse("0.000");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(BigDecimal.ZERO));
    }

    @Test
    void testNegativeZeroEdgeCases() {
        // Negative zero variations
        Object parsed = JSON.parse("-0");
        assertInstanceOf(BigDecimal.class, parsed);
        BigDecimal negZero = (BigDecimal) parsed;
        // Note: BigDecimal("-0").equals(BigDecimal("0")) is true
        assertEquals(0, negZero.compareTo(BigDecimal.ZERO));

        parsed = JSON.parse("-0.0");
        assertInstanceOf(BigDecimal.class, parsed);
        negZero = (BigDecimal) parsed;
        assertEquals(0, negZero.compareTo(BigDecimal.ZERO));

        parsed = JSON.parse("-0e0");
        assertInstanceOf(BigDecimal.class, parsed);
        negZero = (BigDecimal) parsed;
        assertEquals(0, negZero.compareTo(BigDecimal.ZERO));

        parsed = JSON.parse("-0e10");
        assertInstanceOf(BigDecimal.class, parsed);
        negZero = (BigDecimal) parsed;
        assertEquals(0, negZero.compareTo(BigDecimal.ZERO));
    }

    @Test
    void testExponentWithManyDigits() {
        // Exponents with many digits
        Object parsed = JSON.parse("1e1234");
        assertInstanceOf(BigDecimal.class, parsed);

        parsed = JSON.parse("1e-1234");
        assertInstanceOf(BigDecimal.class, parsed);

        // Very large exponent value (but not too large)
        String hugeExp = "1e" + "9".repeat(5); // 1e99999
        parsed = JSON.parse(hugeExp);
        assertInstanceOf(BigDecimal.class, parsed);
    }
}
