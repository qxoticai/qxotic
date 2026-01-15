
package ai.qxotic.jota;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BFloat16Test {

    @Test
    void roundTripPreservesZeroAndSign() {
        assertEquals(0, BFloat16.fromFloat(0.0f));
        assertEquals(0x8000, BFloat16.fromFloat(-0.0f) & 0xffff);
        assertEquals(0x3f80, BFloat16.fromFloat(1.0f) & 0xffff);
        assertEquals(0x7f80, BFloat16.fromFloat(Float.POSITIVE_INFINITY) & 0xffff);
        assertEquals(0xff80, BFloat16.fromFloat(Float.NEGATIVE_INFINITY) & 0xffff);
        assertTrue(Float.isNaN(BFloat16.toFloat(BFloat16.fromFloat(Float.NaN))));
    }

    @Test
    void float16ToBFloat16HandlesSpecials() {
        short halfInf = (short) 0x7c00;
        short halfNegInf = (short) 0xfc00;
        short halfNaN = (short) 0x7e01;

        assertEquals(0x7f80, BFloat16.fromFloat16(halfInf) & 0xffff);
        assertEquals(0xff80, BFloat16.fromFloat16(halfNegInf) & 0xffff);

        int nanBits = BFloat16.fromFloat16(halfNaN) & 0xffff;
        assertEquals(0x7f80, nanBits & 0x7f80);
        assertTrue((nanBits & 0x007f) != 0);
    }

    @Test
    void bfloat16ToFloat16HandlesSpecials() {
        short bfloatInf = (short) 0x7f80;
        short bfloatNegInf = (short) 0xff80;
        short bfloatNaN = (short) 0x7fc1;

        assertEquals(0x7c00, BFloat16.toFloat16(bfloatInf) & 0xffff);
        assertEquals(0xfc00, BFloat16.toFloat16(bfloatNegInf) & 0xffff);

        int halfNaN = BFloat16.toFloat16(bfloatNaN) & 0xffff;
        assertEquals(0x7c00, halfNaN & 0x7c00);
        assertTrue((halfNaN & 0x03ff) != 0);
    }

    @Test
    void float16PatternsPreserveSpecials() {
        for (int bits = 0; bits <= 0xffff; bits++) {
            short half = (short) bits;
            float value = Float.float16ToFloat(half);
            short bfloat = BFloat16.fromFloat16(half);
            float bfValue = BFloat16.toFloat(bfloat);

            if (Float.isNaN(value)) {
                assertTrue(Float.isNaN(bfValue));
            } else if (Float.isInfinite(value)) {
                assertEquals(Float.floatToRawIntBits(value) < 0, Float.floatToRawIntBits(bfValue) < 0);
                assertTrue(Float.isInfinite(bfValue));
            } else {
                assertTrue(Float.isFinite(bfValue));
            }
        }
    }

    @Test
    void bfloat16PatternsToFloat16PreserveSpecials() {
        for (int bits = 0; bits <= 0xffff; bits++) {
            short bfloat = (short) bits;
            float value = BFloat16.toFloat(bfloat);
            short half = BFloat16.toFloat16(bfloat);
            float halfValue = Float.float16ToFloat(half);

            if (Float.isNaN(value)) {
                assertTrue(Float.isNaN(halfValue));
            } else if (Float.isInfinite(value)) {
                assertEquals(Float.floatToRawIntBits(value) < 0, Float.floatToRawIntBits(halfValue) < 0);
                assertTrue(Float.isInfinite(halfValue));
            } else if (Float.isInfinite(halfValue)) {
                assertEquals(Float.floatToRawIntBits(value) < 0, Float.floatToRawIntBits(halfValue) < 0);
            } else {
                assertTrue(Float.isFinite(halfValue));
            }
        }
    }

    @Test
    void toFloatRestoresExactBFloatValue() {
        assertEquals(1.0f, BFloat16.toFloat((short) 0x3f80));
        assertEquals(-2.0f, BFloat16.toFloat((short) 0xc000));
    }

    @Test
    void roundToNearestEven() {
        float roundDown = Float.intBitsToFloat(0x3f808000);
        float roundUp = Float.intBitsToFloat(0x3f818000);

        assertEquals(0x3f80, BFloat16.fromFloat(roundDown) & 0xffff);
        assertEquals(0x3f82, BFloat16.fromFloat(roundUp) & 0xffff);
    }

    @Test
    void nanConversionKeepsMantissa() {
        short encoded = BFloat16.fromFloat(Float.intBitsToFloat(0x7fc01234));
        int bits = encoded & 0xffff;
        assertEquals(0x7f80, bits & 0x7f80);
        assertTrue((bits & 0x007f) != 0);
    }
}
