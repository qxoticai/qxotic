package com.qxotic.jota.stablehlo.type;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class FloatFormatTest {

    @Test
    void knownEncodings() {
        assertEquals(0x3F800000L, FloatFormat.F32.encode(1.0));
        assertEquals(0x3C00L, FloatFormat.F16.encode(1.0));
        assertEquals(0x3F80L, FloatFormat.BF16.encode(1.0));
        assertEquals(0x7BFFL, FloatFormat.F16.encode(65504.0)); // max finite
        assertEquals(0x7C00L, FloatFormat.F16.encode(65520.0)); // rounds to +inf
        assertEquals(0x0001L, FloatFormat.F16.encode(Math.scalb(1.0, -24))); // min subnormal

        assertEquals(0x38L, FloatFormat.F8E4M3FN.encode(1.0));
        assertEquals(0x7EL, FloatFormat.F8E4M3FN.encode(448.0)); // max finite
        assertEquals(0x7FL, FloatFormat.F8E4M3FN.encode(Double.NaN));
        assertEquals(0x7FL, FloatFormat.F8E4M3FN.encode(1e6)); // no inf: overflow is NaN

        assertEquals(0x3CL, FloatFormat.F8E5M2.encode(1.0));
        assertEquals(0x7CL, FloatFormat.F8E5M2.encode(Double.POSITIVE_INFINITY));

        assertEquals(0x40L, FloatFormat.F8E4M3FNUZ.encode(1.0));
        assertEquals(0x80L, FloatFormat.F8E4M3FNUZ.encode(Double.NaN));
        assertEquals(0x00L, FloatFormat.F8E4M3FNUZ.encode(-0.0)); // no negative zero

        assertEquals(0x7FL, FloatFormat.F8E8M0FNU.encode(1.0));
        assertEquals(0xFFL, FloatFormat.F8E8M0FNU.encode(Double.NaN));
        assertEquals(126L, FloatFormat.F8E8M0FNU.encode(0.375)); // tie: even exponent field (0.5)

        assertEquals(0x7L, FloatFormat.F4E2M1FN.encode(6.0)); // max finite
        assertEquals(0x1L, FloatFormat.F4E2M1FN.encode(0.5)); // subnormal
        assertEquals(0x7L, FloatFormat.F4E2M1FN.encode(7.0)); // FINITE formats saturate
        assertEquals(0xAL, FloatFormat.F4E2M1FN.encode(-1.0));
    }

    @Test
    void unsignedFormatsMapNegativesToNaN() {
        assertTrue(Double.isNaN(FloatFormat.F8E5M3FNU.decode(FloatFormat.F8E5M3FNU.encode(-1.0))));
        assertEquals(0L, FloatFormat.F8E5M3FNU.encode(-0.0));
    }

    /** decode then encode is the identity on every non-NaN bit pattern of the small formats. */
    @ParameterizedTest
    @EnumSource(
            value = FloatFormat.class,
            names = {
                "F4E2M1FN",
                "F6E2M3FN",
                "F6E3M2FN",
                "F8E3M4",
                "F8E4M3",
                "F8E4M3FN",
                "F8E4M3FNUZ",
                "F8E4M3B11FNUZ",
                "F8E5M2",
                "F8E5M2FNUZ",
                "F8E5M3FNU",
                "F8E8M0FNU"
            })
    void exhaustiveRoundTrip(FloatFormat format) {
        for (long bits = 0; bits < (1L << format.bitWidth()); bits++) {
            double v = format.decode(bits);
            if (Double.isNaN(v)) {
                continue;
            }
            assertEquals(bits, format.encode(v), format + " pattern " + Long.toHexString(bits));
        }
    }

    @Test
    void f16MatchesTheJdk() {
        for (int bits = 0; bits < 0x10000; bits++) {
            float jdk = Float.float16ToFloat((short) bits);
            double ours = FloatFormat.F16.decode(bits);
            if (Float.isNaN(jdk)) {
                assertTrue(Double.isNaN(ours));
                continue;
            }
            assertEquals(jdk, ours, 0.0);
            assertEquals(bits, FloatFormat.F16.encode(ours));
        }
    }

    @Test
    void lookupByMlirName() {
        for (FloatFormat f : FloatFormat.values()) {
            assertEquals(f, FloatFormat.byName(f.mlirName()));
        }
    }
}
