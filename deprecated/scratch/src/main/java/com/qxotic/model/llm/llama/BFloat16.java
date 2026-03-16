package com.qxotic.model.llm.llama;

public class BFloat16 {
    /**
     * Converts a 32-bit float to a bfloat16 representation stored in a short. BFloat16 keeps the
     * sign bit and exponent (8 bits) from float32, and the 7 most significant bits of the mantissa.
     *
     * @param value The float value to convert
     * @return A short containing the bfloat16 representation
     */
    public static short floatToBFloat16(float value) {
        // Get the bit representation of the float
        int bits = Float.floatToRawIntBits(value);

        // Extract the upper 16 bits which contain:
        // - Sign bit (1 bit)
        // - Exponent (8 bits)
        // - Most significant 7 bits of mantissa
        short bfloat16 = (short) (bits >>> 16);

        // Implement rounding
        int remainingBits = bits & 0xFFFF;
        if (remainingBits > 0x8000 || (remainingBits == 0x8000 && (bfloat16 & 1) != 0)) {
            // Round up
            bfloat16++;
        }

        return bfloat16;
    }

    /**
     * Converts a bfloat16 value (stored in a short) back to a 32-bit float.
     *
     * @param value The bfloat16 value stored in a short
     * @return The converted float value
     */
    public static float bfloat16ToFloat(short value) {
        // Convert the 16-bit value to a 32-bit float representation
        // by shifting left 16 bits (padding with zeros)
        return Float.intBitsToFloat(((int) value) << 16);
    }

    public static final int BYTES = 2;
}
