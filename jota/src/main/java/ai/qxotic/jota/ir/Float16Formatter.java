package ai.qxotic.jota.ir;

import ai.qxotic.jota.BFloat16;

/**
 * Utility class for formatting FP16 and BF16 floating point values with their bit representations
 * and human-readable annotations.
 *
 * <p>Format: {@code f16:0x3C00 (1.0)} or {@code bf16:0x7F80 (+Inf)}
 *
 * <p>Special values are detected and annotated:
 *
 * <ul>
 *   <li>Infinity: {@code (+Inf)}, {@code (-Inf)}
 *   <li>NaN: {@code (NaN)}
 *   <li>Zero: {@code (0.0)}, {@code (-0.0)}
 *   <li>Subnormal: {@code (value) (sub-normal)}
 *   <li>Normal: {@code (value)}
 * </ul>
 */
public final class Float16Formatter {

    private Float16Formatter() {
        // Utility class
    }

    /**
     * Formats a FP16 (IEEE 754 binary16) value.
     *
     * @param bits the 16-bit representation
     * @return formatted string like "f16:0x3C00 (1.0)"
     */
    public static String formatFloat16(short bits) {
        int intBits = bits & 0xFFFF;
        float value = Float.float16ToFloat(bits);
        String hex = String.format("0x%04X", intBits);
        String valueStr = formatValue(value, bits, true);
        return "f16:" + hex + " (" + valueStr + ")";
    }

    /**
     * Formats a BF16 (bfloat16) value.
     *
     * @param bits the 16-bit representation
     * @return formatted string like "bf16:0x3F80 (1.0)"
     */
    public static String formatBFloat16(short bits) {
        int intBits = bits & 0xFFFF;
        float value = BFloat16.toFloat(bits);
        String hex = String.format("0x%04X", intBits);
        String valueStr = formatValue(value, bits, false);
        return "bf16:" + hex + " (" + valueStr + ")";
    }

    private static String formatValue(float value, short bits, boolean isFloat16) {
        // Check for special values
        if (Float.isNaN(value)) {
            return "NaN";
        }
        if (Float.isInfinite(value)) {
            return value > 0 ? "+Inf" : "-Inf";
        }
        if (value == 0.0f) {
            // Check sign bit for -0.0
            return (bits & 0x8000) != 0 ? "-0.0" : "0.0";
        }

        // Check for subnormal
        boolean isSubnormal = isSubnormal(bits, isFloat16);
        String valueStr = formatCompactFloat(value);

        if (isSubnormal) {
            return valueStr + " (sub-normal)";
        }
        return valueStr;
    }

    private static boolean isSubnormal(short bits, boolean isFloat16) {
        int intBits = bits & 0xFFFF;
        if (isFloat16) {
            // FP16: exponent is bits 14-10 (0x7C00), mantissa is bits 9-0 (0x03FF)
            int exponent = intBits & 0x7C00;
            int mantissa = intBits & 0x03FF;
            return exponent == 0 && mantissa != 0;
        } else {
            // BF16: exponent is bits 14-7 (0x7F80), mantissa is bits 6-0 (0x007F)
            int exponent = intBits & 0x7F80;
            int mantissa = intBits & 0x007F;
            return exponent == 0 && mantissa != 0;
        }
    }

    private static String formatCompactFloat(float value) {
        // Format with reasonable precision
        String str = String.valueOf(value);

        // Remove unnecessary trailing zeros for cleaner output
        if (str.contains("E") || str.contains("e")) {
            // Scientific notation - keep as is
            return str;
        }

        if (str.endsWith(".0")) {
            return str.substring(0, str.length() - 2);
        }

        // Limit to reasonable decimal places
        int dotIndex = str.indexOf('.');
        if (dotIndex != -1 && str.length() - dotIndex > 6) {
            // Truncate to 5 decimal places
            return String.format("%.5f", value);
        }

        return str;
    }
}
