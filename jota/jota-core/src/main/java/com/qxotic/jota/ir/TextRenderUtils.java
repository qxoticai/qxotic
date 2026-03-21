package com.qxotic.jota.ir;

import com.qxotic.jota.BFloat16;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.ir.tir.BinaryOperator;
import com.qxotic.jota.ir.tir.ReductionOperator;
import com.qxotic.jota.ir.tir.TernaryOperator;
import com.qxotic.jota.ir.tir.UnaryOperator;

/**
 * Shared utilities for text rendering of IR graphs (TIR and LIR).
 *
 * <p>Provides common formatting functions to ensure consistency across different IR
 * representations.
 */
public final class TextRenderUtils {

    private TextRenderUtils() {
        // Utility class
    }

    /**
     * Formats a data type using its toString representation.
     *
     * @param dataType the data type to format
     * @return lowercase string representation (e.g., "fp32", "i32")
     */
    public static String formatDataType(DataType dataType) {
        return dataType.toString();
    }

    /**
     * Formats a tuple of long values as (a, b, c).
     *
     * @param values the values to format
     * @return formatted tuple string
     */
    public static String formatTuple(long[] values) {
        if (values == null || values.length == 0) {
            return "()";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("(");
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(values[i]);
        }
        sb.append(")");
        return sb.toString();
    }

    /**
     * Checks if any stride is zero (broadcast pattern).
     *
     * @param strides the strides to check
     * @return true if any stride is zero
     */
    public static boolean isBroadcast(long[] strides) {
        if (strides == null) {
            return false;
        }
        for (long stride : strides) {
            if (stride == 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * Checks if the given shape and strides represent a contiguous (row-major) layout using byte
     * strides.
     *
     * <p>This is used for LIR which stores byte strides.
     *
     * @param shape the shape dimensions
     * @param strides the strides in bytes
     * @param elementSize the element size in bytes
     * @return true if contiguous, false otherwise
     */
    public static boolean isContiguous(long[] shape, long[] strides, int elementSize) {
        if (shape == null || strides == null || shape.length == 0) {
            return true;
        }
        if (shape.length != strides.length) {
            return false;
        }

        long expectedStride = elementSize;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expectedStride) {
                return false;
            }
            expectedStride *= shape[i];
        }
        return true;
    }

    /**
     * Checks if the given shape and strides represent a contiguous (row-major) layout using element
     * strides.
     *
     * <p>This is used for TIR which stores element strides (stride=1 means contiguous).
     *
     * @param shape the shape dimensions
     * @param strides the strides in elements
     * @return true if contiguous, false otherwise
     */
    public static boolean isContiguous(long[] shape, long[] strides) {
        if (shape == null || strides == null || shape.length == 0) {
            return true;
        }
        if (shape.length != strides.length) {
            return false;
        }

        long expectedStride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expectedStride) {
                return false;
            }
            expectedStride *= shape[i];
        }
        return true;
    }

    /**
     * Formats a buffer reference with shape and strides.
     *
     * <p>Format examples:
     *
     * <ul>
     *   <li>Normal: {@code %in0: fp32[(3):(1)] contiguous}
     *   <li>Strided: {@code %in0: fp32[(3):(4)] strided}
     *   <li>Broadcast: {@code %in0: fp32[(3):(0)] broadcasted scalar}
     *   <li>Scalar: {@code %in0: fp32 scalar}
     * </ul>
     *
     * @param prefix the prefix ("in" or "out")
     * @param id the buffer id
     * @param dataType the data type
     * @param shape the shape dimensions
     * @param strides the strides
     * @param elementBased if true, use element-based contiguity check (TIR), otherwise byte-based
     *     (LIR)
     * @return formatted buffer string
     */
    public static String formatBuffer(
            String prefix,
            int id,
            DataType dataType,
            long[] shape,
            long[] strides,
            boolean elementBased) {
        StringBuilder sb = new StringBuilder();
        sb.append("%").append(prefix).append(id).append(": ");
        sb.append(formatDataType(dataType));

        // Check if scalar (empty shape)
        if (shape == null || shape.length == 0) {
            sb.append(" scalar");
            return sb.toString();
        }

        sb.append("[");
        sb.append(formatTuple(shape));
        sb.append(":");
        sb.append(formatTuple(strides));
        sb.append("]");

        // Check for broadcast pattern
        if (isBroadcast(strides)) {
            sb.append(" broadcasted scalar");
            return sb.toString();
        }

        // Check contiguity
        boolean contiguous =
                elementBased
                        ? isContiguous(shape, strides)
                        : isContiguous(shape, strides, (int) dataType.byteSize());
        if (contiguous) {
            sb.append(" contiguous");
        } else {
            sb.append(" strided");
        }

        return sb.toString();
    }

    /**
     * Formats a buffer reference with shape and strides using byte-based contiguity (for LIR).
     *
     * @param prefix the prefix ("in" or "out")
     * @param id the buffer id
     * @param dataType the data type
     * @param shape the shape dimensions
     * @param strides the strides in bytes
     * @return formatted buffer string
     */
    public static String formatBuffer(
            String prefix, int id, DataType dataType, long[] shape, long[] strides) {
        return formatBuffer(prefix, id, dataType, shape, strides, false);
    }

    /**
     * Formats a buffer reference with Layout using byte-based contiguity (for LIR).
     *
     * @param prefix the prefix ("in" or "out")
     * @param id the buffer id
     * @param dataType the data type
     * @param layout the layout (uses element strides, converted to byte strides for display)
     * @return formatted buffer string
     */
    public static String formatBuffer(String prefix, int id, DataType dataType, Layout layout) {
        int rank = layout.shape().flatRank();
        long[] shape = new long[rank];
        long[] strides = new long[rank];
        long byteSize = dataType.byteSize();
        for (int i = 0; i < rank; i++) {
            shape[i] = layout.shape().flatAt(i);
            strides[i] = layout.stride().flatAt(i) * byteSize;
        }
        return formatBuffer(prefix, id, dataType, shape, strides, false);
    }

    /**
     * Formats just the type and layout information without the variable name.
     *
     * <p>Format examples:
     *
     * <ul>
     *   <li>Normal: {@code fp32[(3):(1)] contiguous}
     *   <li>Strided: {@code fp32[(3):(4)] strided}
     *   <li>Broadcast: {@code fp32[(3):(0)] broadcasted scalar}
     *   <li>Scalar: {@code fp32 scalar}
     * </ul>
     *
     * @param dataType the data type
     * @param shape the shape dimensions
     * @param strides the strides
     * @param elementBased if true, use element-based contiguity check (TIR), otherwise byte-based
     *     (LIR)
     * @return formatted type/layout string (without variable name)
     */
    public static String formatTypeLayout(
            DataType dataType, long[] shape, long[] strides, boolean elementBased) {
        StringBuilder sb = new StringBuilder();
        sb.append(formatDataType(dataType));

        // Check if scalar (empty shape)
        if (shape == null || shape.length == 0) {
            sb.append(" scalar");
            return sb.toString();
        }

        sb.append("[");
        sb.append(formatTuple(shape));
        sb.append(":");
        sb.append(formatTuple(strides));
        sb.append("]");

        // Check for broadcast pattern
        if (isBroadcast(strides)) {
            sb.append(" broadcasted scalar");
            return sb.toString();
        }

        // Check contiguity
        boolean contiguous =
                elementBased
                        ? isContiguous(shape, strides)
                        : isContiguous(shape, strides, (int) dataType.byteSize());
        if (contiguous) {
            sb.append(" contiguous");
        } else {
            sb.append(" strided");
        }

        return sb.toString();
    }

    /**
     * Formats a FP16 (IEEE 754 binary16) value.
     *
     * @param bits 16-bit representation
     * @return formatted string like "f16:0x3C00 (1.0)"
     */
    public static String formatFloat16(short bits) {
        int intBits = bits & 0xFFFF;
        float value = Float.float16ToFloat(bits);
        String hex = String.format("0x%04X", intBits);
        String valueStr = formatFloat16Value(value, bits, true);
        return "f16:" + hex + " (" + valueStr + ")";
    }

    /**
     * Formats a BF16 (bfloat16) value.
     *
     * @param bits 16-bit representation
     * @return formatted string like "bf16:0x3F80 (1.0)"
     */
    public static String formatBFloat16(short bits) {
        int intBits = bits & 0xFFFF;
        float value = BFloat16.toFloat(bits);
        String hex = String.format("0x%04X", intBits);
        String valueStr = formatFloat16Value(value, bits, false);
        return "bf16:" + hex + " (" + valueStr + ")";
    }

    /**
     * Formats a unary operator name to lowercase string.
     *
     * @param op unary operator
     * @return lowercase operator name (e.g., "negate", "reciprocal")
     */
    public static String formatUnaryOp(UnaryOperator op) {
        return op.name().toLowerCase();
    }

    /**
     * Formats a binary operator name to lowercase string.
     *
     * @param op binary operator
     * @return lowercase operator name (e.g., "add", "multiply")
     */
    public static String formatBinaryOp(BinaryOperator op) {
        return op.name().toLowerCase();
    }

    /**
     * Formats a ternary operator name to lowercase string.
     *
     * @param op ternary operator
     * @return lowercase operator name
     */
    public static String formatTernaryOp(TernaryOperator op) {
        return op.name().toLowerCase();
    }

    /**
     * Formats a reduction operator name to lowercase string.
     *
     * @param op reduction operator
     * @return lowercase operator name
     */
    public static String formatReductionOp(ReductionOperator op) {
        return op.name().toLowerCase();
    }

    /**
     * Formats a scalar value from raw bits based on data type.
     *
     * <p>Handles FP32, FP64, I32, I64, BOOL, FP16, BF16, I8, I16 data types.
     *
     * @param rawBits the raw bit representation of the scalar
     * @param dataType the data type
     * @return formatted scalar value string
     */
    public static String formatScalarValue(long rawBits, DataType dataType) {
        if (dataType.equals(DataType.FP32)) {
            return Float.intBitsToFloat((int) rawBits) + "f";
        } else if (dataType.equals(DataType.FP64)) {
            return Double.longBitsToDouble(rawBits) + "";
        } else if (dataType.equals(DataType.I32)) {
            return String.valueOf((int) rawBits);
        } else if (dataType.equals(DataType.I64)) {
            return String.valueOf(rawBits);
        } else if (dataType.equals(DataType.BOOL)) {
            return rawBits != 0 ? "true" : "false";
        } else if (dataType.equals(DataType.FP16)) {
            return formatFloat16((short) rawBits);
        } else if (dataType.equals(DataType.BF16)) {
            return formatBFloat16((short) rawBits);
        } else if (dataType.equals(DataType.I8) || dataType.equals(DataType.I16)) {
            return String.valueOf(rawBits);
        } else {
            return "0x" + Long.toHexString(rawBits);
        }
    }

    private static String formatFloat16Value(float value, short bits, boolean isFloat16) {
        if (Float.isNaN(value)) {
            return "NaN";
        }
        if (Float.isInfinite(value)) {
            return value > 0 ? "+Inf" : "-Inf";
        }
        if (value == 0.0f) {
            return (bits & 0x8000) != 0 ? "-0.0" : "0.0";
        }
        boolean isSubnormal = isFloat16Subnormal(bits, isFloat16);
        String valueStr = formatCompactFloat16(value);
        if (isSubnormal) {
            return valueStr + " (sub-normal)";
        }
        return valueStr;
    }

    private static boolean isFloat16Subnormal(short bits, boolean isFloat16) {
        int intBits = bits & 0xFFFF;
        if (isFloat16) {
            int exponent = intBits & 0x7C00;
            int mantissa = intBits & 0x03FF;
            return exponent == 0 && mantissa != 0;
        } else {
            int exponent = intBits & 0x7F80;
            int mantissa = intBits & 0x007F;
            return exponent == 0 && mantissa != 0;
        }
    }

    private static String formatCompactFloat16(float value) {
        String str = String.valueOf(value);
        if (str.contains("E") || str.contains("e")) {
            return str;
        }
        if (str.endsWith(".0")) {
            return str.substring(0, str.length() - 2);
        }
        int dotIndex = str.indexOf('.');
        if (dotIndex != -1 && str.length() - dotIndex > 6) {
            return String.format("%.5f", value);
        }
        return str;
    }

    // ==================== MLIR-Type Formatting ====================

    /**
     * Formats a tensor type in MLIR syntax.
     *
     * <p>Examples:
     *
     * <ul>
     *   <li>Scalar: {@code f32}
     *   <li>Contiguous 1D: {@code tensor<5xf32>}
     *   <li>Contiguous 2D: {@code tensor<2x3xf32>}
     *   <li>Non-contiguous: {@code tensor<5xf32, strided<[8]>>}
     * </ul>
     *
     * @param dataType the element data type
     * @param shape the shape dimensions
     * @param strides the strides in elements
     * @param isContiguous whether the layout is contiguous
     * @return MLIR tensor type string
     */
    public static String formatTensorType(
            DataType dataType, long[] shape, long[] strides, boolean isContiguous) {
        if (shape == null || shape.length == 0) {
            return formatDataType(dataType);
        }

        StringBuilder sb = new StringBuilder();
        sb.append("tensor<");

        for (int i = 0; i < shape.length; i++) {
            if (i > 0) {
                sb.append("x");
            }
            sb.append(shape[i]);
        }

        sb.append("x").append(formatDataType(dataType));

        if (!isContiguous && strides != null) {
            sb.append(", strided<[");
            for (int i = 0; i < strides.length; i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(strides[i]);
            }
            sb.append("]>");
        }

        sb.append(">");
        return sb.toString();
    }

    /**
     * Formats a memref type in MLIR syntax (for LIR).
     *
     * <p>Examples:
     *
     * <ul>
     *   <li>Contiguous 1D: {@code memref<5xf32>}
     *   <li>Contiguous 2D: {@code memref<2x3xf32>}
     *   <li>Non-contiguous: {@code memref<5xf32, strided<[8]>>}
     * </ul>
     *
     * @param dataType the element data type
     * @param shape the shape dimensions
     * @param strides the strides in bytes
     * @param elementSize the element size in bytes
     * @return MLIR memref type string
     */
    public static String formatMemRefType(
            DataType dataType,
            long[] shape,
            long[] strides,
            int elementSize,
            boolean isContiguous) {
        if (shape == null || shape.length == 0) {
            return formatDataType(dataType);
        }

        StringBuilder sb = new StringBuilder();
        sb.append("memref<");

        for (int i = 0; i < shape.length; i++) {
            if (i > 0) {
                sb.append("x");
            }
            sb.append(shape[i]);
        }

        sb.append("x").append(formatDataType(dataType));

        if (!isContiguous && strides != null) {
            sb.append(", strided<[");
            for (int i = 0; i < strides.length; i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(strides[i]);
            }
            sb.append("]>");
        }

        sb.append(">");
        return sb.toString();
    }

    /**
     * Formats an operation type suffix in MLIR syntax.
     *
     * @param dataType the result data type
     * @return MLIR type suffix (e.g., ": f32")
     */
    public static String formatOpTypeSuffix(DataType dataType) {
        return " : " + formatDataType(dataType);
    }

    /**
     * Formats an operation with explicit result types in MLIR syntax.
     *
     * @param opName the operation name (e.g., "arith.addf")
     * @param operands the operand strings
     * @param dataType the result data type
     * @return formatted MLIR operation
     */
    public static String formatMLIROp(String opName, String[] operands, DataType dataType) {
        StringBuilder sb = new StringBuilder();
        sb.append(opName).append(" ");
        for (int i = 0; i < operands.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(operands[i]);
        }
        sb.append(" : ").append(formatDataType(dataType));
        return sb.toString();
    }
}
