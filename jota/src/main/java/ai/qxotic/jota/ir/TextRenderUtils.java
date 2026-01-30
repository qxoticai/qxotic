package ai.qxotic.jota.ir;

import ai.qxotic.jota.DataType;

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
}
