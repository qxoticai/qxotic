package ai.qxotic.jota;

import java.util.List;
import java.util.Objects;

public final class TypeRules {

    private static final List<DataType> INTEGRAL_ORDER = List.of(
            DataType.I8,
            DataType.I16,
            DataType.I32,
            DataType.I64
    );

    private TypeRules() {
    }

    public static DataType promote(DataType left, DataType right) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");

        if (left == right) {
            return left;
        }

        if (isQuantized(left) || isQuantized(right)) {
            throw new IllegalArgumentException("quantized types require explicit target");
        }

        boolean leftBool = isBoolean(left);
        boolean rightBool = isBoolean(right);

        if (leftBool && rightBool) {
            return left;
        }
        if (leftBool && right.isIntegral()) {
            return right;
        }
        if (rightBool && left.isIntegral()) {
            return left;
        }
        if ((leftBool && right.isFloatingPoint()) || (rightBool && left.isFloatingPoint())) {
            return promoteFloat(left, right);
        }

        if (left.isFloatingPoint() || right.isFloatingPoint()) {
            return promoteFloat(left, right);
        }

        if (left.isIntegral() && right.isIntegral()) {
            return promoteIntegral(left, right);
        }

        throw new IllegalArgumentException("unsupported data type combination: " + left + ", " + right);
    }

    private static DataType promoteFloat(DataType left, DataType right) {
        if (left == DataType.FP64 || right == DataType.FP64) {
            return DataType.FP64;
        }
        if (left == DataType.FP32 || right == DataType.FP32) {
            return DataType.FP32;
        }
        return DataType.FP32;
    }

    private static DataType promoteIntegral(DataType left, DataType right) {
        int leftIndex = INTEGRAL_ORDER.indexOf(left);
        int rightIndex = INTEGRAL_ORDER.indexOf(right);
        if (leftIndex == -1 || rightIndex == -1) {
            throw new IllegalArgumentException("unsupported integral type combination: " + left + ", " + right);
        }
        return leftIndex >= rightIndex ? left : right;
    }

    private static boolean isQuantized(DataType dataType) {
        return dataType == DataType.Q8_0 || dataType == DataType.Q4_0;
    }

    private static boolean isBoolean(DataType dataType) {
        return "bool".equals(dataType.name());
    }
}
