package ai.qxotic.jota;

import java.util.List;
import java.util.Objects;

public final class TypeRules {

    // Promotion hierarchies (index = precision rank)
    private static final List<DataType> INTEGRAL_ORDER =
            List.of(DataType.I8, DataType.I16, DataType.I32, DataType.I64);

    // FP16 and BF16 are in separate hierarchies, both promote to FP32
    private static final List<DataType> FP16_HIERARCHY =
            List.of(DataType.FP16, DataType.FP32, DataType.FP64);

    private static final List<DataType> BF16_HIERARCHY =
            List.of(DataType.BF16, DataType.FP32, DataType.FP64);

    private TypeRules() {}

    /**
     * Returns the common type for numeric binary operations.
     *
     * <p>Promotion rules:
     *
     * <ul>
     *   <li>Same numeric type → that type
     *   <li>Integral: I8 &lt; I16 &lt; I32 &lt; I64 (wider wins)
     *   <li>Float: FP16 &lt; FP32 &lt; FP64, BF16 &lt; FP32 &lt; FP64 (wider wins)
     *   <li>FP16 + BF16 → Error (incompatible 16-bit formats)
     *   <li>BOOL is not numeric-promotable
     *   <li>Lossless int→float: I8 &lt; FP16/BF16/FP32/FP64, I16 &lt; FP32/FP64, I32 &lt; FP64
     *   <li>I64 + Float → Error (no float can represent all I64 values)
     *   <li>Quantized types → Error (require explicit cast)
     * </ul>
     *
     * @throws IllegalArgumentException if types cannot be promoted
     */
    public static DataType promote(DataType left, DataType right) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");

        if (left == DataType.BOOL || right == DataType.BOOL) {
            throw new IllegalArgumentException(
                    "BOOL cannot be promoted with numeric types: " + left + " vs " + right);
        }

        if (left == right) {
            return left;
        }

        if (isQuantized(left) || isQuantized(right)) {
            throw new IllegalArgumentException(
                    "quantized types require explicit cast: " + left + " vs " + right);
        }

        // Both integral → wider wins
        if (left.isIntegral() && right.isIntegral()) {
            return promoteIntegral(left, right);
        }

        // Both floating-point → check compatible hierarchies
        if (left.isFloatingPoint() && right.isFloatingPoint()) {
            return promoteFloat(left, right);
        }

        // Mixed float/integral → check for lossless promotion
        return promoteIntegralToFloat(left, right);
    }

    /**
     * Returns the common type for comparison operations.
     *
     * <p>Special case: BOOL and BOOL compares as BOOL. All other cases follow numeric promotion
     * rules.
     */
    public static DataType promoteForComparison(DataType left, DataType right) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");

        if (left == DataType.BOOL && right == DataType.BOOL) {
            return DataType.BOOL;
        }
        return promote(left, right);
    }

    private static DataType promoteFloat(DataType left, DataType right) {
        // Check FP16 hierarchy
        int leftFp16 = FP16_HIERARCHY.indexOf(left);
        int rightFp16 = FP16_HIERARCHY.indexOf(right);

        // Check BF16 hierarchy
        int leftBf16 = BF16_HIERARCHY.indexOf(left);
        int rightBf16 = BF16_HIERARCHY.indexOf(right);

        // Both in FP16 hierarchy (FP16, FP32, FP64)
        if (leftFp16 >= 0 && rightFp16 >= 0) {
            return leftFp16 >= rightFp16 ? left : right;
        }

        // Both in BF16 hierarchy (BF16, FP32, FP64)
        if (leftBf16 >= 0 && rightBf16 >= 0) {
            return leftBf16 >= rightBf16 ? left : right;
        }

        // One is FP16, other is BF16 → incompatible
        if ((left == DataType.FP16 && right == DataType.BF16)
                || (left == DataType.BF16 && right == DataType.FP16)) {
            throw new IllegalArgumentException(
                    "incompatible 16-bit float types require explicit cast: "
                            + left
                            + " vs "
                            + right);
        }

        // Should not reach here for valid float types
        throw new IllegalArgumentException(
                "unsupported float type combination: " + left + " vs " + right);
    }

    private static DataType promoteIntegral(DataType left, DataType right) {
        int leftIndex = INTEGRAL_ORDER.indexOf(left);
        int rightIndex = INTEGRAL_ORDER.indexOf(right);
        if (leftIndex == -1 || rightIndex == -1) {
            throw new IllegalArgumentException(
                    "unsupported integral type combination: " + left + " vs " + right);
        }
        return leftIndex >= rightIndex ? left : right;
    }

    /**
     * Promotes an integral type to a float type if lossless.
     *
     * <p>Lossless promotions based on mantissa bits:
     *
     * <ul>
     *   <li>I8 (8-bit) → FP16 (11-bit mantissa), BF16 (8-bit), FP32 (24-bit), FP64 (53-bit)
     *   <li>I16 (16-bit) → FP32 (24-bit), FP64 (53-bit)
     *   <li>I32 (32-bit) → FP64 (53-bit)
     *   <li>I64 (64-bit) → Error (no standard float has enough precision)
     * </ul>
     */
    private static DataType promoteIntegralToFloat(DataType left, DataType right) {
        DataType integral = left.isIntegral() ? left : right;
        DataType floating = left.isFloatingPoint() ? left : right;

        // I8 can promote to any float type
        if (integral == DataType.I8) {
            return floating;
        }

        // I16 can promote to FP32 or FP64
        if (integral == DataType.I16) {
            if (floating == DataType.FP32 || floating == DataType.FP64) {
                return floating;
            }
            throw new IllegalArgumentException(
                    "I16 cannot be losslessly promoted to " + floating + ", use FP32 or FP64");
        }

        // I32 can only promote to FP64
        if (integral == DataType.I32) {
            if (floating == DataType.FP64) {
                return floating;
            }
            throw new IllegalArgumentException(
                    "I32 cannot be losslessly promoted to " + floating + ", use FP64");
        }

        // I64 cannot be losslessly promoted to any float
        throw new IllegalArgumentException(
                "I64 cannot be losslessly promoted to any float type: " + left + " vs " + right);
    }

    private static boolean isQuantized(DataType dataType) {
        return dataType == DataType.Q8_0 || dataType == DataType.Q4_0;
    }
}
