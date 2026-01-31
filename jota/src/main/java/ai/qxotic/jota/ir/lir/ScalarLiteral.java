package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.BFloat16;
import ai.qxotic.jota.DataType;
import java.util.Objects;

/**
 * Literal scalar value stored as raw bits. This represents a compile-time known constant that is
 * inlined in generated code.
 *
 * <p>Unlike {@link ScalarInput} which represents runtime scalar parameters, ScalarLiteral values
 * are known at IR construction time and can be directly embedded in the generated kernel code.
 */
public record ScalarLiteral(long rawBits, DataType dataType) implements ScalarExpr {

    public ScalarLiteral {
        Objects.requireNonNull(dataType, "dataType cannot be null");
    }

    public static ScalarLiteral ofRawBits(long rawBits, DataType dataType) {
        return new ScalarLiteral(rawBits, dataType);
    }

    /** Creates a float literal. */
    public static ScalarLiteral ofFloat(float value) {
        return new ScalarLiteral(Float.floatToRawIntBits(value), DataType.FP32);
    }

    /** Creates a double literal. */
    public static ScalarLiteral ofDouble(double value) {
        return new ScalarLiteral(Double.doubleToRawLongBits(value), DataType.FP64);
    }

    /** Creates an int literal. */
    public static ScalarLiteral ofInt(int value) {
        return new ScalarLiteral(value, DataType.I32);
    }

    /** Creates a long literal. */
    public static ScalarLiteral ofLong(long value) {
        return new ScalarLiteral(value, DataType.I64);
    }

    /** Creates a boolean literal. */
    public static ScalarLiteral ofBool(boolean value) {
        return new ScalarLiteral(value ? 1 : 0, DataType.BOOL);
    }

    /**
     * Returns the value as a float. Works for FP16, BF16, and FP32 types.
     *
     * @throws IllegalStateException if dataType is not a 32-bit or smaller floating-point type
     */
    public float asFloat() {
        if (dataType == DataType.FP32) {
            return Float.intBitsToFloat((int) rawBits);
        } else if (dataType == DataType.FP16) {
            return Float.float16ToFloat((short) rawBits);
        } else if (dataType == DataType.BF16) {
            return BFloat16.toFloat((short) rawBits);
        }
        throw new IllegalStateException(
                "Cannot extract float from " + dataType + ", expected FP16, BF16, or FP32");
    }

    /**
     * Returns the value as a double. Works only for FP64 type.
     *
     * @throws IllegalStateException if dataType is not FP64
     */
    public double asDouble() {
        if (dataType == DataType.FP64) {
            return Double.longBitsToDouble(rawBits);
        }
        throw new IllegalStateException("Cannot extract double from " + dataType + ", expected FP64");
    }

    /**
     * Returns the value as an int. Works for I8, I16, and I32 types.
     *
     * @throws IllegalStateException if dataType is not a 32-bit or smaller integral type
     */
    public int asInt() {
        if (dataType == DataType.I32) {
            return (int) rawBits;
        } else if (dataType == DataType.I16) {
            return (short) rawBits;
        } else if (dataType == DataType.I8) {
            return (byte) rawBits;
        }
        throw new IllegalStateException(
                "Cannot extract int from " + dataType + ", expected I8, I16, or I32");
    }

    /**
     * Returns the value as a long. Works for all integral types (I8, I16, I32, I64).
     *
     * @throws IllegalStateException if dataType is not an integral type
     */
    public long asLong() {
        if (dataType == DataType.I64) {
            return rawBits;
        } else if (dataType == DataType.I32) {
            return (int) rawBits;
        } else if (dataType == DataType.I16) {
            return (short) rawBits;
        } else if (dataType == DataType.I8) {
            return (byte) rawBits;
        }
        throw new IllegalStateException(
                "Cannot extract long from " + dataType + ", expected I8, I16, I32, or I64");
    }

    /**
     * Returns the value as a boolean. Works only for BOOL type.
     *
     * @throws IllegalStateException if dataType is not BOOL
     */
    public boolean asBool() {
        if (dataType == DataType.BOOL) {
            return rawBits != 0;
        }
        throw new IllegalStateException("Cannot extract boolean from " + dataType + ", expected BOOL");
    }
}
