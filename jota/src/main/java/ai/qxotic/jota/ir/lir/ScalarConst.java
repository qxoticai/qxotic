package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import java.util.Objects;

/** Constant scalar value stored as raw bits. */
public record ScalarConst(long rawBits, DataType dataType) implements ScalarExpr {

    public ScalarConst {
        Objects.requireNonNull(dataType, "dataType cannot be null");
    }

    public static ScalarConst ofRawBits(long rawBits, DataType dataType) {
        return new ScalarConst(rawBits, DataType.FP32);
    }

    /** Creates a float constant. */
    public static ScalarConst ofFloat(float value) {
        return new ScalarConst(Float.floatToRawIntBits(value), DataType.FP32);
    }

    /** Creates a double constant. */
    public static ScalarConst ofDouble(double value) {
        return new ScalarConst(Double.doubleToRawLongBits(value), DataType.FP64);
    }

    /** Creates an int constant. */
    public static ScalarConst ofInt(int value) {
        return new ScalarConst(value, DataType.I32);
    }

    /** Creates a long constant. */
    public static ScalarConst ofLong(long value) {
        return new ScalarConst(value, DataType.I64);
    }

    /** Creates a boolean constant. */
    public static ScalarConst ofBool(boolean value) {
        return new ScalarConst(value ? 1 : 0, DataType.BOOL);
    }

    /** Returns the value as a float (assumes dataType is FP32). */
    public float asFloat() {
        return Float.intBitsToFloat((int) rawBits);
    }

    /** Returns the value as a double (assumes dataType is FP64). */
    public double asDouble() {
        return Double.longBitsToDouble(rawBits);
    }

    /** Returns the value as an int. */
    public int asInt() {
        return (int) rawBits;
    }

    /** Returns the value as a long. */
    public long asLong() {
        return rawBits;
    }

    /** Returns the value as a boolean. */
    public boolean asBool() {
        return rawBits != 0;
    }
}
