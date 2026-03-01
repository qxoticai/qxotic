package com.qxotic.jota.runtime;

import com.qxotic.jota.BFloat16;
import com.qxotic.jota.DataType;

/** Scalar kernel argument passed by value. */
public record ScalarArg(long rawBits, DataType dataType) {

    public static ScalarArg ofFloat(float value) {
        return new ScalarArg(Float.floatToRawIntBits(value), DataType.FP32);
    }

    public static ScalarArg ofDouble(double value) {
        return new ScalarArg(Double.doubleToRawLongBits(value), DataType.FP64);
    }

    public static ScalarArg ofInt(int value) {
        return new ScalarArg(value, DataType.I32);
    }

    public static ScalarArg ofLong(long value) {
        return new ScalarArg(value, DataType.I64);
    }

    public static ScalarArg ofBool(boolean value) {
        return new ScalarArg(value ? 1L : 0L, DataType.BOOL);
    }

    public static ScalarArg ofFloat16(float value) {
        return new ScalarArg(Float.floatToFloat16(value), DataType.FP16);
    }

    public static ScalarArg ofBFloat16(float value) {
        return new ScalarArg(BFloat16.fromFloat(value), DataType.BF16);
    }
}
