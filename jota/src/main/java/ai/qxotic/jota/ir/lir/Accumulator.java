package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.tir.ReductionOperator;
import java.util.Objects;

/** Declares an accumulator for reduction operations. */
public record Accumulator(String name, DataType dataType, long identityBits, ReductionOperator op)
        implements LIRNode {

    public Accumulator {
        Objects.requireNonNull(name, "name cannot be null");
        if (name.isEmpty()) {
            throw new IllegalArgumentException("name cannot be empty");
        }
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(op, "op cannot be null");
    }

    /** Creates an accumulator for SUM reduction (identity = 0). */
    public static Accumulator sum(String name, DataType dataType) {
        long identity = getZeroBits(dataType);
        return new Accumulator(name, dataType, identity, ReductionOperator.SUM);
    }

    /** Creates an accumulator for PROD reduction (identity = 1). */
    public static Accumulator product(String name, DataType dataType) {
        long identity = getOneBits(dataType);
        return new Accumulator(name, dataType, identity, ReductionOperator.PROD);
    }

    /** Creates an accumulator for MIN reduction (identity = +infinity/MAX_VALUE). */
    public static Accumulator min(String name, DataType dataType) {
        long identity = getMaxBits(dataType);
        return new Accumulator(name, dataType, identity, ReductionOperator.MIN);
    }

    /** Creates an accumulator for MAX reduction (identity = -infinity/MIN_VALUE). */
    public static Accumulator max(String name, DataType dataType) {
        long identity = getMinBits(dataType);
        return new Accumulator(name, dataType, identity, ReductionOperator.MAX);
    }

    private static long getZeroBits(DataType dataType) {
        if (dataType == DataType.FP32) {
            return Float.floatToRawIntBits(0.0f);
        } else if (dataType == DataType.FP64) {
            return Double.doubleToRawLongBits(0.0);
        }
        return 0L;
    }

    private static long getOneBits(DataType dataType) {
        if (dataType == DataType.FP32) {
            return Float.floatToRawIntBits(1.0f);
        } else if (dataType == DataType.FP64) {
            return Double.doubleToRawLongBits(1.0);
        }
        return 1L;
    }

    private static long getMaxBits(DataType dataType) {
        if (dataType == DataType.FP32) {
            return Float.floatToRawIntBits(Float.POSITIVE_INFINITY);
        } else if (dataType == DataType.FP64) {
            return Double.doubleToRawLongBits(Double.POSITIVE_INFINITY);
        } else if (dataType == DataType.I32) {
            return Integer.MAX_VALUE;
        } else if (dataType == DataType.I64) {
            return Long.MAX_VALUE;
        } else if (dataType == DataType.I8) {
            return Byte.MAX_VALUE;
        } else if (dataType == DataType.I16) {
            return Short.MAX_VALUE;
        }
        return Long.MAX_VALUE;
    }

    private static long getMinBits(DataType dataType) {
        if (dataType == DataType.FP32) {
            return Float.floatToRawIntBits(Float.NEGATIVE_INFINITY);
        } else if (dataType == DataType.FP64) {
            return Double.doubleToRawLongBits(Double.NEGATIVE_INFINITY);
        } else if (dataType == DataType.I32) {
            return Integer.MIN_VALUE;
        } else if (dataType == DataType.I64) {
            return Long.MIN_VALUE;
        } else if (dataType == DataType.I8) {
            return Byte.MIN_VALUE;
        } else if (dataType == DataType.I16) {
            return Short.MIN_VALUE;
        }
        return Long.MIN_VALUE;
    }
}
