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
    public static Accumulator sum(String name, DataType dtype) {
        long identity = getZeroBits(dtype);
        return new Accumulator(name, dtype, identity, ReductionOperator.SUM);
    }

    /** Creates an accumulator for PROD reduction (identity = 1). */
    public static Accumulator prod(String name, DataType dtype) {
        long identity = getOneBits(dtype);
        return new Accumulator(name, dtype, identity, ReductionOperator.PROD);
    }

    /** Creates an accumulator for MIN reduction (identity = +infinity/MAX_VALUE). */
    public static Accumulator min(String name, DataType dtype) {
        long identity = getMaxBits(dtype);
        return new Accumulator(name, dtype, identity, ReductionOperator.MIN);
    }

    /** Creates an accumulator for MAX reduction (identity = -infinity/MIN_VALUE). */
    public static Accumulator max(String name, DataType dtype) {
        long identity = getMinBits(dtype);
        return new Accumulator(name, dtype, identity, ReductionOperator.MAX);
    }

    private static long getZeroBits(DataType dtype) {
        if (dtype == DataType.FP32) {
            return Float.floatToRawIntBits(0.0f);
        } else if (dtype == DataType.FP64) {
            return Double.doubleToRawLongBits(0.0);
        }
        return 0L;
    }

    private static long getOneBits(DataType dtype) {
        if (dtype == DataType.FP32) {
            return Float.floatToRawIntBits(1.0f);
        } else if (dtype == DataType.FP64) {
            return Double.doubleToRawLongBits(1.0);
        }
        return 1L;
    }

    private static long getMaxBits(DataType dtype) {
        if (dtype == DataType.FP32) {
            return Float.floatToRawIntBits(Float.MAX_VALUE);
        } else if (dtype == DataType.FP64) {
            return Double.doubleToRawLongBits(Double.MAX_VALUE);
        } else if (dtype == DataType.I32) {
            return Integer.MAX_VALUE;
        } else if (dtype == DataType.I64) {
            return Long.MAX_VALUE;
        } else if (dtype == DataType.I8) {
            return Byte.MAX_VALUE;
        } else if (dtype == DataType.I16) {
            return Short.MAX_VALUE;
        }
        return Long.MAX_VALUE;
    }

    private static long getMinBits(DataType dtype) {
        if (dtype == DataType.FP32) {
            return Float.floatToRawIntBits(-Float.MAX_VALUE);
        } else if (dtype == DataType.FP64) {
            return Double.doubleToRawLongBits(-Double.MAX_VALUE);
        } else if (dtype == DataType.I32) {
            return Integer.MIN_VALUE;
        } else if (dtype == DataType.I64) {
            return Long.MIN_VALUE;
        } else if (dtype == DataType.I8) {
            return Byte.MIN_VALUE;
        } else if (dtype == DataType.I16) {
            return Short.MIN_VALUE;
        }
        return Long.MIN_VALUE;
    }
}
