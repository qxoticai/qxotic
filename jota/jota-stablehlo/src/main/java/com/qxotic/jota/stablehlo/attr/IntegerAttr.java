package com.qxotic.jota.stablehlo.attr;

import com.qxotic.jota.stablehlo.type.IndexType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.Objects;

/**
 * Typed integer {@code 42 : i64}. The value is stored as a two's-complement {@code long} (unsigned
 * 64-bit values above {@link Long#MAX_VALUE} appear negative).
 */
public record IntegerAttr(long value, Type type) implements Attribute {

    public IntegerAttr {
        Objects.requireNonNull(type, "type");
        if (!(type instanceof IntegerType || type instanceof IndexType)) {
            throw new IllegalArgumentException("integer attribute needs an integer type: " + type);
        }
    }

    public static IntegerAttr i64(long value) {
        return new IntegerAttr(value, IntegerType.I64);
    }

    public static IntegerAttr i32(long value) {
        return new IntegerAttr(value, IntegerType.I32);
    }

    /** The value as printed: unsigned types print their unsigned value. */
    public String literal() {
        if (type instanceof IntegerType it && it.isUnsigned()) {
            return Long.toUnsignedString(value);
        }
        return Long.toString(value);
    }

    @Override
    public String toString() {
        return literal() + " : " + type.mlir();
    }
}
