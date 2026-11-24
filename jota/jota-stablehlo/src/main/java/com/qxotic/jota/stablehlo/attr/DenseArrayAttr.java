package com.qxotic.jota.stablehlo.attr;

import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.Arrays;
import java.util.Objects;

/**
 * {@code array<i64: 1, 2>} / {@code array<i1: true, false>}: a flat array of scalars, stored as raw
 * bits (integers sign-extended, floats as their format's bit pattern).
 */
public record DenseArrayAttr(Type elementType, long[] values) implements Attribute {

    public DenseArrayAttr {
        Objects.requireNonNull(elementType, "elementType");
        if (!(elementType instanceof IntegerType || elementType instanceof FloatType)) {
            throw new IllegalArgumentException("dense array element type: " + elementType);
        }
        values = values.clone();
    }

    public static DenseArrayAttr i64(long... values) {
        return new DenseArrayAttr(IntegerType.I64, values);
    }

    public static DenseArrayAttr i1(boolean... values) {
        long[] bits = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = values[i] ? 1 : 0;
        }
        return new DenseArrayAttr(IntegerType.I1, bits);
    }

    @Override
    public long[] values() {
        return values.clone();
    }

    public int size() {
        return values.length;
    }

    public long get(int i) {
        return values[i];
    }

    public boolean[] booleans() {
        boolean[] b = new boolean[values.length];
        for (int i = 0; i < b.length; i++) {
            b[i] = values[i] != 0;
        }
        return b;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof DenseArrayAttr other
                && elementType.equals(other.elementType)
                && Arrays.equals(values, other.values);
    }

    @Override
    public int hashCode() {
        return 31 * elementType.hashCode() + Arrays.hashCode(values);
    }

    /** The element list without the {@code array<T: >} wrapper. */
    public String elementsLiteral() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(elementLiteral(values[i]));
        }
        return sb.toString();
    }

    private String elementLiteral(long v) {
        if (elementType instanceof IntegerType it) {
            if (it.isBoolean()) {
                return Boolean.toString(v != 0);
            }
            return it.isUnsigned() ? Long.toUnsignedString(v) : Long.toString(v);
        }
        return FloatAttr.literal(v, ((FloatType) elementType).format());
    }

    @Override
    public String toString() {
        if (values.length == 0) {
            return "array<" + elementType.mlir() + ">";
        }
        return "array<" + elementType.mlir() + ": " + elementsLiteral() + ">";
    }
}
