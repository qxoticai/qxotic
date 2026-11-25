package com.qxotic.jota.stablehlo.type;

/** Floating-point element type. */
public record FloatType(FloatFormat format) implements ElementType {

    public static final FloatType F16 = new FloatType(FloatFormat.F16);
    public static final FloatType BF16 = new FloatType(FloatFormat.BF16);
    public static final FloatType F32 = new FloatType(FloatFormat.F32);
    public static final FloatType F64 = new FloatType(FloatFormat.F64);

    public FloatType {
        java.util.Objects.requireNonNull(format, "format");
    }

    @Override
    public int storageBytes() {
        return format.storageBytes();
    }

    @Override
    public String mlir() {
        return format.mlirName();
    }

    @Override
    public String toString() {
        return mlir();
    }
}
