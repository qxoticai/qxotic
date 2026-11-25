package com.qxotic.jota.stablehlo.type;

/** MLIR builtin {@code index} type (shape computations, e.g. {@code tensor<2xindex>} operands). */
public record IndexType() implements ElementType {

    public static final IndexType INSTANCE = new IndexType();

    /** Index values are stored as 64-bit integers in dense buffers. */
    @Override
    public int storageBytes() {
        return 8;
    }

    @Override
    public String mlir() {
        return "index";
    }

    @Override
    public String toString() {
        return mlir();
    }
}
