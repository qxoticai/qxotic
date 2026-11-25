package com.qxotic.jota.stablehlo.type;

import java.util.Objects;

/** Complex element type {@code complex<f32>}: real and imaginary parts of the same float type. */
public record ComplexType(FloatType elementType) implements ElementType {

    public ComplexType {
        Objects.requireNonNull(elementType, "elementType");
    }

    @Override
    public int storageBytes() {
        return 2 * elementType.storageBytes();
    }

    @Override
    public String mlir() {
        return "complex<" + elementType.mlir() + ">";
    }

    @Override
    public String toString() {
        return mlir();
    }
}
