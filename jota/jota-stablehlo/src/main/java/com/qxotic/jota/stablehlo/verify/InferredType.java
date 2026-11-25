package com.qxotic.jota.stablehlo.verify;

import com.qxotic.jota.stablehlo.type.Type;
import java.util.Objects;

/**
 * A result type found by {@link TypeInference}.
 *
 * <p>Some ops determine only the shape of their result ({@code convert}, {@code dot_general},
 * {@code convolution}, {@code uniform_quantize}): their element type is a free choice of the
 * producer. Such a result is a tensor type carrying the operand's element type as a placeholder,
 * with {@code elementTypeInferred} false; {@link #isCompatibleWith} then compares shapes only.
 *
 * @param type the inferred type; dimensions may be dynamic and carry bounds
 * @param elementTypeInferred false when only the shape (and bounds) is determined
 */
public record InferredType(Type type, boolean elementTypeInferred) {

    public InferredType {
        Objects.requireNonNull(type, "type");
    }

    public static InferredType of(Type type) {
        return new InferredType(type, true);
    }

    /** Shape-only inference: {@code placeholder}'s element type is not determined. */
    public static InferredType shapeOnly(Type placeholder) {
        return new InferredType(placeholder, false);
    }

    /** Whether {@code actual} is a valid result type for this inference. */
    public boolean isCompatibleWith(Type actual) {
        return elementTypeInferred
                ? TypeInference.isCompatible(type, actual)
                : TypeInference.isShapeCompatible(type, actual);
    }

    @Override
    public String toString() {
        return elementTypeInferred ? type.mlir() : type.mlir() + " (shape only)";
    }
}
