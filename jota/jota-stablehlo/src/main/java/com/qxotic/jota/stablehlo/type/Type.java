package com.qxotic.jota.stablehlo.type;

/**
 * A StableHLO type (spec "Types"): value types (tensor, quantized tensor, token, tuple, future) and
 * non-value types (element types, function type).
 *
 * <p>{@link #toString()} of every implementation returns the MLIR assembly spelling.
 */
public sealed interface Type
        permits ElementType,
                TensorType,
                TokenType,
                TupleType,
                FunctionType,
                FutureType,
                OpaqueType {

    /** The MLIR assembly spelling of this type. */
    String mlir();
}
