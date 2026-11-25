package com.qxotic.jota.stablehlo.type;

/** Tensor element types: boolean/integer, floating-point, complex and quantized. */
public sealed interface ElementType extends Type
        permits IntegerType, FloatType, ComplexType, QuantizedType, IndexType {

    /** Bytes used per element in dense element buffers (i1 uses one byte). */
    int storageBytes();
}
