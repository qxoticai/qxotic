package com.qxotic.jota.stablehlo.attr;

/**
 * A StableHLO / MLIR attribute (spec "Constants"): the static metadata of operations.
 *
 * <p>{@link #toString()} of every implementation returns the MLIR assembly spelling.
 */
public sealed interface Attribute
        permits BoolAttr,
                IntegerAttr,
                FloatAttr,
                StringAttr,
                UnitAttr,
                ArrayAttr,
                DictionaryAttr,
                DenseArrayAttr,
                DenseElementsAttr,
                TypeAttr,
                SymbolRefAttr,
                EnumAttr,
                DialectAttr,
                OpaqueAttr {}
