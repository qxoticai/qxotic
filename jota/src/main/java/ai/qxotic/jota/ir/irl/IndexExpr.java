package ai.qxotic.jota.ir.irl;

import ai.qxotic.jota.DataType;

/** Sealed interface for index expressions in IR-L. Indices are always 64-bit longs. */
public sealed interface IndexExpr extends IRLNode permits IndexVar, IndexConst, IndexBinary {

    @Override
    default DataType dataType() {
        return DataType.I64;
    }
}
