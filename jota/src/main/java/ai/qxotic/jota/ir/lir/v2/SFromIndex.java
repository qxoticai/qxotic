package ai.qxotic.jota.ir.lir.v2;

import ai.qxotic.jota.DataType;

public final class SFromIndex extends ScalarNode {
    SFromIndex(int id, V2Node indexExpr, DataType dataType) {
        super(id, V2Kind.S_FROM_INDEX, dataType, new V2Node[] {indexExpr}, true, false);
    }

    public V2Node indexExpr() {
        return inputs()[0];
    }

    @Override
    public V2Node canonicalize(LirV2Graph graph) {
        return this;
    }
}
