package ai.qxotic.jota.ir.lir.v2;

import ai.qxotic.jota.DataType;

public final class SConst extends ScalarNode {
    private final long rawBits;

    SConst(int id, long rawBits, DataType dataType) {
        super(id, V2Kind.S_CONST, dataType, new V2Node[0], true, false);
        this.rawBits = rawBits;
    }

    public long rawBits() {
        return rawBits;
    }

    @Override
    public V2Node canonicalize(LirV2Graph graph) {
        return this;
    }
}
