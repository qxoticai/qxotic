package ai.qxotic.jota.ir.lir.v2;

import ai.qxotic.jota.DataType;

public final class SInput extends ScalarNode {
    private final int inputId;

    SInput(int id, int inputId, DataType dataType) {
        super(id, V2Kind.S_INPUT, dataType, new V2Node[0], true, false);
        this.inputId = inputId;
    }

    public int inputId() {
        return inputId;
    }

    @Override
    public V2Node canonicalize(LirV2Graph graph) {
        return this;
    }
}
