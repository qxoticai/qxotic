package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;

public final class SInput extends ScalarNode {
    private final int inputId;

    SInput(int id, int inputId, DataType dataType) {
        super(id, LIRExprKind.S_INPUT, dataType, new LIRExprNode[0], true, false);
        this.inputId = inputId;
    }

    public int inputId() {
        return inputId;
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        return this;
    }
}
