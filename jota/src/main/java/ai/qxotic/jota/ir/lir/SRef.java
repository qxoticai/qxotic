package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;

public final class SRef extends ScalarNode {
    private final String name;

    SRef(int id, String name, DataType dataType) {
        super(id, LIRExprKind.S_REF, dataType, new LIRExprNode[0], true, false);
        this.name = name;
    }

    public String name() {
        return name;
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        return this;
    }
}
