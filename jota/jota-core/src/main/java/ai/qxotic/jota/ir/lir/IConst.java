package ai.qxotic.jota.ir.lir;

public final class IConst extends IndexNode {
    private final long value;

    IConst(int id, long value) {
        super(id, LIRExprKind.I_CONST, new LIRExprNode[0], true, false);
        this.value = value;
    }

    public long value() {
        return value;
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        return this;
    }
}
